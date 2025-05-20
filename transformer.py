import torch
import torch.nn as nn
from config import Config
from patch_extractor import PatchEmbedding
from process_wav import convert_wav_to_mel_spectrogram, visualize_spectrogram
import numpy as np
from typing import Optional, Tuple, Dict
from masking import Mask
from positional_encoding import SinusoidalPositionalEncoding

# Encoder del Vision Transformer
class MAE_Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim
        self.num_heads = config.num_enc_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # dimensione di ogni head
        self.scale = self.head_dim ** -0.5 # 1/sqrt(head_dim) mateniamo magntude simile per ogni head
        self.dropout = config.enc_attention_dropout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [B, num_patches, embed_dim] -> [B, num_patches, all_head_size]
        b, seq_len, _ = hidden_states.size() # seq_len è il numero di patches
        
        # [B, num_patches, all_head_size]
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # [B, num_patches, all_head_size] -> [B, num_heads, num_patches, head_dim]
        query = query.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, num_heads, num_patches, head_dim] * [B, num_heads, head_dim, num_patches] -> [B, num_heads, num_patches, num_patches]
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale 
        
        if attn_weights.size() != (b, self.num_heads, seq_len, seq_len):
            raise ValueError(f"attn_weights shape {attn_weights.size()} is not equal to expected shape {(b, self.num_heads, seq_len, seq_len)}")
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(query.dtype) # vediamo le relazioni tra i token in termini di probabiltà applicando la softmax by row
        # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # dropout per evitare overfitting
        
        # [B, num_heads, num_patches, head_dim] -> [B, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value) # [B, num_heads, num_patches, head_dim] -> [B, num_heads, num_patches, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, seq_len, self.embed_dim) # [B, num_heads, num_patches, head_dim] 
        attn_output = attn_output.reshape(b, seq_len, self.embed_dim) # [B, num_patches, embed_dim]
        
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

# L'input è una lista di emedding, uno per ogni patch. Ogni embedding cattura le informazioni di una specifica patch.
# L output della multi-head attention avra la stessa dimensione dell input ma ogni embedding conterra informaizioni anche delle altre patch.
# NOTA: negli LLM l'ouput della multi-head attention ha la stessa dimensione dell input, ma ogni embedding i-esimo conterra le informazioni di tutti i token j-esimi con j < i
class MAE_Encoder(nn.Module):
    def __init__(self, config: Config, decoder=False):
        super().__init__()
        self.config = config

        if decoder:
            self.layers = nn.ModuleList(
                [MAE_EncoderLayer(config) for _ in range(config.num_dec_hidden_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [MAE_EncoderLayer(config) for _ in range(config.num_enc_hidden_layers)]
            )

    def forward(
        self,
        not_masked_patches: torch.Tensor # indici delle patch non mascherate
    ) -> torch.Tensor:
        # not_masked_patches: [Batch_Size, Num_Patches, Embed_Dim]

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            not_masked_patches_encoded = encoder_layer(not_masked_patches)

        return not_masked_patches_encoded

class MAE_MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config.enc_embed_dim, config.enc_embed_dim*4) # tipicamente è 4x o 3x la hidden size
        self.linear2 = nn.Linear(config.enc_mlp_layer_dim, config.enc_embed_dim) # ricomprimiamo nell hidden size.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, num_patches, embed_dim] -> [B, num_patches, intermeditate_size]
        hidden_states = self.linear1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate="tanh") # non linearity per imparare funzioni complesse
        # [B, num_patches, intermeditate_size] -> [B, num_patches, enc_embed_dim]
        hidden_states = self.linear2(hidden_states) 
        return hidden_states

class MAE_EncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim
        self.self_attn = MAE_Attention(config)  
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.enc_layer_norm_eps)
        self.mlp = MAE_MLP(config) # feed forward network
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.enc_layer_norm_eps)


    def forward(self, not_masked_patches: torch.Tensor) -> torch.Tensor:
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        residual = not_masked_patches
        # layer norm 1 - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        not_masked_patches = self.layer_norm1(not_masked_patches)
        # multi head attention - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        not_masked_patches_after_attn, _ = self.self_attn(not_masked_patches)
        # residual connection - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        not_masked_patches_after_attn = not_masked_patches_after_attn + residual
        residual = not_masked_patches_after_attn
        # layer norm 2 - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        not_masked_patches_after_attn = self.layer_norm2(not_masked_patches_after_attn)
        # feed forward network - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        not_masked_patches_after_mlp = self.mlp(not_masked_patches_after_attn)
        # residual connection  
        not_masked_patches_after_mlp = not_masked_patches_after_mlp + residual

        return not_masked_patches_after_mlp 

    
class VisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.enc_embed_dim = config.enc_embed_dim
        self.dec_embed_dim = config.dec_embed_dim

        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)
        self.patch_embeddings = PatchEmbedding(config) # [B, num_patches, embed_dim]  
        self.mask = Mask(config) 
        self.positional_embeddings_before_encoder = SinusoidalPositionalEncoding(embed_dim=self.enc_embed_dim)
        self.encoder = MAE_Encoder(config) # sara una lista di TransformerLayer 
        self.post_enc_layernorm = nn.LayerNorm(config.enc_embed_dim, eps=config.enc_layer_norm_eps)
        self.linear = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
        
        self.decoder_mask_emb = nn.Parameter(torch.FloatTensor(config.dec_embed_dim).uniform_()) # è l'embedding che rappresenta la patch mascherata e dovrebbe essere appresa durante il training
        self.positional_embeddings_before_decoder = SinusoidalPositionalEncoding(embed_dim=self.dec_embed_dim)
        self.decoder = MAE_Encoder(config, decoder=True)
        
        self.final_proj_reconstruction = nn.Linear(
            config.dec_embed_dim,
            config.patch_size[0] * config.patch_size[1]
        )
        self.final_proj_classification = nn.Linear(
            config.dec_embed_dim,
            config.patch_size[0] * config.patch_size[1]
        )
    
    def forward(self, spectrogram_values:torch.Tensor) -> torch.Tensor:
        B = spectrogram_values.shape[0]
        spectrogram_values = self.batch_norm(spectrogram_values) * 0.5
        patch_embeddings, patch_embeddings_after_proj =  self.patch_embeddings(spectrogram_values) # [B, C, H, W] -> [B, num_patches, patch_embed_dim] 
        masked_patch_embeddings, masked_indices, unmasked_indices = self.mask(patch_embeddings_after_proj) 
        positional_embeddings = self.positional_embeddings_before_encoder(masked_patch_embeddings)
        masked_patch_embeddings_with_pe = masked_patch_embeddings + positional_embeddings

        encoder_input = []
        for b in range(B):
            visible_b = masked_patch_embeddings_with_pe[b, unmasked_indices[b], :].float()
            encoder_input.append(visible_b)
        encoder_input = torch.stack(encoder_input, dim=0)

        encoder_output = self.encoder(encoder_input)
        encoder_output = self.post_enc_layernorm(encoder_output)

        if self.enc_embed_dim != self.dec_embed_dim:
            encoder_output = self.linear(encoder_output) # [B, num_patches, enc_embed_dim] -> [B, num_patches, dec_embed_dim]

        # Add masked patches to encoder output
        N_patches = masked_patch_embeddings_with_pe.size(1)
        patch_embeddings_recostructed = torch.zeros(
            (B, N_patches, self.dec_embed_dim), 
            device=encoder_output.device, 
            dtype=encoder_output.dtype)
        
        mask_indices_bool = torch.zeros(
            (B, N_patches), 
            device=encoder_output.device, 
            dtype=torch.bool
        )

        for b in range(B):
            # encoder output shape: [B, patch_non_mascherate, patch_dim]
            patch_embeddings_recostructed[b, unmasked_indices[b], :] = encoder_output[b]
            mask_tokens = self.decoder_mask_emb.unsqueeze(0).expand(len(masked_indices[0]), -1)
            patch_embeddings_recostructed[b, masked_indices[b], :] = mask_tokens
            
            mask_indices_bool[b, masked_indices[b]] = True
        
        patch_embeddings_recostructed_with_pe = self.positional_embeddings_before_decoder(patch_embeddings_recostructed) + patch_embeddings_recostructed 
        
        decoder_output = self.decoder(patch_embeddings_recostructed_with_pe) 
        # print("Decoder output shape:", decoder_output.shape) 
        
        # recupero patch che avevo mascherato
        batch_idx = torch.arange(B, device=decoder_output.device).unsqueeze(-1)  # → [B, 1]
        decoded_masked_patches = decoder_output[batch_idx, masked_indices, :]  
        # print("masked_patches_after_decoder shape: ", masked_patches_after_decoder.shape)
        recostruction_logits = self.final_proj_reconstruction(decoded_masked_patches) # [B, N_masked, patch_size]
        classification_logits = self.final_proj_classification(decoded_masked_patches) # [B, N_masked, patch_size]

        patch_dim = patch_embeddings.size(-1) 
        # print(f"patch_dim: {patch_dim}")

        target_patches = torch.zeros(
            (B, len(masked_indices[0]), patch_dim),
            device=patch_embeddings.device,
            dtype=patch_embeddings.dtype
        )
        
        for b in range(B):
            target_patches[b] = patch_embeddings[b, masked_indices[b], :]

        return {
            "encoder_output": encoder_output,
            "decoder_output": decoder_output,
            "recon_logits": recostruction_logits,
            "class_logits": classification_logits,
            "target_patches": target_patches,

        }
    
def test_model():
    config = Config()
    model = VisionTransformer(config)
    print(model)
    # print(model.patch_embeddings)
    # print(model.encoder)
    # print(model.post_enc_layernorm)

    # spectrogram_np = np.load("C:/Users/admin/Desktop/VS Code/VisualTransformer/datasets/numpy/balanced_train_segments/EX_-_6RxZyi30Q.npy")
    # spectrogram = torch.from_numpy(spectrogram_np)
    spectrogram_np = np.arange(0, 1000*128).reshape(1000, 128)
    spectrogram = torch.tensor(spectrogram_np, dtype=torch.float32)
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0) # [B, C, W, H]
    spectrogram = spectrogram.permute(0, 1, 3, 2) # [B, C, H, W]

    # visualize_spectrogram(spectrogram, "C:/Users/admin/Desktop/VS Code/VisualTransformer/plots")
    output = model(spectrogram)
    encoder_output = output["encoder_output"]
    decoder_output = output["decoder_output"]
    recon_logits = output["recon_logits"]
    class_logits = output["class_logits"]
    target_patches = output["target_patches"]
    
    # print("Encoder output shape:", encoder_output.shape)
    # print("Decoder output shape:", decoder_output.shape)
    # print("Reconstruction logits shape:", recon_logits.shape)
    # print("Classification logits shape:", class_logits.shape)
    # print("Target patches shape:", target_patches.shape)
    # print("Spectrogram shape:", spectrogram.shape)

if __name__ == "__main__":
    test_model()