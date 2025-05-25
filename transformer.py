import torch
import torch.nn as nn
from config import Config
from patch_extractor import PatchEmbedding
import numpy as np
from typing import Optional, Tuple, Dict
from masking import Mask
from positional_encoding import SinusoidalPositionalEncoding
from utils.visualize import visualize_spectrogram, visualize_patches
from types import SimpleNamespace 

class Vit_Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim
        self.num_heads = config.num_enc_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # dimensione di ogni head
        self.scale = self.head_dim ** -0.5 # 1/sqrt(head_dim) mateniamo magntude simile per ogni head
        self.dropout = config.enc_attention_dropout
        
        assert self.embed_dim % self.num_heads == 0, f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"

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
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(query.dtype) # vediamo le relazioni tra i token in termini di probabiltà applicando la softmax by row
        # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # dropout per evitare overfitting
        
        # [B, num_heads, num_patches, head_dim] -> [B, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value) # [B, num_heads, num_patches, head_dim] -> [B, num_heads, num_patches, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, seq_len, self.embed_dim) # [B, num_heads, num_patches, head_dim] 
        attn_output = attn_output.reshape(b, seq_len, self.embed_dim) # [B, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class Vit_MLP(nn.Module):
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
    
class Vit_EncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim
        self.self_attn = Vit_Attention(config)  
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.enc_layer_norm_eps)
        self.mlp = Vit_MLP(config) 
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.enc_layer_norm_eps)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        not_masked_patches = hidden_states

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
    
class MAE_Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [Vit_EncoderLayer(config) for _ in range(config.num_enc_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.enc_embed_dim, eps=config.enc_layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor # indici delle patch non mascherate
    ) -> torch.Tensor:
        
        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            not_masked_patches_encoded = encoder_layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)

        return hidden_states

class MAE_Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [Vit_EncoderLayer(config) for _ in range(config.num_dec_hidden_layers)]
        )

        self.norm = nn.LayerNorm(config.dec_embed_dim, eps=config.dec_layer_norm_eps)


    def forward(
        self,
        masked_patches: torch.Tensor # patch mascherate 
    ) -> torch.Tensor:
        # masked_patches: [Batch_Size, Num_Patches, Embed_Dim]

        for decoder_layer in self.layers:
            # [Batch_Size, Num_Patches, dec_embed_dim] -> [Batch_Size, Num_Patches, dec_embed_dim]
            masked_patches_decoded = decoder_layer(masked_patches)
        
        masked_patches_decoded = self.norm(masked_patches_decoded)

        return masked_patches_decoded

class MAE(nn.Module):
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
        self.linear = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
        
        self.decoder_mask_emb = nn.Parameter(torch.FloatTensor(config.dec_embed_dim).uniform_()) # è l'embedding che rappresenta la patch mascherata e dovrebbe essere appresa durante il training
        self.positional_embeddings_before_decoder = SinusoidalPositionalEncoding(embed_dim=self.dec_embed_dim)
        self.decoder = MAE_Decoder(config)
        
        self.final_proj_reconstruction = nn.Linear(
            config.dec_embed_dim,
            config.patch_size[0] * config.patch_size[1],
            bias=True
        )
        self.final_proj_classification = nn.Linear(
            config.dec_embed_dim,
            config.patch_size[0] * config.patch_size[1],
            bias=True
        )

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize mask token
        torch.nn.init.normal_(self.decoder_mask_emb, std=0.02)
        
        # Initialize linear layers
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(_init_weights)
    
    def get_original_patches(self, patch_embeddings: torch.Tensor, masked_indices: list[torch.Tensor]) -> torch.Tensor:
        B, num_patches, patch_dim = patch_embeddings.shape
        M = masked_indices[0].shape[0]
        orig_patches = torch.zeros((B, M, patch_dim), device=patch_embeddings.device, dtype=patch_embeddings.dtype)

        for b in range(B):
            orig_patches[b] = patch_embeddings[b, masked_indices[b], :]

        return orig_patches
    
    def forward(self, spectrogram_values:torch.Tensor) -> torch.Tensor:
        if len(spectrogram_values.shape) == 5:
            spectrogram_values = spectrogram_values.squeeze(dim=2)

        spectrogram_values = self.batch_norm(spectrogram_values) * 0.5
        patch_embeddings, patch_embeddings_after_proj =  self.patch_embeddings(spectrogram_values) # [B, num_patches, patch_embed_dim] 

        B, num_patches, _ = patch_embeddings.shape
        (
            patch_embeddings_with_mask_embeddings, 
            unmasked_patches_only, 
            bool_mask,
            masked_indices, 
            unmasked_indices 
        ) = self.mask(patch_embeddings_after_proj)

        pe = self.positional_embeddings_before_encoder(patch_embeddings_with_mask_embeddings)
        pe_for_unmasked_pathes_only = []
        pe_unmasked_indices = []

        for b in range(B):
            # Indici delle righe non mascherate per il batch b
            pe_unmasked_idx = torch.where(~bool_mask[b,:,0])[0]  
            pe_unmasked_indices.append(pe_unmasked_idx)

            # Estrai tutte le righe corrispondenti da pe[b, 0, :, :]
            selected_rows = pe[b, pe_unmasked_idx, :]  # shape: [num_unmasked, W]
            pe_for_unmasked_pathes_only.append(selected_rows)

        pe_for_unmasked_pathes_only = torch.stack(pe_for_unmasked_pathes_only)
        
        unmasked_patches_only_with_pe = pe_for_unmasked_pathes_only + unmasked_patches_only
        
        encoder_output = self.encoder(unmasked_patches_only_with_pe)

        if self.enc_embed_dim != self.dec_embed_dim:
            encoder_output = self.linear(encoder_output) # [B, num_patches, enc_embed_dim] -> [B, num_patches, dec_embed_dim]
        
        decoder_input = []

        for b in range(B):
            x = torch.zeros((num_patches, self.dec_embed_dim), device=encoder_output[b].device)

            # Inserisci embeddings encoder nei punti non mascherati
            x[unmasked_indices[b]] = encoder_output[b]

            # Inserisci decoder_mask_emb nei punti mascherati
            x[masked_indices[b]] = self.decoder_mask_emb  # broadcast su più righe

            decoder_input.append(x)
        decoder_input = torch.stack(decoder_input)

        decoder_input = decoder_input + self.positional_embeddings_before_decoder(patch_embeddings_with_mask_embeddings)  # (B, num_patches, D_dec)
        decoder_output = self.decoder(decoder_input) 
        
        # B_indices = torch.arange(B, device=decoder_output.device).unsqueeze(1)  # (B, 1)
        # recon_masked = []
        # for b in range(B):
        #     recon_masked.append(decoder_output[b, masked_indices[b], :])  # (M, D_dec)
        # recon_masked = torch.stack(recon_masked, dim=0)

        recostruction_logits = self.final_proj_reconstruction(decoder_output)
        classification_logits = self.final_proj_reconstruction(decoder_output)
        # target_patches = self.get_original_patches(patch_embeddings, masked_indices)
        
        return {
            "recon_logits": recostruction_logits,
            "class_logits": classification_logits,
            "target_patches": patch_embeddings,
        }
    
def test():
    config = Config()
    model = MAE(config)
    
    # print(spectrogram.shape) # 999, 128

    #---------------------------------------------
    # spectrogram_np = np.load("C:/Users/admin/Desktop/VS Code/VisualTransformer/datasets/numpy/balanced_train_segments/EX_-_6RxZyi30Q.npy")
    # spectrogram = torch.from_numpy(spectrogram_np)
    # spectrogram = torch.arange(
    #     start=0,
    #     end=spectrogram.shape[0] * spectrogram.shape[1],
    #     dtype=torch.float32
    # ).view(spectrogram.shape)  
    # spectrogram = spectrogram.unsqueeze(0).unsqueeze(0) # [B, C, W, H]
    # print(spectrogram.shape)
    # print(spectrogram)
    #---------------------------------------------

    w = 900
    l = 128
    spectrogram = torch.arange(0, w*l, dtype=torch.float32).reshape(w, l)
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0) # [B, C, W, H]
    # spectrogram = spectrogram.permute(0, 1, 3, 2) # [B, C, H, W]
    # print(spectrogram.shape)
    # print(spectrogram)


    # visualize_spectrogram(spectrogram, "C:/Users/admin/Desktop/VS Code/VisualTransformer/plots")
    output = model(spectrogram)
    recon_logits = output["recon_logits"]
    class_logits = output["class_logits"]
    target_patches = output["target_patches"]
    
    # print("Reconstruction logits shape:", recon_logits.shape)
    # print("Classification logits shape:", class_logits.shape)
    # print("Target patches shape:", target_patches.shape)

if __name__ == "__main__":
    test()

