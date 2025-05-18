import torch
import torch.nn as nn
from config import Config
from patch_extractor import PatchEmbedding
from process_wav import convert_wav_to_mel_spectrogram, visualize_spectrogram
import numpy as np
from typing import Optional, Tuple

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
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

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


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        residual = hidden_states
        # layer norm 1 - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # multi head attention - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)
        # residual connection - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = hidden_states + residual
        residual = hidden_states
        # layer norm 2 - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # feed forward network - [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # residual connection  
        hidden_states = hidden_states + residual

        return hidden_states 
    
class VisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.enc_embed_dim = config.enc_embed_dim
        self.dec_embed_dim = config.dec_embed_dim

        # [B, num_patches, embed_dim]
        self.embeddings = PatchEmbedding(config) # patches estratte da ogni spettrogramma e convertite in embedding  
        
        # maskind del 75% delle patches

        self.encoder = MAE_Encoder(config) # sara una lista di TransformerLayer 
        self.post_enc_layernorm = nn.LayerNorm(config.enc_embed_dim, eps=config.enc_layer_norm_eps)
        self.linear = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
        
        # self.decoder = MAE_Encoder(config, decoder=True)

    
    def forward(self, spectrogram_values:torch.Tensor) -> torch.Tensor:
            
        patch_embeddings = self.embeddings(spectrogram_values) # estraggo patches e converto in embedding (con positional embedding)
        
        # masking delle patches
        
        encoder_output = self.encoder(inputs_embeds = patch_embeddings)
        encoder_output = self.post_enc_layernorm(encoder_output)

        if self.enc_embed_dim != self.dec_embed_dim:
            encoder_output = self.linear(encoder_output) # [B, num_patches, enc_embed_dim] -> [B, num_patches, dec_embed_dim]
        # decoder_output = self.decoder(inputs_embeds = encoder_output)

        print(f"Input shape: {spectrogram_values.shape}") # [B, C, H, W]
        print(f"Patch embedding shape: {patch_embeddings.shape}") # [B, num_patches, embed_dim]

        return encoder_output
    
def test_model():
    config = Config()
    model = VisionTransformer(config)
    print(model)
    print(model.embeddings)
    print(model.encoder)
    print(model.post_enc_layernorm)

    spectrogram_np = np.load("C:/Users/admin/Desktop/VS Code/VisualTransformer/datasets/numpy/balanced_train_segments/EX_-_6RxZyi30Q.npy")
    spectrogram = torch.from_numpy(spectrogram_np)
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0) # [B, C, W, H]
    spectrogram = spectrogram.permute(0, 1, 3, 2) # [B, C, H, W]

    # visualize_spectrogram(spectrogram, "C:/Users/admin/Desktop/VS Code/VisualTransformer/plots")
    out = model(spectrogram)
    print(out.shape)


if __name__ == "__main__":
    test_model()