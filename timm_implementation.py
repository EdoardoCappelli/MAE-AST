import torch
import torch.nn as nn
from config import Config
import numpy as np
from typing import Optional, Tuple, Dict
from masking import Mask
from positional_encoding import SinusoidalPositionalEncoding
from utils.visualize import visualize_spectrogram, visualize_patches
from types import SimpleNamespace 

from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
import timm.models.vision_transformer as vit_timm


class MAE_Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Usa i Block ottimizzati di timm invece di layer personalizzati
        self.blocks = nn.ModuleList([
            Block(
                dim=config.enc_embed_dim,
                num_heads=config.num_enc_attention_heads,
                mlp_ratio=4.0,  # Tipicamente 4x per ViT
                qkv_bias=True,
                attn_drop=config.enc_attention_dropout,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            ) 
            for _ in range(config.num_enc_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(config.enc_embed_dim, eps=config.enc_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Layer norm
        hidden_states = self.norm(hidden_states)
        return hidden_states

class MAE_Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.blocks = nn.ModuleList([
            Block(
                dim=config.dec_embed_dim,
                num_heads=config.num_dec_attention_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                attn_drop=config.dec_attention_dropout, 
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for _ in range(config.num_dec_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(config.dec_embed_dim, eps=config.dec_layer_norm_eps)

    def forward(self, masked_patches: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            masked_patches = block(masked_patches)
        
        # Layer norm 
        masked_patches = self.norm(masked_patches)
        return masked_patches

class MAE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.enc_embed_dim = config.enc_embed_dim
        self.dec_embed_dim = config.dec_embed_dim

        # Batch normalization 
        self.batch_norm = nn.BatchNorm2d(num_features=config.num_channels, affine=False)
        
        # PatchEmbed di timm
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.num_channels,  
            embed_dim=config.enc_embed_dim,
            norm_layer=None, 
            flatten=True,
            bias=True
        )
        
        # Altri componenti rimangono uguali
        self.mask = Mask(config) 
        self.positional_embeddings_before_encoder = SinusoidalPositionalEncoding(embed_dim=self.enc_embed_dim)
        # self.positional_embeddings_before_encoder = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.enc_embed_dim), requires_grad=False)
        # Encoder e decoder con componenti timm
        self.encoder = MAE_Encoder(config)
        
        # Proiezione lineare tra encoder e decoder se dimensioni diverse
        self.linear = nn.Linear(self.enc_embed_dim, self.dec_embed_dim) if self.enc_embed_dim != self.dec_embed_dim else nn.Identity()
        
        # Decoder mask embedding
        self.decoder_mask_emb = nn.Parameter(torch.zeros(config.dec_embed_dim))
        self.positional_embeddings_before_decoder = SinusoidalPositionalEncoding(embed_dim=self.dec_embed_dim)
        # self.positional_embeddings_before_decoder = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.dec_embed_dim), requires_grad=False)
        self.decoder = MAE_Decoder(config)
        
        # Proiezioni finali
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
        # initialize (and freeze) pos_embed by sin-cos embedding
        # positional_embeddings_before_encoder = get_2d_sincos_pos_embed(self.positional_embeddings_before_encoder.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        # self.positional_embeddings_before_encoder.data.copy_(torch.from_numpy(positional_embeddings_before_encoder).float().unsqueeze(0))
        
        
        # positional_embeddings_before_decoder = get_2d_sincos_pos_embed(self.positional_embeddings_before_decoder.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        # self.positional_embeddings_before_decoder.data.copy_(torch.from_numpy(positional_embeddings_before_decoder).float().unsqueeze(0))
       
       # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Inizializzazione speciale per mask token
        nn.init.trunc_normal_(self.decoder_mask_emb, std=0.02)
    
    def forward(self, spectrogram_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(spectrogram_values.shape) == 5:
            spectrogram_values = spectrogram_values.squeeze(dim=2)

        # Preprocessing
        spectrogram_values = self.batch_norm(spectrogram_values) * 0.5
        
        # Usa PatchEmbed di timm - più efficiente e ottimizzato
        patch_embeddings = self.patch_embed(spectrogram_values)  # [B, num_patches, embed_dim]
        
        B, num_patches, _ = patch_embeddings.shape
        
        # Masking (mantieni la tua logica esistente)
        (
            patch_embeddings_with_mask_embeddings, 
            unmasked_patches_only, 
            bool_mask,
            masked_indices, 
            unmasked_indices 
        ) = self.mask(patch_embeddings)

        # Positional encoding per encoder
        pe = self.positional_embeddings_before_encoder(patch_embeddings_with_mask_embeddings)
        pe_for_unmasked_pathes_only = []

        for b in range(B):
            pe_unmasked_idx = torch.where(~bool_mask[b,:,0])[0]  
            selected_rows = pe[b, pe_unmasked_idx, :]
            pe_for_unmasked_pathes_only.append(selected_rows)

        pe_for_unmasked_pathes_only = torch.stack(pe_for_unmasked_pathes_only)
        unmasked_patches_only_with_pe = pe_for_unmasked_pathes_only + unmasked_patches_only
        
        # Encoding con timm blocks ottimizzati
        encoder_output = self.encoder(unmasked_patches_only_with_pe)

        # Proiezione encoder -> decoder se necessario
        encoder_output = self.linear(encoder_output)
        
        # Prepara input per decoder
        decoder_input = []
        for b in range(B):
            x = torch.zeros((num_patches, self.dec_embed_dim), device=encoder_output[b].device)
            x[unmasked_indices[b]] = encoder_output[b]
            x[masked_indices[b]] = self.decoder_mask_emb
            decoder_input.append(x)
        
        decoder_input = torch.stack(decoder_input)

        # Positional encoding per decoder
        decoder_input = decoder_input + self.positional_embeddings_before_decoder(patch_embeddings_with_mask_embeddings)
        
        # Decoding con timm blocks ottimizzati
        decoder_output = self.decoder(decoder_input)
        
        # Proiezioni finali
        reconstruction_logits = self.final_proj_reconstruction(decoder_output)
        classification_logits = self.final_proj_classification(decoder_output)
        
        return {
            "recon_patches": reconstruction_logits,
            "class_logits": classification_logits,
            "target_patches": patch_embeddings,
        }

def test():
    config = Config()
    model = MAE(config)
    
    # Test con dati sintetici
    w, l = 900, 128
    spectrogram = torch.arange(0, w*l, dtype=torch.float32).reshape(w, l)
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # [B, C, W, H]

    output = model(spectrogram)
    recon_patches = output["recon_patches"]
    class_logits = output["class_logits"]
    target_patches = output["target_patches"]
    
    print("Reconstruction logits shape:", recon_patches.shape)
    print("Classification logits shape:", class_logits.shape)
    print("Target patches shape:", target_patches.shape)

if __name__ == "__main__":
    test()