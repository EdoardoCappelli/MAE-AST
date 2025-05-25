import torch
import torch.nn as nn
from config import Config
import math
import matplotlib.pyplot as plt
import os
from positional_encoding import SinusoidalPositionalEncoding
from utils.visualize import visualize_patches

class PatchEmbedding(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.enc_embed_dim = config.enc_embed_dim  
        self.patch_size = config.patch_size  
        self.n_mel_bins = config.n_mel_bins # numero di bins del mel spectrogram (asse y)
        
        self.patch_embedding = nn.Unfold(
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        if self.patch_size[0] == self.patch_size[1]:
            self.num_patches = (self.n_mel_bins // self.patch_size[0]) ** 2 
        else:
            print("Gestire le patches non quadrate")

        self.linear = nn.Linear(self.patch_size[0] * self.patch_size[1], self.enc_embed_dim) 
        self.dropout = nn.Dropout(config.patch_embedding_dropout)    
 
    def forward(
            self, 
            spectrogram_values: torch.Tensor,
        ) -> torch.Tensor:
        
        # [B, C, H, W] -> [B, patch_embedding_dim, num_patches]
        patch_embeddings = self.patch_embedding(spectrogram_values) # ogni colonna rappresenta gli indici di una patch
        # print(f"Patch embeddings shape: {patch_embeddings.shape}") 
        # visualize_patches(patch_embeddings, "patch_embeddings")
        
        # [B, embed_dim, num_patches] -> [B, num_patches, patch_embedding_dim] 
        patch_embeddings = patch_embeddings.transpose(-1,-2) # ogni colonna rappresenta gli indici di una patch [B, numero_patches, dimensione_patch]

        # [B, num_patches, patch_embedding_dim] -> [B, num_patches, enc_embed_dim]
        patch_embeddings_after_proj = self.linear(patch_embeddings) 
        # print(f"Patch embeddings shape after linear layer: {patch_embeddings.shape}")
        
        patch_embeddings_after_proj = self.dropout(patch_embeddings_after_proj)
        
        return patch_embeddings, patch_embeddings_after_proj
    
