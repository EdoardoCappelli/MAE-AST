import torch
import torch.nn as nn
from config import Config
import math
import matplotlib.pyplot as plt
import os

# Invece di usare embedding posizionali learnable, li calcoliamo con una funzione sinusoidale
# che calcola il valore di ogni posizione in base alla dimensione dell'embedding
class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    """
    def __init__(self, embed_dim: int, max_len: int = 480000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )  # [embed_dim/2]
        pe = torch.zeros(1, max_len, embed_dim) 
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embed_dim]
        Returns:
            Tensor: positional encodings, shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        # Slice and expand positional encodings to match input shape
        return self.pe[:, :seq_len, :].expand(batch_size, -1, -1)

# Estrare le patches da ogni spettrogramma e le converte in embedding di dimensione embed_dim
class PatchEmbedding(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim # dimensione dell'embedding vector
        self.patch_size = config.patch_size # dimensione della patch
        self.n_mel_bins = config.n_mel_bins # numero di bins del mel spectrogram (asse y)
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, # capire quanni canali ha lo spettrogramma
            out_channels=self.embed_dim,
            kernel_size=self.patch_size[0],
            stride=self.patch_size[1],
            padding="valid" # per non avere padding
        )
        
        if self.patch_size[0] == self.patch_size[1]:
            self.num_patches = (self.n_mel_bins // self.patch_size[0]) ** 2 # numero di patches
        else:
            print("Gestire le patches non quadrate")
        
        self.position_embedding = SinusoidalPositionalEncoding(embed_dim=self.embed_dim) # embedding posizionale
        
        # self.num_positions = self.num_patches # numero di posizioni
        # self.register_buffer(
        #     "positions_ids",
        #     torch.arange(self.num_positions).expand((1, -1)) # espando la dimensione batch
        # )


    def forward(
            self, 
            spectrogram_values: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            spectrogram_values: Tensor, shape [B, C, H, W]
        Returns:
            patch embeddings: Tensor, shape [B, num_patches, embed_dim]
        """
        
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        patch_embeddings = self.patch_embedding(spectrogram_values) # estraggo le patches usando la convoluzione
        
        # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches] 
        patch_embeddings = patch_embeddings.flatten(2)

        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim] 
        patch_embeddings = patch_embeddings.transpose(1, 2) # in questo modo diamo tranformer una lista di embeddings

        patch_embeddings = patch_embeddings + self.position_embedding(patch_embeddings) # aggiungo l'embedding posizionale
        
        return patch_embeddings
