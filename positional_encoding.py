import torch
import torch.nn as nn
import math

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
            x: Tensor, shape [batch_size, num_patches, embed_dim]
        Returns:
            Tensor: positional encodings, shape [batch_size, num_patches, embed_dim]
        """
        batch_size, num_patches, patch_dim = x.size()
        # Slice and expand positional encodings to match input shape
        # print(f"Positional encoding shape: {self.pe[:, :num_patches, :].shape}")
        return self.pe[:, :num_patches, :].expand(batch_size, -1, -1)
