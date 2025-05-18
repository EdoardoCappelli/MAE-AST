import torch
import torch.nn as nn
from config import Config


class MaskingGenerator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mask_ratio
        self.mask_length = config.mask_length
        self.num_patches = config.num_patches
        self.masking_strategy = config.masking_strategy
        self.perc_masked_tokens = config.perc_masked_tokens

    def forward(self, spectrogram_values) -> torch.Tensor:
        B, T, C = spectrogram_values.shape

        masked_indices = []
        unmasked_indices = []

        if self.masking_strategy == "random":
            for i in range(B):
                random_indices = list(range(T))
                random_indices = torch.randperm(T).tolist()
                unmasked_indices = random_indices[:int((1-self.mask_ratio) * T)]

        
        elif self.masking_strategy == "chunk":
            pass

        else:
            raise ValueError("Invalid masking strategy")
        
        return masked_indices, unmasked_indices
        
