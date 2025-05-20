import torch
import torch.nn as nn
from config import Config


class Mask(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.masking_percentage = config.masking_percentage
        self.masking_strategy = config.masking_strategy
        self.encoder_mask_emb = nn.Parameter(torch.FloatTensor(config.enc_embed_dim).uniform_()) # è l'embedding che rappresenta la patch mascherata e dovrebbe essere appresa durante il training

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        B, num_patches, patch_embedding_dim = patch_embeddings.shape
        total_patches_to_mask = int(self.masking_percentage * num_patches)
        total_patches_to_not_mask = num_patches - total_patches_to_mask

        masked_indices = torch.zeros((B, total_patches_to_mask), dtype=torch.long, device=patch_embeddings.device)
        unmasked_indices = torch.zeros((B, total_patches_to_not_mask), dtype=torch.long, device=patch_embeddings.device)
        masked_patch_embeddings = torch.zeros((B, num_patches, patch_embedding_dim), dtype=torch.long, device=patch_embeddings.device)

        if self.masking_strategy == "random":
            for b in range(B):
                random_indices = torch.randperm(num_patches, device=patch_embeddings.device)
                
                masked_indices_i = random_indices[:total_patches_to_mask]
                unmasked_indices_i = random_indices[total_patches_to_mask:]
                
                masked_indices[b, :] = masked_indices_i
                unmasked_indices[b, :] = unmasked_indices_i
                
                patch_embeddings[b, masked_indices_i, :] = self.encoder_mask_emb # Mask the patches
                masked_patch_embeddings[b] = patch_embeddings[b]

                # print("Masked indices i:", masked_indices_i)
                # print("Unmasked indices i:", unmasked_indices_i)
                # print("Masked patch embeddings i:", masked_patch_embeddings[b])
                # print("Unmasked patch embeddings i:", patch_embeddings[b, unmasked_indices_i])
                # print("Masked patch embeddings shape i:", masked_patch_embeddings[b].shape)
                # print("Unmasked patch embeddings shape:", patch_embeddings[b, unmasked_indices_i].shape)
            
        elif self.masking_strategy == "chunk":
            pass

        else:
            raise ValueError("Invalid masking strategy")
        
        return masked_patch_embeddings, masked_indices, unmasked_indices
        
