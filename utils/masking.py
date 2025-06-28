import torch
import torch.nn as nn
import random
from typing import Tuple, List
from config import Config  
import argparse

class Mask(nn.Module):
 
    def __init__(self, args: argparse, config: Config):
        super().__init__()
        self.config = config
        self.masking_percentage = config.masking_percentage
        self.masking_strategy = args.masking_strategy
        
        if self.masking_strategy == 'patch':
            self.chunk_size = config.chunk_size
            self.patches_per_row = config.img_size[0] // config.patch_size[0] 

    def _chunk_masking(self, B: int, num_patches: int, total_patches_to_mask: int, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        masked_indices_list = []
        unmasked_indices_list = []
        
        for _ in range(B):
            masked_idx_set = set()
            
            while len(masked_idx_set) < total_patches_to_mask:
                chunk_size = random.randrange(self.chunk_size[0], self.chunk_size[1] + 1)
                start_pos = random.randrange(num_patches)
                
                for row_offset in range(chunk_size):
                    for col_offset in range(chunk_size):
                        patch_idx = start_pos + (row_offset * self.patches_per_row) + col_offset
                        if patch_idx < num_patches:
                            masked_idx_set.add(patch_idx)

            final_masked_indices = random.sample(list(masked_idx_set), total_patches_to_mask)
            final_masked_indices.sort() 
            
            all_indices = set(range(num_patches))
            final_unmasked_indices = list(all_indices - set(final_masked_indices))
            final_unmasked_indices.sort()
            
            masked_indices_list.append(torch.tensor(final_masked_indices, device=device, dtype=torch.long))
            unmasked_indices_list.append(torch.tensor(final_unmasked_indices, device=device, dtype=torch.long))
            
        return masked_indices_list, unmasked_indices_list

    def _random_masking(self, B: int, num_patches: int, total_patches_to_mask: int, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        num_unmasked = num_patches - total_patches_to_mask
        perm = torch.rand(B, num_patches, device=device).argsort(dim=1)
        
        masked_indices = perm[:, :total_patches_to_mask]
        unmasked_indices = perm[:, total_patches_to_mask:]
        
        masked_indices, _ = torch.sort(masked_indices, dim=1)
        unmasked_indices, _ = torch.sort(unmasked_indices, dim=1)
        
        masked_indices_list = [masked_indices[b] for b in range(B)]
        unmasked_indices_list = [unmasked_indices[b] for b in range(B)]
        
        return masked_indices_list, unmasked_indices_list

    def forward(self, patch_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        B, num_patches, embed_dim = patch_embeddings.shape
        total_patches_to_mask = int(self.masking_percentage * num_patches)
        device = patch_embeddings.device
        
        if self.masking_strategy == 'frame':
            masked_indices_list, unmasked_indices_list = self._random_masking(
                B, num_patches, total_patches_to_mask, device
            )
        else: 
            masked_indices_list, unmasked_indices_list = self._chunk_masking(
                B, num_patches, total_patches_to_mask, device
            )
        
        unmasked_indices = torch.stack(unmasked_indices_list)

        batch_indices = torch.arange(B, device=device).unsqueeze(1)
        unmasked_patches_only = patch_embeddings[batch_indices, unmasked_indices]
        
        bool_mask = torch.ones((B, num_patches), dtype=torch.bool, device=device)
        bool_mask.scatter_(dim=1, index=unmasked_indices, value=False)
        
        return (
            unmasked_patches_only,  
            bool_mask,           
            masked_indices_list,    
            unmasked_indices_list  
        )
