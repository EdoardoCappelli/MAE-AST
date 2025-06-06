    
import torch
import torch.nn as nn
from config import Config
from types import SimpleNamespace  # per simulare Config

class Mask(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.masking_percentage = config.masking_percentage
        self.masking_strategy = config.masking_strategy
        self.encoder_mask_emb = nn.Parameter(torch.randn(config.enc_embed_dim) * 0.02)
        self.patch_size = config.patch_size[0]**2

    def forward(self, patch_embeddings: torch.Tensor) -> tuple:
        B, num_patches, patch_embedding_dim = patch_embeddings.shape
        total_patches_to_mask = int(self.masking_percentage * num_patches)
        
        # Generazione della permutazione casuale per ogni batch
        perm = torch.rand(B, num_patches, device=patch_embeddings.device).argsort(dim=1)
        
        # Creazione degli indici mascherati e non mascherati PRIMA della permutazione
        # I primi total_patches_to_mask saranno mascherati dopo la permutazione
        masked_indices = perm[:, :total_patches_to_mask]  # (B, num_masked)
        unmasked_indices = perm[:, total_patches_to_mask:]  # (B, num_unmasked)
        
        # Creazione della maschera binaria (True = mascherato)
        bool_mask = torch.zeros((B, num_patches), dtype=torch.bool, device=patch_embeddings.device)
        bool_mask.scatter_(dim=1, index=masked_indices, value=True)
        
        # Espansione della maschera per le dimensioni delle patch embeddings
        bool_mask_expanded = bool_mask.unsqueeze(-1).expand(-1, -1, patch_embedding_dim)
        
        # Preparazione del mask embedding
        encoder_mask = self.encoder_mask_emb.view(1, 1, -1).expand(B, num_patches, -1)
        
        # Applicazione del mask embedding alle posizioni mascherate
        patch_embeddings_with_mask = torch.where(bool_mask_expanded, encoder_mask, patch_embeddings)
        
        # Estrazione delle patch non mascherate usando advanced indexing
        # Questo è più efficiente del loop
        batch_indices = torch.arange(B, device=patch_embeddings.device).unsqueeze(1)
        unmasked_patches_only = patch_embeddings[batch_indices, unmasked_indices]
        
        # Conversione degli indici da tensori a liste se necessario per compatibilità
        masked_indices_list = [masked_indices[b] for b in range(B)]
        unmasked_indices_list = [unmasked_indices[b] for b in range(B)]
        
        return (patch_embeddings_with_mask, 
                unmasked_patches_only, 
                bool_mask, 
                masked_indices_list, 
                unmasked_indices_list)
        # B, num_patches, patch_embedding_dim = patch_embeddings.shape
        # total_patches_to_mask = int(self.masking_percentage * num_patches)
        
        # # Generazione della permutazione casuale per ogni batch
        # perm = torch.rand(B, num_patches, device=patch_embeddings.device).argsort(dim=1)

        # # Creazione della maschera binaria
        # bool_mask = torch.zeros((B, num_patches), dtype=torch.bool, device=patch_embeddings.device)
        # bool_mask[:, :total_patches_to_mask] = True
        
        # # Applicazione della permutazione alla maschera
        # bool_mask = bool_mask.gather(dim=1, index=perm)
        # bool_mask = bool_mask.unsqueeze(-1).expand(-1, -1, patch_embedding_dim)
        
        # # Preparazione del mask embedding
        # encoder_mask = self.encoder_mask_emb.view(1, 1, -1).expand(B, -1, -1)

        # # Patches complete con mask embedding applicato
        # patch_embeddings_with_mask_embeddings = torch.where(bool_mask, encoder_mask, patch_embeddings)
        
        # # Estrazione degli indici mascherati e non mascherati
        # masked_indices = []
        # unmasked_indices = []

        # for b in range(B):
        #     masked_idx = torch.where(bool_mask[b, :, 0])[0] # prendo solo la prima colonna tanto sono tutte uguali (le patch sono per riga)
        #     unmasked_idx = torch.where(~bool_mask[b, :, 0])[0]
        #     masked_indices.append(masked_idx)
        #     unmasked_indices.append(unmasked_idx)
        
        # # Estrazione delle patches non mascherate per l'encoder
        # unmasked_patches_only = []
        # for b in range(B):
        #     unmasked_patches_only.append(patch_embeddings[b, unmasked_indices[b]])
        # unmasked_patches_only = torch.stack(unmasked_patches_only)
       
        # return patch_embeddings_with_mask_embeddings, unmasked_patches_only, bool_mask, masked_indices, unmasked_indices
        


'''
torch.Size([2, 1, 4, 4])
spectrogram:
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]],


        [[[16., 17., 18., 19.],
          [20., 21., 22., 23.],
          [24., 25., 26., 27.],
          [28., 29., 30., 31.]]]])
patch_embeddings:
tensor([[[ 0.,  2.,  8., 10.],
         [ 1.,  3.,  9., 11.],
         [ 4.,  6., 12., 14.],
         [ 5.,  7., 13., 15.]],

        [[16., 18., 24., 26.],
         [17., 19., 25., 27.],
         [20., 22., 28., 30.],
         [21., 23., 29., 31.]]])
patch_embeddings transposed:
tensor([[[ 0.,  1.,  4.,  5.],
         [ 2.,  3.,  6.,  7.],
         [ 8.,  9., 12., 13.],
         [10., 11., 14., 15.]],

        [[16., 17., 20., 21.],
         [18., 19., 22., 23.],
         [24., 25., 28., 29.],
         [26., 27., 30., 31.]]])
first patch patch_embeddings[b, riga, colonna] (batch=0):
tensor([0., 1., 4., 5.])
bool_mask:
tensor([[ True,  True, False,  True],
        [ True, False,  True,  True]])
bool_mask:
torch.Size([2, 4])
bool_mask:
tensor([[[ True,  True,  True,  True],
         [ True,  True,  True,  True],
         [False, False, False, False],
         [ True,  True,  True,  True]],

        [[ True,  True,  True,  True],
         [False, False, False, False],
         [ True,  True,  True,  True],
         [ True,  True,  True,  True]]])
encoder_mask:
tensor([[[-0.0267, -0.0348, -0.0044, -0.0309]],

        [[-0.0267, -0.0348, -0.0044, -0.0309]]], grad_fn=<ExpandBackward0>)
patch_embeddings:
tensor([[[ 0.,  1.,  4.,  5.],
         [ 2.,  3.,  6.,  7.],
         [ 8.,  9., 12., 13.],
         [10., 11., 14., 15.]],

        [[16., 17., 20., 21.],
         [18., 19., 22., 23.],
         [24., 25., 28., 29.],
         [26., 27., 30., 31.]]])
patch_embeddings_with_mask_embeddings:
tensor([[[-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [ 8.0000e+00,  9.0000e+00,  1.2000e+01,  1.3000e+01],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02]],

        [[-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [ 1.8000e+01,  1.9000e+01,  2.2000e+01,  2.3000e+01],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02]]],
       grad_fn=<WhereBackward0>)
masked_idx:
tensor([0, 1, 3])
unmasked_idx:
tensor([2])
masked_idx:
tensor([0, 2, 3])
unmasked_idx:
tensor([1])
masked_indices:
[tensor([0, 1, 3]), tensor([0, 2, 3])]
unmasked_indices:
[tensor([2]), tensor([1])]
=== Shape dei tensori in output ===
patch_embeddings_with_mask_embeddings:
tensor([[[-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [ 8.0000e+00,  9.0000e+00,  1.2000e+01,  1.3000e+01],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02]],

        [[-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [ 1.8000e+01,  1.9000e+01,  2.2000e+01,  2.3000e+01],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02],
         [-2.6660e-02, -3.4805e-02, -4.4182e-03, -3.0880e-02]]],
       grad_fn=<WhereBackward0>)  (atteso: (2, 4, 4))
unmasked_patches:
[tensor([[ 8.,  9., 12., 13.]]), tensor([[18., 19., 22., 23.]])]  (atteso: (2, 1, 4))
bool_mask:
torch.Size([2, 4, 4])  (atteso: (2, 4))

Batch 0: mascherate = 3, non mascherate = 1
Batch 1: mascherate = 3, non mascherate = 1

Esempio indici Batch 0:
  masked_indices[0]:    [0, 1, 3]
  unmasked_indices[0]:  [2]

'''


import torch
import torch.nn as nn
from config import Config

class Mask(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.masking_percentage = config.masking_percentage
        self.masking_strategy = config.masking_strategy  # For future extensions
        
        # No encoder_mask_emb needed - masking means complete removal in MAE

    def forward(self, patch_embeddings: torch.Tensor) -> tuple:
        """
        Apply random masking to patch embeddings.
        
        Args:
            patch_embeddings: (B, num_patches, embed_dim) - patches with positional encoding
            
        Returns:
            tuple containing:
            - unmasked_patches_only: (B, num_unmasked, embed_dim) - only unmasked patches for encoder
            - bool_mask: (B, num_patches) - True for masked patches
            - masked_indices_list: list of masked indices per batch
            - unmasked_indices_list: list of unmasked indices per batch
        """
        B, num_patches, embed_dim = patch_embeddings.shape
        device = patch_embeddings.device
        
        # Calculate number of patches to mask
        num_masked = int(self.masking_percentage * num_patches)
        num_unmasked = num_patches - num_masked
        
        # Generate random permutation for each batch
        if self.masking_strategy == "random":
            # Random masking (default MAE strategy)
            perm = torch.rand(B, num_patches, device=device).argsort(dim=1)
        else:
            # Could add other strategies here (block masking, etc.)
            perm = torch.rand(B, num_patches, device=device).argsort(dim=1)
        
        # Split indices into masked and unmasked
        masked_indices = perm[:, :num_masked]  # (B, num_masked)
        unmasked_indices = perm[:, num_masked:]  # (B, num_unmasked)
        
        # Create boolean mask (True = masked, False = unmasked)
        bool_mask = torch.zeros((B, num_patches), dtype=torch.bool, device=device)
        bool_mask.scatter_(dim=1, index=masked_indices, value=True)
        
        # Extract only unmasked patches for the encoder
        # This is the key insight of MAE: encoder only sees unmasked patches
        batch_indices = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
        unmasked_patches_only = patch_embeddings[batch_indices, unmasked_indices]  # (B, num_unmasked, embed_dim)
        
        # Convert to lists for easier handling in decoder reconstruction
        masked_indices_list = [masked_indices[b] for b in range(B)]
        unmasked_indices_list = [unmasked_indices[b] for b in range(B)]
        
        return (
            unmasked_patches_only,      # Only this goes to encoder
            bool_mask,                  # For loss computation
            masked_indices_list,        # For decoder reconstruction
            unmasked_indices_list       # For decoder reconstruction
        )
    
    def get_mask_ratio(self) -> float:
        """Return the masking ratio for logging/debugging."""
        return self.masking_percentage
    
    def set_mask_ratio(self, ratio: float):
        """Dynamically adjust masking ratio if needed."""
        assert 0.0 <= ratio <= 1.0, "Masking ratio must be between 0 and 1"
        self.masking_percentage = ratio

def test_masker():
    """Comprehensive test of the MAE Masker"""
    
    print("=" * 80)
    print("MAE MASKER TEST")
    print("=" * 80)
    
    # Create test configuration
    config = SimpleNamespace(
        masking_percentage=0.75,    # Mask 75% of patches
        masking_strategy="random",   
        enc_embed_dim=4,            # Small embedding dimension for easy visualization
        patch_size=(2, 2),
        img_size=(4, 4)
    )
    
    # Test parameters
    batch_size = 2
    num_patches = 4  # For a 4x4 image with 2x2 patches = 4 patches total
    embed_dim = config.enc_embed_dim
    
    print(f"Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of patches: {num_patches}")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - Masking percentage: {config.masking_percentage * 100}%")
    print(f"  - Expected masked patches per image: {int(config.masking_percentage * num_patches)}")
    print(f"  - Expected unmasked patches per image: {num_patches - int(config.masking_percentage * num_patches)}")
    print()
    
    # Create masker
    masker = Mask(config)
    
    # Create dummy patch embeddings with identifiable values
    # Each patch will have unique values so we can track them
    patch_embeddings = torch.zeros(batch_size, num_patches, embed_dim)
    
    # Fill with identifiable patterns
    for b in range(batch_size):
        for p in range(num_patches):
            # Each patch gets values like [batch_id, patch_id, batch_id, patch_id]
            patch_embeddings[b, p, :] = torch.tensor([b*10 + p + 1] * embed_dim, dtype=torch.float32)
    
    print("INPUT PATCH EMBEDDINGS:")
    print(f"Shape: {patch_embeddings.shape}")
    for b in range(batch_size):
        print(f"Batch {b}:")
        for p in range(num_patches):
            print(f"  Patch {p}: {patch_embeddings[b, p].tolist()}")
    print()
    
    # Apply masking
    torch.manual_seed(42)  # For reproducible results
    outputs = masker(patch_embeddings)
    
    unmasked_patches_only, bool_mask, masked_indices_list, unmasked_indices_list = outputs
    
    print("MASKING RESULTS:")
    print("=" * 50)
    
    # 1. Boolean mask
    print("1. BOOLEAN MASK:")
    print(f"   Shape: {bool_mask.shape}")
    print(f"   Values:")
    for b in range(batch_size):
        mask_str = str(bool_mask[b].tolist()).replace('True', 'T').replace('False', 'F')
        print(f"   Batch {b}: {mask_str}")
        print(f"             {''.join(['M' if x else 'V' for x in bool_mask[b].tolist()])} (M=Masked, V=Visible)")
    print()
    
    # 2. Masked indices
    print("2. MASKED INDICES:")
    for b in range(batch_size):
        print(f"   Batch {b}: {masked_indices_list[b].tolist()}")
    print()
    
    # 3. Unmasked indices  
    print("3. UNMASKED INDICES:")
    for b in range(batch_size):
        print(f"   Batch {b}: {unmasked_indices_list[b].tolist()}")
    print()
    
    # 4. Unmasked patches only
    print("4. UNMASKED PATCHES ONLY (Input to Encoder):")
    print(f"   Shape: {unmasked_patches_only.shape}")
    for b in range(batch_size):
        print(f"   Batch {b}:")
        for i, patch in enumerate(unmasked_patches_only[b]):
            original_patch_idx = unmasked_indices_list[b][i].item()
            print(f"     Unmasked patch {i} (original patch {original_patch_idx}): {patch.tolist()}")
    print()
    
    # 5. Verification - check that masking is consistent
    print("5. VERIFICATION:")
    print("   Checking consistency between outputs...")
    
    all_consistent = True
    for b in range(batch_size):
        # Check that bool_mask matches indices
        expected_mask = torch.zeros(num_patches, dtype=torch.bool)
        expected_mask[masked_indices_list[b]] = True
        
        if not torch.equal(bool_mask[b], expected_mask):
            print(f"   ❌ Batch {b}: bool_mask doesn't match masked_indices")
            all_consistent = False
        
        # Check that unmasked patches match original
        for i, unmasked_idx in enumerate(unmasked_indices_list[b]):
            original_patch = patch_embeddings[b, unmasked_idx]
            extracted_patch = unmasked_patches_only[b, i]
            if not torch.equal(original_patch, extracted_patch):
                print(f"   ❌ Batch {b}: Unmasked patch {i} doesn't match original patch {unmasked_idx}")
                all_consistent = False
        
        # Check counts
        expected_masked = int(config.masking_percentage * num_patches)
        actual_masked = len(masked_indices_list[b])
        actual_unmasked = len(unmasked_indices_list[b])
        
        if actual_masked != expected_masked:
            print(f"   ❌ Batch {b}: Expected {expected_masked} masked patches, got {actual_masked}")
            all_consistent = False
            
        if actual_masked + actual_unmasked != num_patches:
            print(f"   ❌ Batch {b}: Masked + unmasked ({actual_masked} + {actual_unmasked}) != total patches ({num_patches})")
            all_consistent = False
    
    if all_consistent:
        print("   ✅ All consistency checks passed!")
    print()
    
    # 6. Statistics
    print("6. STATISTICS:")
    total_patches = batch_size * num_patches
    total_masked = bool_mask.sum().item()
    total_unmasked = total_patches - total_masked
    
    print(f"   Total patches across all batches: {total_patches}")
    print(f"   Total masked patches: {total_masked}")
    print(f"   Total unmasked patches: {total_unmasked}")
    print(f"   Actual masking ratio: {total_masked / total_patches:.2%}")
    print(f"   Expected masking ratio: {config.masking_percentage:.2%}")
    print()
    
    # 7. Visual representation
    print("7. VISUAL REPRESENTATION:")
    print("   Original patches layout (2x2 grid for each batch):")
    
    for b in range(batch_size):
        print(f"   Batch {b}:")
        mask = bool_mask[b]
        # Assuming 2x2 grid of patches
        grid = [
            [f"P0{'(M)' if mask[0] else '(V)'}", f"P1{'(M)' if mask[1] else '(V)'}"],
            [f"P2{'(M)' if mask[2] else '(V)'}", f"P3{'(M)' if mask[3] else '(V)'}"]
        ]
        for row in grid:
            print(f"     {row[0]:8} {row[1]:8}")
        print()


def test_different_scenarios():
    """Test masker with different configurations"""
    
    print("=" * 80)
    print("TESTING DIFFERENT MASKING RATIOS")
    print("=" * 80)
    
    # Test different masking ratios
    ratios = [0.25, 0.5, 0.75, 0.9]
    
    for ratio in ratios:
        print(f"\nTesting masking ratio: {ratio:.0%}")
        print("-" * 40)
        
        config = SimpleNamespace(
            masking_percentage=ratio,
            masking_strategy="random",
            enc_embed_dim=2
        )
        
        masker = Mask(config)
        
        # Small test case
        batch_size = 1
        num_patches = 8
        embed_dim = 2
        
        patch_embeddings = torch.randn(batch_size, num_patches, embed_dim)
        
        torch.manual_seed(123)  # For consistent results
        outputs = masker(patch_embeddings)
        unmasked_patches_only, bool_mask, masked_indices_list, unmasked_indices_list = outputs
        
        num_masked = len(masked_indices_list[0])
        num_unmasked = len(unmasked_indices_list[0])
        actual_ratio = num_masked / num_patches
        
        print(f"Expected masked: {int(ratio * num_patches)}, Actual: {num_masked}")
        print(f"Expected unmasked: {num_patches - int(ratio * num_patches)}, Actual: {num_unmasked}")
        print(f"Actual ratio: {actual_ratio:.2%}")
        print(f"Mask pattern: {['M' if x else 'V' for x in bool_mask[0].tolist()]}")


if __name__ == "__main__":
    # Run comprehensive test
    test_masker()
    
    # Run additional scenario tests
    test_different_scenarios()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)