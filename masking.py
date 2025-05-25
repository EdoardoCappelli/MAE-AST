    
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
        
        # Creazione della maschera binaria
        bool_mask = torch.zeros((B, num_patches), dtype=torch.bool, device=patch_embeddings.device)
        bool_mask[:, :total_patches_to_mask] = True
        
        # Applicazione della permutazione alla maschera
        bool_mask = bool_mask.gather(dim=1, index=perm)
        print(f"bool_mask:\n{bool_mask}")
        print(f"bool_mask:\n{bool_mask.shape}") # B, num_patches
        
        bool_mask = bool_mask.unsqueeze(-1).expand(-1, -1, patch_embedding_dim)

        print(f"bool_mask:\n{bool_mask}")
        
        # Preparazione del mask embedding
        encoder_mask = self.encoder_mask_emb.view(1, 1, -1).expand(B, -1, -1)
        print(f"encoder_mask:\n{encoder_mask}")

        print(f"patch_embeddings:\n{patch_embeddings}")

        # Patches complete con mask embedding applicato
        patch_embeddings_with_mask_embeddings = torch.where(bool_mask, encoder_mask, patch_embeddings)
        print(f"patch_embeddings_with_mask_embeddings:\n{patch_embeddings_with_mask_embeddings}")
        
        # Estrazione degli indici mascherati e non mascherati
        masked_indices = []
        unmasked_indices = []

        for b in range(B):
            masked_idx = torch.where(bool_mask[b, :, 0])[0] # prendo solo la prima colonna tanto sono tutte uguali (le patch sono per riga)
            unmasked_idx = torch.where(~bool_mask[b, :, 0])[0]
            masked_indices.append(masked_idx)
            unmasked_indices.append(unmasked_idx)
            print(f"masked_idx:\n{masked_idx}")
            print(f"unmasked_idx:\n{unmasked_idx}")
        print(f"masked_indices:\n{masked_indices}")
        print(f"unmasked_indices:\n{unmasked_indices}")
        
        # Estrazione delle patches non mascherate per l'encoder
        unmasked_patches_only = []
        for b in range(B):
            unmasked_patches_only.append(patch_embeddings[b, unmasked_indices[b]])
        unmasked_patches_only = torch.stack(unmasked_patches_only)
       
        print(f"unmasked_patches_only:\n{unmasked_patches_only}")

        return patch_embeddings_with_mask_embeddings, unmasked_patches_only, bool_mask, masked_indices, unmasked_indices
        
def test():
    
    config = SimpleNamespace(
        masking_percentage=0.75,    
        masking_strategy="random",   
        enc_embed_dim=4,           
        patch_size=(2, 2),
    )

    B = 2
    num_patches = 4  # 448
    n_mel_bins = 4
    spect_width = 4

    patch_embedding = nn.Unfold(
        kernel_size=config.patch_size,
        stride=config.patch_size,
    )

    num_patches = (n_mel_bins // config.patch_size[0]) ** 2 

    spectrogram1 = torch.arange(
        start=0,
        end=n_mel_bins * spect_width,
        dtype=torch.float32
    ).view(1, n_mel_bins, spect_width)  
    
    spectrogram2 = torch.arange(
        start=n_mel_bins * spect_width,
        end=n_mel_bins * spect_width * 2,
        dtype=torch.float32
    ).view(1, n_mel_bins, spect_width)  
    
    spectrogram = torch.stack([spectrogram1, spectrogram2])
    print(spectrogram.shape)
    print(f"spectrogram:\n{spectrogram}")
   
    patch_embeddings = patch_embedding(spectrogram)
    # patch_embeddings = torch.randn(B, num_patches, config.enc_embed_dim)
    print(f"patch_embeddings:\n{patch_embeddings}")
    patch_embeddings = patch_embeddings.transpose(-1,-2) # ogni riga rappresenta una patch
    print(f"patch_embeddings transposed:\n{patch_embeddings}")

    print(f"first patch patch_embeddings[b, riga, colonna] (batch=0):\n{patch_embeddings[0, 0, :]}")

    masker = Mask(config)
    
    (
        patch_embeddings_with_mask_embeddings, 
        unmasked_patches_only, 
        bool_mask,
        masked_indices, 
        unmasked_indices 
    ) = masker(patch_embeddings)

    # Stampa risultati
    print("=== Shape dei tensori in output ===")
    print(f"patch_embeddings_with_mask_embeddings:\n{patch_embeddings_with_mask_embeddings}  "
        f"(atteso: ({B}, {num_patches}, {config.enc_embed_dim}))")
    print(f"unmasked_patches:\n{unmasked_patches_only}  "
        f"(atteso: ({B}, {num_patches - int(config.masking_percentage * num_patches)}, {config.enc_embed_dim}))")
    print(f"bool_mask:\n{bool_mask.shape}  (atteso: ({B}, {num_patches}))")
    print()

    # 7. Controlliamo quante patch sono effettivamente mascherate in ciascun batch
    unmasked = 0
    for b in range(B):
        num_masked = bool_mask[b,:,0].sum().item()
        num_unmasked = (~bool_mask[b,:,0]).sum().item()
        unmasked = num_unmasked
        print(f"Batch {b}: mascherate = {num_masked}, non mascherate = {num_unmasked}")

    print()

    # 8. Stampiamo gli indici estratti per il primo batch per verifica
    print("Esempio indici Batch 0:")
    print("  masked_indices[0]:   ", masked_indices[0].tolist())
    print("  unmasked_indices[0]: ", unmasked_indices[0].tolist())
    print()

    print("Test positional encoding")
    pe1 = torch.arange(
        start=0,
        end=n_mel_bins * spect_width,
        dtype=torch.float32
    ).view(num_patches, config.enc_embed_dim)  
    
    pe2 = torch.arange(
        start=n_mel_bins * spect_width,
        end=n_mel_bins * spect_width * 2,
        dtype=torch.float32
    ).view(num_patches, config.enc_embed_dim)  
    
    pe = torch.stack([pe1, pe2])

    print(f"pe:\n{pe}")

    # pe_for_unmasked_pathes_only = []
    # pe_unmasked_indices = []

    # for b in range(B):
    #     pe_unmasked_idx = torch.where(~bool_mask[b,:,0])[0]
    #     pe_unmasked_indices.append(pe_unmasked_idx)
    # print(f"pe_unmasked_indices:\n{pe_unmasked_indices}")
    

    # for b in range(B):
    #     idx = pe_unmasked_indices[b].item()
    #     pe_for_unmasked_pathes_only.append(pe[b, 0, idx, :])
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
    print(f"pe_for_unmasked_pathes_only:\n{pe_for_unmasked_pathes_only}")

    unmasked_patches_only_with_pe = pe_for_unmasked_pathes_only + unmasked_patches_only
    print(f"unmasked_patches_only_with_pe:\n{unmasked_patches_only_with_pe}")
    
    encoder_output1 = torch.arange(
        start=0,
        end=unmasked * config.enc_embed_dim,
        dtype=torch.float32
    ).view(unmasked, config.enc_embed_dim)  
    
    encoder_output2 = torch.arange(
        start=unmasked * config.enc_embed_dim,
        end=unmasked * config.enc_embed_dim * 2,
        dtype=torch.float32
    ).view(unmasked, config.enc_embed_dim)  

    encoder_output = torch.stack([encoder_output1, encoder_output2])
    print(f"encoder_output:\n{encoder_output}")

    decoder_mask_emb = nn.Parameter(torch.FloatTensor(config.enc_embed_dim).uniform_()) # è l'embedding che rappresenta la patch mascherata e dovrebbe essere appresa durante il training

    x_full = []

    for b in range(B):
        x = torch.zeros((num_patches, config.enc_embed_dim), device=encoder_output[b].device)

        # Inserisci embeddings encoder nei punti non mascherati
        x[unmasked_indices[b]] = encoder_output[b]

        # Inserisci decoder_mask_emb nei punti mascherati
        x[masked_indices[b]] = decoder_mask_emb  # broadcast su più righe

        x_full.append(x)

    print(f"x_full:\n{x_full}")

if __name__ == "__main__":
    test()


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