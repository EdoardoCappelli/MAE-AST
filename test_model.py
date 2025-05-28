import torch
from torchvision import transforms, datasets
from PIL import Image
import os
import random
from mae import MAE
from types import SimpleNamespace 
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np 
import torch
import torchvision
import matplotlib.pyplot as plt

def reconstruct(config, recon_logits, original, bool_mask):

    B, N, D = recon_logits.shape
    ph, pw = config.patch_size  # es: (16, 16)
    c = config.channels         # = 3
    H, W = config.img_size      # = (224, 224)

    # Check dimensioni
    expected_dim = ph * pw * c
    assert D == expected_dim, f"Embed dim {D} non è compatibile con patch RGB {ph}x{pw}x{c}"

    grid_h = H // ph
    grid_w = W // pw
    assert N == grid_h * grid_w, f"Num patch {N} ≠ {grid_h}x{grid_w}"

    # 1. Estrai le patch dall'immagine originale
    original_patches = extract_patches(original, ph, pw)  # [B, N, C, ph, pw]
    
    # 2. Converti recon_logits in formato patch
    recon_patches = recon_logits.view(B, N, c, ph, pw)  # [B, N, C, ph, pw]
    
    # 3. Combina patch originali e ricostruite usando bool_mask
    # bool_mask: True = patch mascherata (usa ricostruita), False = patch visibile (usa originale)
    combined_patches = torch.where(
        bool_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),  # [B, N, 1, 1, 1]
        recon_patches,    # patch ricostruite
        original_patches  # patch originali
    )
    
    # 4. Ricompone l'immagine completa
    patches = combined_patches.view(B, grid_h, grid_w, c, ph, pw)
    patches = patches.permute(0, 3, 1, 4, 2, 5)  # [B, C, grid_h, ph, grid_w, pw]
    images = patches.contiguous().view(B, c, H, W)

    return images[0]  # ritorna la prima immagine [C, H, W]


def extract_patches(img, patch_h, patch_w):
    """
    Estrae patch da un'immagine.
    
    Args:
        img: tensor [B, C, H, W]
        patch_h, patch_w: dimensioni delle patch
    
    Returns:
        patches: tensor [B, N, C, patch_h, patch_w]
    """
    B, C, H, W = img.shape
    
    # Unfold per estrarre patch
    patches = img.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # patches shape: [B, C, grid_h, grid_w, patch_h, patch_w]
    
    grid_h, grid_w = patches.shape[2], patches.shape[3]
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, grid_h, grid_w, C, patch_h, patch_w]
    patches = patches.contiguous().view(B, grid_h * grid_w, C, patch_h, patch_w)
    
    return patches

config = SimpleNamespace(
        # Dimensioni immagine / patch
        img_size = (224, 224),
        patch_size = (16, 16),
        channels = 3,
        n_mel_bins = 224,
        patch_embedding_dropout = 0.0,
        num_channels = 3,

        # MAE Encoder
        enc_embed_dim = 768,
        enc_mlp_layer_dim = 3072,
        enc_hidden_layers = 6,
        enc_attention_heads = 12,
        enc_layer_norm_eps = 1e-6,
        enc_attention_dropout = 0.0,
        enc_mlp_ratio = 4,

        # MAE Decoder
        dec_hidden_layers = 2,
        dec_embed_dim = 768,
        dec_attention_heads = 12,
        dec_layer_norm_eps = 1e-6,
        dec_attention_dropout = 0.0,
        dec_mlp_ratio = 4,
        
        # Masking
        masking_strategy = "random",
        masking_percentage = 0.75,

        # Training
        batch_size = 32,
        initial_lr = 1e-4,
        weight_decay = 0.05,
        epochs = 50,
        print_freq = 50,

        # Cartelle dataset
        train_dir = r"tiny-imagenet-200/train",
        val_dir   = r"tiny-imagenet-200/val",
        test_dir = r"tiny-imagenet-200/test",
        # Checkpoint
        checkpoints_dir = "checkpoints_mae_imagenet"
    )
checkpoint = torch.load("C:/Users/admin/Desktop/VS Code/MAE/checkpoints_mae_imagenet/model_best.pth", map_location='cpu')
model = MAE(config)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  

# Preprocessing usato per MAE
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Carica una sola immagine da test
test_root = "C:/Users/admin/Desktop/VS Code/tiny-imagenet-200/test/images"
img_paths = [os.path.join(test_root, f) for f in os.listdir(test_root) if f.endswith('.JPEG')]
img_path = random.choice(img_paths)
img = Image.open(img_path).convert("RGB")
# img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)
with torch.no_grad():
    # output: (reconstruction, mask)
    # modello deve avere una forward che restituisce la ricostruzione
    _, reconstructed, _, bool_mask = model(img_tensor) # B, num_patches, embed_dim
    # reconstructed = reconstruct(config, reconstructed, img_tensor)
    reconstructed = reconstruct(config, reconstructed, img_tensor, bool_mask)
# funzione per invertire normalizzazione
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return tensor * std + mean

original_img = unnormalize(img_tensor.squeeze(0)).permute(1, 2, 0).numpy()
recon_img = unnormalize(reconstructed).permute(1, 2, 0).numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_img.clip(0, 1))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(recon_img.clip(0, 1))
plt.title("Reconstructed")
plt.axis("off")
plt.show()

_, reconstructed, _ = model(img_tensor)



# def reconstruct(config, recon_logits, original):
#     """
#     Ricostruisce un'immagine RGB dai logits del decoder MAE.
#     """
#     B, N, D = recon_logits.shape
#     ph, pw = config.patch_size  # es: (16, 16)
#     c = config.channels         # = 3
#     H, W = config.img_size      # = (224, 224)

#     # Check
#     expected_dim = ph * pw * c
#     assert D == expected_dim, f"Embed dim {D} non è compatibile con patch RGB {ph}x{pw}x{c}"

#     # Decodifica: [B, N, D] → [B, N, C, ph, pw]
#     patches = recon_logits.view(B, N, c, ph, pw)

#     # Assumiamo che patch siano in ordine raster
#     grid_h = H // ph
#     grid_w = W // pw
#     assert N == grid_h * grid_w, f"Num patch {N} ≠ {grid_h}x{grid_w}"

#     # Ricompone immagine
#     patches = patches.view(B, grid_h, grid_w, c, ph, pw)
#     patches = patches.permute(0, 3, 1, 4, 2, 5)  # [B, C, grid_h, ph, grid_w, pw]
#     images = patches.contiguous().view(B, c, H, W)

#     # Mostra la prima immagine
#     img_np = images[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, C]
#     img_np = img_np.clip(0, 1)  # opzionale, se output non è normalizzato

#     return images[0] 