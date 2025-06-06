# import torch
# from torchvision import transforms, datasets
# from PIL import Image
# import os
# import random
# from mae import MAE
# from types import SimpleNamespace 
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
# import numpy as np 
# import torch
# import torchvision
# import matplotlib.pyplot as plt

# def reconstruct(config, recon_logits, original, bool_mask):

#     B, N, D = recon_logits.shape
#     ph, pw = config.patch_size  # es: (16, 16)
#     c = config.channels         # = 3
#     H, W = config.img_size      # = (224, 224)

#     # Check dimensioni
#     expected_dim = ph * pw * c
#     assert D == expected_dim, f"Embed dim {D} non è compatibile con patch RGB {ph}x{pw}x{c}"

#     grid_h = H // ph
#     grid_w = W // pw
#     assert N == grid_h * grid_w, f"Num patch {N} ≠ {grid_h}x{grid_w}"

#     # 1. Estrai le patch dall'immagine originale
#     original_patches = extract_patches(original, ph, pw)  # [B, N, C, ph, pw]
    
#     # 2. Converti recon_logits in formato patch
#     recon_patches = recon_logits.view(B, N, c, ph, pw)  # [B, N, C, ph, pw]
    
#     # 3. Combina patch originali e ricostruite usando bool_mask
#     # bool_mask: True = patch mascherata (usa ricostruita), False = patch visibile (usa originale)
#     combined_patches = torch.where(
#         bool_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),  # [B, N, 1, 1, 1]
#         recon_patches,    # patch ricostruite
#         original_patches  # patch originali
#     )
    
#     # 4. Ricompone l'immagine completa
#     patches = combined_patches.view(B, grid_h, grid_w, c, ph, pw)
#     patches = patches.permute(0, 3, 1, 4, 2, 5)  # [B, C, grid_h, ph, grid_w, pw]
#     images = patches.contiguous().view(B, c, H, W)

#     return images[0]  # ritorna la prima immagine [C, H, W]


# def extract_patches(img, patch_h, patch_w):
#     """
#     Estrae patch da un'immagine.
    
#     Args:
#         img: tensor [B, C, H, W]
#         patch_h, patch_w: dimensioni delle patch
    
#     Returns:
#         patches: tensor [B, N, C, patch_h, patch_w]
#     """
#     B, C, H, W = img.shape
    
#     # Unfold per estrarre patch
#     patches = img.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
#     # patches shape: [B, C, grid_h, grid_w, patch_h, patch_w]
    
#     grid_h, grid_w = patches.shape[2], patches.shape[3]
#     patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, grid_h, grid_w, C, patch_h, patch_w]
#     patches = patches.contiguous().view(B, grid_h * grid_w, C, patch_h, patch_w)
    
#     return patches

# config = SimpleNamespace(
#         # Dimensioni immagine / patch
#         img_size = (224, 224),
#         patch_size = (16, 16),
#         channels = 3,
#         n_mel_bins = 224,
#         patch_embedding_dropout = 0.0,
#         num_channels = 3,

#         # MAE Encoder
#         enc_embed_dim = 768,
#         enc_mlp_layer_dim = 3072,
#         enc_hidden_layers = 6,
#         enc_attention_heads = 12,
#         enc_layer_norm_eps = 1e-6,
#         enc_attention_dropout = 0.0,
#         enc_mlp_ratio = 4,

#         # MAE Decoder
#         dec_hidden_layers = 2,
#         dec_embed_dim = 768,
#         dec_attention_heads = 12,
#         dec_layer_norm_eps = 1e-6,
#         dec_attention_dropout = 0.0,
#         dec_mlp_ratio = 4,
        
#         # Masking
#         masking_strategy = "random",
#         masking_percentage = 0.75,

#         # Training
#         batch_size = 32,
#         initial_lr = 1e-4,
#         weight_decay = 0.05,
#         epochs = 50,
#         print_freq = 50,

#         # Cartelle dataset
#         train_dir = r"tiny-imagenet-200/train",
#         val_dir   = r"tiny-imagenet-200/val",
#         test_dir = r"tiny-imagenet-200/test",
#         # Checkpoint
#         checkpoints_dir = "checkpoints_mae_imagenet"
#     )
# checkpoint = torch.load("C:/Users/admin/Desktop/VS Code/MAE/checkpoints_mae_imagenet/model_best.pth", map_location='cpu')
# model = MAE(config)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()  

# # Preprocessing usato per MAE
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # Carica una sola immagine da test
# test_root = "C:/Users/admin/Desktop/VS Code/tiny-imagenet-200/test/images"
# img_paths = [os.path.join(test_root, f) for f in os.listdir(test_root) if f.endswith('.JPEG')]
# img_path = random.choice(img_paths)
# img = Image.open(img_path).convert("RGB")
# # img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
# img_tensor = transform(img)
# img_tensor = img_tensor.unsqueeze(0)
# print(img_tensor.shape)
# with torch.no_grad():
#     # output: (reconstruction, mask)
#     # modello deve avere una forward che restituisce la ricostruzione
#     _, reconstructed, _, bool_mask = model(img_tensor) # B, num_patches, embed_dim
#     # reconstructed = reconstruct(config, reconstructed, img_tensor)
#     reconstructed = reconstruct(config, reconstructed, img_tensor, bool_mask)
# # funzione per invertire normalizzazione
# def unnormalize(tensor):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#     return tensor * std + mean

# original_img = unnormalize(img_tensor.squeeze(0)).permute(1, 2, 0).numpy()
# recon_img = unnormalize(reconstructed).permute(1, 2, 0).numpy()

# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(original_img.clip(0, 1))
# plt.title("Original")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(recon_img.clip(0, 1))
# plt.title("Reconstructed")
# plt.axis("off")
# plt.show()

# _, reconstructed, _ = model(img_tensor)



# # def reconstruct(config, recon_logits, original):
# #     """
# #     Ricostruisce un'immagine RGB dai logits del decoder MAE.
# #     """
# #     B, N, D = recon_logits.shape
# #     ph, pw = config.patch_size  # es: (16, 16)
# #     c = config.channels         # = 3
# #     H, W = config.img_size      # = (224, 224)

# #     # Check
# #     expected_dim = ph * pw * c
# #     assert D == expected_dim, f"Embed dim {D} non è compatibile con patch RGB {ph}x{pw}x{c}"

# #     # Decodifica: [B, N, D] → [B, N, C, ph, pw]
# #     patches = recon_logits.view(B, N, c, ph, pw)

# #     # Assumiamo che patch siano in ordine raster
# #     grid_h = H // ph
# #     grid_w = W // pw
# #     assert N == grid_h * grid_w, f"Num patch {N} ≠ {grid_h}x{grid_w}"

# #     # Ricompone immagine
# #     patches = patches.view(B, grid_h, grid_w, c, ph, pw)
# #     patches = patches.permute(0, 3, 1, 4, 2, 5)  # [B, C, grid_h, ph, grid_w, pw]
# #     images = patches.contiguous().view(B, c, H, W)

# #     # Mostra la prima immagine
# #     img_np = images[0].permute(1, 2, 0).detach().cpu().numpy()  # [H, W, C]
# #     img_np = img_np.clip(0, 1)  # opzionale, se output non è normalizzato

# #     return images[0] 

import torch
from torchvision import transforms, datasets
from PIL import Image
import os
import random
from mae import MAE # Assicurati che questa importazione sia corretta e che MAE sia definito altrove
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# FUNZIONE DI RICOSTRUZIONE (MODIFICATA PER CHIAREZZA E MNIST)
# --------------------------------------------------
def reconstruct_image_from_patches(config, recon_logits, original_image_tensor, bool_mask):
    """
    Ricostruisce un'immagine combinando patch originali (visibili) e
    patch ricostruite (mascherate) dal modello MAE.

    Args:
        config (SimpleNamespace): Configurazione del modello e dei dati.
        recon_logits (torch.Tensor): Logits delle patch ricostruite dal decoder.
                                     Shape: [B, N, D_patch_flat]
                                     (es. [1, num_patches, patch_height * patch_width * channels])
        original_image_tensor (torch.Tensor): Tensore dell'immagine originale.
                                              Shape: [B, C, H, W]
        bool_mask (torch.Tensor): Maschera booleana che indica le patch mascherate (True).
                                  Shape: [B, N]

    Returns:
        torch.Tensor: Immagine ricostruita (solo la prima del batch). Shape: [C, H, W]
    """
    B, N, D_patch_flat = recon_logits.shape
    ph, pw = config.patch_size  # Dimensioni di una patch (es: (7, 7) per MNIST)
    c = config.num_channels    # Numero di canali (1 per MNIST)
    H, W = config.img_size      # Dimensioni dell'immagine (es: (28, 28) per MNIST)

    # Verifica dimensioni dei logits delle patch ricostruite
    expected_dim = ph * pw * c
    assert D_patch_flat == expected_dim, \
        f"Dimensione D_patch_flat ({D_patch_flat}) non compatibile con patch {ph}x{pw}x{c} (attesa: {expected_dim})"

    grid_h = H // ph
    grid_w = W // pw
    assert N == grid_h * grid_w, \
        f"Numero di patch N ({N}) non corrisponde alla griglia prevista {grid_h}x{grid_w} ({grid_h * grid_w})"

    # 1. Estrai le patch dall'immagine originale
    #    Uscita attesa: [B, N, C, ph, pw]
    original_patches = extract_patches(original_image_tensor, ph, pw)

    # 2. Riformatta recon_logits nel formato delle patch
    #    [B, N, D_patch_flat] -> [B, N, C, ph, pw]
    recon_patches = recon_logits.view(B, N, c, ph, pw)

    # 3. Combina patch originali e ricostruite usando bool_mask
    #    bool_mask: True = patch mascherata (usa ricostruita), False = patch visibile (usa originale)
    #    Espandi bool_mask per il broadcasting: [B, N] -> [B, N, 1, 1, 1]
    mask_expanded = bool_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(recon_patches)
    combined_patches = torch.where(
        mask_expanded,
        recon_patches,    # Usa patch ricostruite dove la maschera è True
        original_patches  # Usa patch originali dove la maschera è False
    )

    # 4. Ricompone l'immagine completa dalle patch combinate
    #    [B, N, C, ph, pw] -> [B, grid_h, grid_w, C, ph, pw]
    combined_patches_gridded = combined_patches.view(B, grid_h, grid_w, c, ph, pw)
    #    Permuta per raggruppare le patch per l'immagine:
    #    [B, grid_h, grid_w, C, ph, pw] -> [B, C, grid_h, ph, grid_w, pw]
    rearranged_patches = combined_patches_gridded.permute(0, 3, 1, 4, 2, 5)
    #    Collassa le dimensioni delle patch per formare l'immagine continua:
    #    [B, C, grid_h, ph, grid_w, pw] -> [B, C, H, W]
    reconstructed_images = rearranged_patches.contiguous().view(B, c, H, W)

    return reconstructed_images[0]  # Ritorna la prima immagine del batch [C, H, W]

# --------------------------------------------------
# FUNZIONE PER ESTRARRE PATCH
# --------------------------------------------------
def extract_patches(img_tensor, patch_h, patch_w):
    """
    Estrae patch non sovrapposte da un batch di immagini.

    Args:
        img_tensor (torch.Tensor): Tensore delle immagini [B, C, H, W].
        patch_h (int): Altezza della patch.
        patch_w (int): Larghezza della patch.

    Returns:
        torch.Tensor: Tensore delle patch [B, N, C, patch_h, patch_w],
                      dove N è il numero di patch (grid_h * grid_w).
    """
    B, C, H, W = img_tensor.shape

    # .unfold(dimension, size, step)
    # Estrae patch lungo l'altezza (dim 2)
    patches_intermediate = img_tensor.unfold(2, patch_h, patch_h)
    # Ora patches_intermediate ha shape [B, C, grid_h, W, patch_h]

    # Estrae patch lungo la larghezza (dim 3) dalla precedente
    patches_intermediate = patches_intermediate.unfold(3, patch_w, patch_w)
    # Ora patches_intermediate ha shape [B, C, grid_h, grid_w, patch_h, patch_w]

    grid_h = patches_intermediate.shape[2]
    grid_w = patches_intermediate.shape[3]

    # Riordina le dimensioni per avere [B, grid_h, grid_w, C, patch_h, patch_w]
    # Questo raggruppa le patch spazialmente.
    patches = patches_intermediate.permute(0, 2, 3, 1, 4, 5)

    # Collassa grid_h e grid_w in un'unica dimensione N (numero di patch)
    # Risultato: [B, N, C, patch_h, patch_w]
    patches = patches.contiguous().view(B, grid_h * grid_w, C, patch_h, patch_w)

    return patches

# --------------------------------------------------
# CONFIGURAZIONE PER MNIST
# --------------------------------------------------
config = SimpleNamespace(
    # Dimensioni immagine / patch per MNIST (28×28 grayscale)
    img_size = (28, 28),
    patch_size = (4, 4),       # es. patch 7×7 → 4×4 patches (28/7=4)
    num_channels = 1,
    # Parametri Encoder (modifica se necessario)
    enc_embed_dim = 768,
    enc_mlp_layer_dim = 512,
    enc_hidden_layers = 12,
    enc_attention_heads = 12,
    enc_layer_norm_eps = 1e-6,
    enc_attention_dropout = 0.0,
    enc_mlp_ratio = 4,
    # Parametri Decoder (modifica se necessario)
    dec_hidden_layers = 8,
    dec_embed_dim = 512,
    dec_attention_heads = 16,
    dec_layer_norm_eps = 1e-6,
    dec_attention_dropout = 0.0,
    dec_mlp_ratio = 4,
    # Masking
    masking_strategy = "random",
    masking_percentage = 0.75,

    # Checkpoint (MODIFICA QUESTO PERCORSO!)
    checkpoints_dir="./checkpoints_mae_mnist", # Cartella per i checkpoint MNIST
    checkpoint_path="C:/Users/admin/Desktop/VS Code/checkpoints_mae_mnist/model_best.pth" # Esempio: checkpoints_mae_mnist/model_best.pth
)

# Assicurati che la cartella dei checkpoint esista se vuoi salvare/caricare da lì
# os.makedirs(config.checkpoints_dir, exist_ok=True)

# --------------------------------------------------
# CARICAMENTO MODELLO (Assumendo che MAE sia definito)
# --------------------------------------------------
# Istanzia il modello MAE con la configurazione per MNIST
# Assicurati che la tua classe MAE sia definita e importata correttamente
# from mae import MAE
try:
    model = MAE(config) # Passa l'intera config al costruttore di MAE
except NameError:
    print("La classe MAE non è definita. Assicurati di averla importata correttamente.")
    exit()
except Exception as e:
    print(f"Errore durante l'istanza di MAE: {e}")
    print("Verifica che la definizione di MAE sia compatibile con la 'config' fornita.")
    exit()

# Carica il checkpoint (se esiste e il percorso è corretto)
# DEVI AVERE UN CHECKPOINT ADDESTRATO SU MNIST!
if os.path.exists(config.checkpoint_path):
    try:
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint caricato da: {config.checkpoint_path}")
    except FileNotFoundError:
        print(f"ATTENZIONE: File checkpoint non trovato in {config.checkpoint_path}. Il modello userà pesi inizializzati a caso.")
    except KeyError:
        print(f"ATTENZIONE: 'state_dict' non trovato nel checkpoint. Il modello userà pesi inizializzati a caso.")
    except Exception as e:
        print(f"ATTENZIONE: Errore nel caricamento del checkpoint: {e}. Il modello userà pesi inizializzati a caso.")
else:
    print(f"ATTENZIONE: File checkpoint non trovato in {config.checkpoint_path}. Il modello userà pesi inizializzati a caso.")

model.eval() # Modalità valutazione

# --------------------------------------------------
# TRASFORMAZIONI PER MNIST
# --------------------------------------------------
# Media e deviazione standard di MNIST
mnist_mean = (0.1307,)
mnist_std = (0.3081,)

transform_mnist = transforms.Compose([
    transforms.ToTensor(),                     # Converte PIL Image o ndarray in FLoatTensor e normalizza a [0.0, 1.0]
    transforms.Normalize(mnist_mean, mnist_std) # Normalizza con media e std di MNIST
])

# --------------------------------------------------
# CARICAMENTO DI UN'IMMAGINE RANDOM DA MNIST TEST SET
# --------------------------------------------------
# Scarica MNIST se non presente e carica il dataset di test
try:
    mnist_test_dataset = datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform_mnist)
    # Prendi un'immagine casuale dal dataset di test
    random_idx = random.randint(0, len(mnist_test_dataset) - 1)
    img_tensor, _ = mnist_test_dataset[random_idx]  # img_tensor è già trasformato
    img_tensor = img_tensor.unsqueeze(0)          # Aggiungi dimensione batch: [1, 1, 28, 28]
    print(f"Forma del tensore dell'immagine MNIST caricata: {img_tensor.shape}")
except Exception as e:
    print(f"Errore durante il caricamento del dataset MNIST: {e}")
    exit()

# --------------------------------------------------
# RICOSTRUZIONE E VISUALIZZAZIONE
# --------------------------------------------------
with torch.no_grad():
    # L'output del modello MAE dovrebbe essere:
    # 1. target_patches (patch originali mascherate e non) - non sempre usato direttamente qui
    # 2. recon_logits (predizioni del decoder per le patch mascherate, o tutte le patch)
    # 3. class_logits (se presenti, per task di classificazione ausiliari) - non usato qui
    # 4. bool_mask (la maschera booleana usata)
    # Adatta questa chiamata in base a ciò che il tuo modello MAE restituisce.
    # Per la funzione `reconstruct_image_from_patches`, ci servono `recon_logits` e `bool_mask`.
    # `img_tensor` è l'originale completo.
    try:
        # Esempio di output atteso dal modello MAE:
        # (qualcosa, logits_ricostruiti, qualcosaltro, maschera_booleana)
        # Assicurati che l'ordine corrisponda a quello del tuo modello MAE.
        # Se il tuo MAE restituisce solo (logits_ricostruiti, maschera_booleana):
        # recon_logits, bool_mask = model(img_tensor)
        # Altrimenti, se restituisce più output:
        _, recon_logits, class_logits, bool_mask = model(img_tensor)

    except Exception as e:
        print(f"Errore durante la forward pass del modello: {e}")
        print("Verifica l'output del tuo modello MAE e la chiamata model(img_tensor).")
        exit()
    
    # La funzione `reconstruct_image_from_patches` combina le patch originali con quelle ricostruite.
    reconstructed_output_tensor = reconstruct_image_from_patches(
        config,
        recon_logits, # Logits delle patch dal decoder
        img_tensor,              # Immagine originale completa per le patch non mascherate
        bool_mask     # Maschera per decidere quali patch usare
    )

# Funzione per invertire la normalizzazione per la visualizzazione
def unnormalize_mnist(tensor, mean, std):
    # Tensor: [C, H, W]
    # Mean e Std dovrebbero essere tuple o liste
    mean_tensor = torch.tensor(mean).view(len(mean), 1, 1)
    std_tensor = torch.tensor(std).view(len(std), 1, 1)
    return tensor * std_tensor + mean_tensor

# Prepara le immagini per la visualizzazione
original_img_display = unnormalize_mnist(img_tensor.squeeze(0), mnist_mean, mnist_std)
original_img_display_np = original_img_display.permute(1, 2, 0).numpy() # [H, W, C]

reconstructed_img_display = unnormalize_mnist(reconstructed_output_tensor, mnist_mean, mnist_std)
reconstructed_img_display_np = reconstructed_img_display.permute(1, 2, 0).numpy() # [H, W, C]


# Visualizza
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
# Per MNIST (1 canale), usa cmap='gray' e rimuovi l'ultimo squeeze se necessario
plt.imshow(original_img_display_np.squeeze(), cmap='gray', vmin=0, vmax=1)
plt.title("Originale MNIST")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img_display_np.squeeze(), cmap='gray', vmin=0, vmax=1)
plt.title("Ricostruita MAE")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Visualizzazione completata.")