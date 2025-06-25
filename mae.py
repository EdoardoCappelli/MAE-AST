import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from positional_embedding import SinusoidalPositionalEncoding, simple_1d_pe
from timm.models.vision_transformer import Block
from masking import Mask
from losses import infoNCE_loss, mae_loss
import matplotlib.pyplot as plt 
import numpy as np
from typing import Tuple, List
import random
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, ToPILImage

def visualize_spectrogram_patches_with_masking(
    original_patches: torch.Tensor,
    spectrogram: torch.Tensor,
    patch_size: Tuple[int, int] = (32, 32),
    bool_mask: torch.Tensor = None,
    masked_indices: List[torch.Tensor] = None,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (18, 6),
    cmap: str = 'viridis'
) -> None:
    """
    Visualizza spettrogramma originale, patch e patch mascherate usando original_patches dal forward
    """
    # Ensure batch dimension
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(0)

    # Select batch
    spec_batch = spectrogram[batch_idx]  # (C, H, W)
    patches_batch = original_patches[batch_idx]  # (num_patches, patch_dim)

    # Unpack patch size
    ph, pw = patch_size
    num_channels = spec_batch.shape[0]
    num_patches = patches_batch.shape[0]

    # Reshape patches to images
    patches = patches_batch.view(num_patches, num_channels, ph, pw)

    # Spectrogram dims and grid dims
    C, H, W = spec_batch.shape
    n_patches_h = H // ph
    n_patches_w = W // pw
    total_patches = n_patches_h * n_patches_w

    # Plot original spectrogram
    plt.figure(figsize=(12, 2))
    img = spec_batch[0].detach().cpu().numpy()
    plt.imshow(img, cmap=cmap, aspect='auto', origin='lower')
    plt.title('Spettrogramma Originale')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'spectrogram_batch_{batch_idx}.png', bbox_inches='tight')
    plt.show()

    # ——————————————————————————————————————————————————————————————
    # PLOT 2: Tutte le Patch con dimensione corretta
    # ——————————————————————————————————————————————————————————————
    
    # Calcola le dimensioni della figura basandosi sulla patch_size
    # Riduci patch_display_size per adattarsi allo schermo
    patch_display_size = 0.8  # Ridotto da 1.5 a 0.8 pollici per patch
    fig_width = min(n_patches_w * patch_display_size, 12)  # Massimo 16 pollici di larghezza
    fig_height = min(n_patches_h * patch_display_size, 2)  # Massimo 10 pollici di altezza
    
    fig, axes = plt.subplots(n_patches_h, n_patches_w, 
                             figsize=(fig_width, fig_height))
    # fig.suptitle(f'Patches (batch {batch_idx}) - Dimensione: {ph}x{pw}')
    
    # Gestisci il caso di una sola patch
    if n_patches_h == 1 and n_patches_w == 1:
        axes = [axes]
    elif n_patches_h == 1 or n_patches_w == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(total_patches):
        ax = axes[i]
        
        if i < patches.shape[0]:
            # Usa il primo canale della patch
            patch_img = patches[i, 0].detach().cpu().numpy()
            
            # Imposta le dimensioni esatte del plot per rispettare patch_size
            im = ax.imshow(patch_img, cmap=cmap, aspect='equal', origin='lower')
            
            # Forza le dimensioni dell'immagine a essere esattamente patch_size
            ax.set_xlim(0, pw)
            ax.set_ylim(0, ph)
            
        else:
            # Patch vuote per completare la griglia
            ax.set_visible(False)
        
        ax.axis('off')
    
    # Rimuovi spazio tra le subplot per mantenere le proporzioni
    plt.subplots_adjust(left=0.05, bottom=0.3, right=0.9, top=0.95, wspace=0.0, hspace=0.0)
    plt.savefig(f'patches_batch_{batch_idx}.png', bbox_inches='tight', dpi=150)
    plt.show()

    # Plot masked status if available
    if bool_mask is not None or masked_indices is not None:
        # Determine masked indices
        if masked_indices is not None and batch_idx < len(masked_indices):
            masked_idx = masked_indices[batch_idx].cpu().numpy()
        elif bool_mask is not None:
            batch_mask = bool_mask.view(spectrogram.shape[0], -1)[batch_idx]
            masked_idx = torch.where(batch_mask)[0].cpu().numpy()
        else:
            masked_idx = []

        fig, axes = plt.subplots(n_patches_h, n_patches_w, 
                                 figsize=(min(fig_width, 12), min(fig_height, 2)))
        
        # Gestisci il caso di una sola patch
        if n_patches_h == 1 and n_patches_w == 1:
            axes = [axes]
        elif n_patches_h == 1 or n_patches_w == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(total_patches):
            ax = axes[i]
            
            if i < patches.shape[0]:
                patch_img = patches[i, 0].detach().cpu().numpy()
                
                # Colore diverso per patch mascherate
                alpha = 0.3 if i in masked_idx else 1.0
                
                ax.imshow(patch_img, aspect='equal', 
                          alpha=alpha, origin='lower')
                
                # Forza le dimensioni
                ax.set_xlim(0, pw)
                ax.set_ylim(0, ph)
                
                title_color = 'red' if i in masked_idx else 'black'
            else:
                ax.set_visible(False)
            
            ax.axis('off')
        
        plt.subplots_adjust(left=0.05, bottom=0.3, right=0.9, top=0.95, wspace=0.0, hspace=0.0)
        plt.savefig(f'masking_batch_{batch_idx}.png', bbox_inches='tight', dpi=150)
        plt.show()

def visualize_spectrogram_reconstruction(
    original_spectrogram: torch.Tensor,
    reconstructed_patches: torch.Tensor,
    bool_mask: torch.Tensor,
    patch_size: Tuple[int, int] = (32, 32),
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'viridis'
) -> None:
    """
    Visualizza confronto tra spettrogramma originale e ricostruzione
    """
    if original_spectrogram.dim() == 3:
        original_spectrogram = original_spectrogram.unsqueeze(0)

    spec_batch = original_spectrogram[batch_idx]  # (C, H, W)
    ph, pw = patch_size
    C, H, W = spec_batch.shape
    n_patches_h = H // ph
    n_patches_w = W // pw

    # Clone for reconstruction
    recon_spec = spec_batch.clone()

    # Find masked indices
    batch_mask = bool_mask.view(original_spectrogram.shape[0], -1)[batch_idx]
    masked_idx = torch.where(batch_mask)[0].cpu().numpy()

    # Apply reconstructed patches
    for i, pidx in enumerate(masked_idx):
        row = pidx // n_patches_w
        col = pidx % n_patches_w
        y0, x0 = row * ph, col * pw
        if i < reconstructed_patches.shape[1]:
            patch = reconstructed_patches[batch_idx, i].view(C, ph, pw)
            recon_spec[:, y0:y0+ph, x0:x0+pw] = patch

    # Plot comparison con dimensioni corrette
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    imgs = [spec_batch[0], recon_spec[0], torch.abs(spec_batch - recon_spec)[0]]
    titles = ['Originale', 'Ricostruito', 'Differenza']
    cmaps = [cmap, cmap, 'hot']
    
    for i, (img, title, cmap_i) in enumerate(zip(imgs, titles, cmaps)):
        img_np = img.detach().cpu().numpy()
        axs[i].imshow(img_np, cmap=cmap_i, aspect='auto', origin='lower')
        axs[i].axis('off')
        
        # Mantieni le proporzioni originali dello spettrogramma
        axs[i].set_xlim(0, W)
        axs[i].set_ylim(0, H)
    
    plt.tight_layout()
    plt.savefig(f'spectrogram_reconstruction_batch_{batch_idx}.png', bbox_inches='tight', dpi=150)
    plt.show()

def crop_or_repeat(spec: torch.Tensor, target_width: int):
    """
    Applica un crop dalla coda o ripete lo spettrogramma dall'inizio per raggiungere la target_width.

    Args:
        spec (torch.Tensor): Lo spettrogramma di input (Freq, Time).
        target_width (int): La larghezza desiderata per l'output.

    Returns:
        torch.Tensor: Lo spettrogramma trasformato.
    """
    current_width = spec.shape[-1]

    if current_width > target_width:
        # Crop dalla coda: prendi le ultime target_width colonne
        return spec[..., -target_width:]

    elif current_width < target_width:
        # Repeat: calcola quante volte ripetere e gestisci il resto
        repeat_times = target_width // current_width
        remainder = target_width % current_width
        
        # Ripeti lo spettrogramma per il numero di volte necessario
        if repeat_times > 1:
            repeated_spec = spec.repeat(1, repeat_times)
        else:
            repeated_spec = spec 
            
        # Se c'è un resto, aggiungi le prime 'remainder' colonne dall'inizio
        if remainder > 0:
            repeated_spec = torch.cat([repeated_spec, spec[..., :remainder]], dim=-1)
        
        return repeated_spec

    else: 
        return spec

class MAE_Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.enc_embed_dim = config.enc_embed_dim
        self.enc_attention_heads = config.enc_attention_heads
        self.enc_mlp_ratio = config.enc_mlp_ratio
        self.enc_hidden_layers = config.enc_hidden_layers
        self.enc_layer_norm_eps = config.enc_layer_norm_eps
        self.norm_layer = nn.LayerNorm

        self.layers = nn.ModuleList([
            Block(
                dim=self.enc_embed_dim,
                num_heads=self.enc_attention_heads,
                mlp_ratio=self.enc_mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                norm_layer=self.norm_layer
            )
            for _ in range(self.enc_hidden_layers)
        ])

        self.norm = nn.LayerNorm(self.enc_embed_dim, eps=self.enc_layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=0.1, training=self.training) # solo durante il training
        
        for block in self.layers:
            x = block(x)
        
        x = self.norm(x)
        return x

class MAE_Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.dec_embed_dim = config.dec_embed_dim
        self.dec_attention_heads = config.dec_attention_heads
        self.dec_mlp_ratio = config.dec_mlp_ratio
        self.dec_hidden_layers = config.dec_hidden_layers
        self.dec_layer_norm_eps = config.dec_layer_norm_eps
        self.norm_layer = nn.LayerNorm

        self.layers = nn.ModuleList([
            Block(
                dim=self.dec_embed_dim,
                num_heads=self.dec_attention_heads,
                mlp_ratio=self.dec_mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                norm_layer=self.norm_layer
            )
            for _ in range(self.dec_hidden_layers)
        ])

        self.norm = nn.LayerNorm(self.dec_embed_dim, eps=self.dec_layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        
        x = self.norm(x)
        return x

class MAE(nn.Module):
    def __init__(self, config: Config, pretraining=True):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.enc_embed_dim = config.enc_embed_dim
        self.dec_embed_dim = config.dec_embed_dim
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.learnable_pos_emb = config.learnable_pos_emb 
        self.use_cls = config.use_cls # Aggiunto: Flag per l'uso del CLS token
        self.pretraining = pretraining
        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)
        
        self.unfold = nn.Unfold(kernel_size=(config.patch_size[0], config.patch_size[1]), stride=(config.patch_size[0], config.patch_size[1]))
        self.proj_patches = nn.Linear(config.patch_size[0] * config.patch_size[1], config.enc_embed_dim)

        self.mask = Mask(config) 

        # CLS tokens: Inizializza solo se use_cls è True
        if self.use_cls:
            self.cls_token_enc = nn.Parameter(torch.zeros(1, 1, self.enc_embed_dim))
            if pretraining: # Il decoder CLS token serve solo in pretraining
                self.cls_token_dec = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        
        num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # Adjust positional embedding size based on use_cls
        enc_pos_embed_size = num_patches + (1 if self.use_cls else 0)
        dec_pos_embed_size = num_patches + (1 if self.use_cls and pretraining else 0)

        if self.learnable_pos_emb:
            self.pos_embed_enc = nn.Parameter(torch.zeros(1, enc_pos_embed_size, self.enc_embed_dim))
            if pretraining:
                self.pos_embed_dec = nn.Parameter(torch.zeros(1, dec_pos_embed_size, self.dec_embed_dim))
        else:
            pos_embed_enc = SinusoidalPositionalEncoding(
                embed_dim=self.enc_embed_dim, 
                height=int(self.img_size[0]/self.patch_size[0]), 
                width=int(self.img_size[1]/self.patch_size[1]), 
                cls_token=self.use_cls) # Passa use_cls al positional encoding
            
            if pretraining:
                pos_embed_dec = SinusoidalPositionalEncoding(
                    embed_dim=self.dec_embed_dim, 
                    height=int(self.img_size[0]/self.patch_size[0]), 
                    width=int(self.img_size[1]/self.patch_size[1]), 
                    cls_token=self.use_cls) # Passa use_cls al positional encoding
                self.register_buffer('pos_embed_dec', pos_embed_dec)
            
            self.register_buffer('pos_embed_enc', pos_embed_enc)


        # Encoder
        self.encoder = MAE_Encoder(config) 
        
        if pretraining:
            # Decoder components
            self.project_into_decoder_space = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
            self.decoder_mask_emb = nn.Parameter(torch.FloatTensor(self.dec_embed_dim).uniform_())
            self.decoder = MAE_Decoder(config)
            
            self.final_recon_proj = nn.Linear(
                self.dec_embed_dim,
                self.patch_size[0] * self.patch_size[1] * self.num_channels,
                bias=True
            )
            self.final_class_proj = nn.Linear(
                self.dec_embed_dim, 
                self.patch_size[0] * self.patch_size[1] * self.num_channels, # For InfoNCE, output dim should match target dim
                bias=True
            )

        else:
            # For fine-tuning
            self.classification_head = nn.Linear(
                self.enc_embed_dim,  
                config.num_classes,  
                bias=True
            )    

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize CLS tokens conditionally
        if self.use_cls:
            torch.nn.init.trunc_normal_(self.cls_token_enc, std=0.02)
            if self.pretraining:
                torch.nn.init.trunc_normal_(self.cls_token_dec, std=0.02)
        
        if self.learnable_pos_emb:
            torch.nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)
            if self.pretraining:
                torch.nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

        # Initialize mask embedding
        if self.pretraining:
            torch.nn.init.trunc_normal_(self.decoder_mask_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device # (B, num_channels, H, W)
        B, C, H, W = x.shape

        if x.dim() == 3:
            x = x.unsqueeze(1)

        shape_before = x.shape
        x = crop_or_repeat(x,target_width=1024) 
        shape_after = x.shape 

        # Batch Normalization
        x = self.batch_norm(x) * 0.5

        # Extract patches
        original_patches = self.unfold(x).transpose(-1, -2) # (B, num_patches, patch_size[0] * patch_size[1] * num_channels)
        
        n_patches_h = self.img_size[0] // self.patch_size[0]
        n_patches_w = self.img_size[1] // self.patch_size[1]
        # total_patches = n_patches_h * n_patches_w # Not explicitly used here for logic

        original_patches = original_patches.contiguous().view(B, n_patches_h, n_patches_w, -1)
        original_patches = torch.flip(original_patches, dims=[1])
        original_patches = original_patches.contiguous().view(B, n_patches_h * n_patches_w, -1)
        
        patch_embedding = self.proj_patches(original_patches) # (B, num_patches, enc_embed_dim)
        
        # Add positional embed 
        # Adjust index for positional embedding based on CLS token usage
        if self.use_cls:
            patch_embedding_with_pe = patch_embedding + self.pos_embed_enc[:, 1:, :].expand(B, -1, -1)
        else:
            patch_embedding_with_pe = patch_embedding + self.pos_embed_enc.expand(B, -1, -1)
        
        # Apply masking
        (
            unmasked_patches_only, 
            bool_mask,
            masked_indices, 
            unmasked_indices 
        ) = self.mask(patch_embedding_with_pe)

        # Add CLS token to encoder input conditionally
        if self.use_cls:
            cls_token = self.cls_token_enc + self.pos_embed_enc[:, :1, :]
            cls_tokens = cls_token.expand(unmasked_patches_only.shape[0], -1, -1)
            enc_input = torch.cat((cls_tokens, unmasked_patches_only), dim=1)
        else:
            enc_input = unmasked_patches_only
        
        # Encode
        unmasked_embeddings = self.encoder(enc_input)

        if self.pretraining:
            # Project to decoder space
            unmasked_embeddings_projected = self.project_into_decoder_space(unmasked_embeddings)
            
            # Prepare decoder input
            decoder_input = []
            num_patches = original_patches.shape[1]

            # Determine the starting index for embeddings based on CLS token
            start_idx = 1 if self.use_cls else 0

            for b in range(B):
                batch_decoder_input = torch.zeros(
                    (num_patches, self.dec_embed_dim), 
                    device=device
                )

                # Place unmasked embeddings (skip CLS token from encoder output if used)
                batch_decoder_input[unmasked_indices[b]] = unmasked_embeddings_projected[b, start_idx:, :]

                # Place mask embeddings
                mask_emb = self.decoder_mask_emb.unsqueeze(0).expand(len(masked_indices[b]), -1)
                batch_decoder_input[masked_indices[b]] = mask_emb

                decoder_input.append(batch_decoder_input)
            
            decoder_input = torch.stack(decoder_input)
            
            # Decoder positional encoding
            pos_embed_dec = self.pos_embed_dec.expand(B, -1, -1)

            # Add positional encoding to decoder input
            # Adjust index for positional embedding based on CLS token usage for decoder
            if self.use_cls:
                decoder_input = decoder_input + pos_embed_dec[:, 1:, :]
            else:
                decoder_input = decoder_input + pos_embed_dec # No CLS token offset

            # Add CLS token to decoder input conditionally
            if self.use_cls:
                cls_token_dec = self.cls_token_dec + pos_embed_dec[:, :1, :]
                cls_tokens_dec = cls_token_dec.expand(decoder_input.shape[0], -1, -1)
                decoder_input = torch.cat((cls_tokens_dec, decoder_input), dim=1)
            
            # Decode
            decoder_output = self.decoder(decoder_input)
            
            # Remove CLS token from decoder output if used
            if self.use_cls:
                decoder_output = decoder_output[:, 1:, :] 
            
            # Prendo solo le patch che erano mascherate
            masked_decoder_output = decoder_output[bool_mask].view(B, -1, self.dec_embed_dim)

            # Reconstruct patches
            recon_logits_masked = self.final_recon_proj(masked_decoder_output)
            class_logits_masked = self.final_class_proj(masked_decoder_output)

            # Prepare target patches
            target_patches = original_patches
            target_masked = target_patches[bool_mask].view(B, -1, recon_logits_masked.shape[-1])
            
            if shape_before != shape_after:
                visualize_spectrogram_patches_with_masking(
                    original_patches=original_patches,
                    spectrogram=x, 
                    patch_size=self.patch_size,
                    bool_mask=bool_mask,
                    masked_indices=masked_indices,
                    batch_idx=0
                )

            return target_masked, recon_logits_masked, class_logits_masked, bool_mask
            
        else:
            # For fine-tuning, the output is typically from the CLS token
            # or mean pooling if no CLS token is used.
            if self.use_cls:
                # If CLS token is used, typically we take the CLS token's output for classification
                features = unmasked_embeddings[:, 0, :] # Take the CLS token output
            else:
                # If no CLS token, use mean pooling of all patch embeddings
                features = unmasked_embeddings.mean(dim=1) 
            
            # head for classification task (gender identification 0,1)
            class_logits = self.classification_head(features)
            
            return class_logits
    
def run_mae_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # Configurazione per testare con e senza CLS token
    class TestConfig(Config):
        def __init__(self, use_cls_token: bool):
            super().__init__()
            self.use_cls = use_cls_token
            # Assicurati che le altre configurazioni siano appropriate per il test
            self.img_size = (128, 1024)
            self.patch_size = (16, 16)
            self.enc_embed_dim = 256
            self.dec_embed_dim = 128
            self.enc_hidden_layers = 2
            self.dec_hidden_layers = 1
            self.mask_ratio = 0.75
            self.num_classes = 2 # per il fine-tuning

    # --- Test per la Modalità Pre-training con CLS Token ---
    print("\n## Test MAE in Modalità Pre-training (con CLS Token) ##")
    config_cls = TestConfig(use_cls_token=True)
    mae_pt_model_cls = MAE(config_cls, pretraining=True).to(device)
    mae_pt_model_cls.eval()

    B = 2 # Batch size
    dummy_images_cls = torch.randn(B, 1, config_cls.img_size[0], config_cls.img_size[1]).to(device) # Aggiunto canale

    with torch.no_grad():
        target_cls, recon_logits_cls, class_logits_cls, bool_mask_cls = mae_pt_model_cls(dummy_images_cls)

    loss_recon_cls = mae_loss(target_cls, recon_logits_cls)
    loss_info_nce_cls = infoNCE_loss(target_cls, class_logits_cls)
    loss_cls = loss_recon_cls + loss_info_nce_cls

    print(f"Shape target_cls: {target_cls.shape}")
    print(f"Shape recon_logits_cls: {recon_logits_cls.shape}")
    print(f"Shape class_logits_cls: {class_logits_cls.shape}")
    print(f"MAE Pre-training Loss (con CLS): {loss_cls.item():.4f}")

    # --- Test per la Modalità Pre-training senza CLS Token ---
    print("\n## Test MAE in Modalità Pre-training (senza CLS Token) ##")
    config_no_cls = TestConfig(use_cls_token=False)
    mae_pt_model_no_cls = MAE(config_no_cls, pretraining=True).to(device)
    mae_pt_model_no_cls.eval()

    dummy_images_no_cls = torch.randn(B, 1, config_no_cls.img_size[0], config_no_cls.img_size[1]).to(device) # Aggiunto canale

    with torch.no_grad():
        target_no_cls, recon_logits_no_cls, class_logits_no_cls, bool_mask_no_cls = mae_pt_model_no_cls(dummy_images_no_cls)

    loss_recon_no_cls = mae_loss(target_no_cls, recon_logits_no_cls)
    loss_info_nce_no_cls = infoNCE_loss(target_no_cls, class_logits_no_cls)
    loss_no_cls = loss_recon_no_cls + loss_info_nce_no_cls

    print(f"Shape target_no_cls: {target_no_cls.shape}")
    print(f"Shape recon_logits_no_cls: {recon_logits_no_cls.shape}")
    print(f"Shape class_logits_no_cls: {class_logits_no_cls.shape}")
    print(f"MAE Pre-training Loss (senza CLS): {loss_no_cls.item():.4f}")

    # --- Test per la Modalità Fine-tuning con CLS Token ---
    print("\n## Test MAE in Modalità Fine-tuning (con CLS Token) ##")
    config_finetune_cls = TestConfig(use_cls_token=True)
    mae_ft_model_cls = MAE(config_finetune_cls, pretraining=False).to(device)
    mae_ft_model_cls.eval()

    dummy_images_ft_cls = torch.randn(B, 1, config_finetune_cls.img_size[0], config_finetune_cls.img_size[1]).to(device)

    with torch.no_grad():
        class_logits_ft_cls = mae_ft_model_cls(dummy_images_ft_cls)

    print(f"Shape Class Logits Fine-tuning (con CLS): {class_logits_ft_cls.shape}")

    # --- Test per la Modalità Fine-tuning senza CLS Token ---
    print("\n## Test MAE in Modalità Fine-tuning (senza CLS Token) ##")
    config_finetune_no_cls = TestConfig(use_cls_token=False)
    mae_ft_model_no_cls = MAE(config_finetune_no_cls, pretraining=False).to(device)
    mae_ft_model_no_cls.eval()

    dummy_images_ft_no_cls = torch.randn(B, 1, config_finetune_no_cls.img_size[0], config_finetune_no_cls.img_size[1]).to(device)

    with torch.no_grad():
        class_logits_ft_no_cls = mae_ft_model_no_cls(dummy_images_ft_no_cls)

    print(f"Shape Class Logits Fine-tuning (senza CLS): {class_logits_ft_no_cls.shape}")

# Esegui i test per verificare il funzionamento
if __name__ == '__main__':
    run_mae_tests()