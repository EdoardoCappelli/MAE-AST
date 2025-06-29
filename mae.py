import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import SinusoidalPositionalEncoding
from timm.models.vision_transformer import Block
from utils.masking import Mask
from utils.losses import infoNCE_loss, mae_loss 
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import random
import math  
from PIL import Image
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose, ToTensor, ToPILImage
import os
from colorama import Fore, Style, init  
from config import Config  
import argparse 
from utils.plot import visualize_spectrogram_patches_with_masking, visualize_spectrogram_reconstruction, crop_or_repeat

init(autoreset=True) # Inizializza colorama

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
    def __init__(self, args: argparse,config: Config, pretraining=True):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.enc_embed_dim = config.enc_embed_dim
        self.dec_embed_dim = config.dec_embed_dim
        self.num_channels = config.num_channels
        self.base_patch_size = config.patch_size
        self.learnable_pos_emb = config.learnable_pos_emb
        self.use_cls = config.use_cls
        self.pretraining = pretraining
        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)  

        if args.masking_strategy == "patch":
            print(f"{Fore.CYAN}Strategia di Patching: 'patch'{Style.RESET_ALL}")
            self.effective_patch_size = self.base_patch_size
            ph, pw = self.effective_patch_size
            patch_dim = ph * pw * self.num_channels

        elif args.masking_strategy == "frame":
            print(f"{Fore.CYAN}Strategia di Patching: 'frame'{Style.RESET_ALL}")
            ph = self.img_size[0]
            pw = self.base_patch_size[1]
            self.effective_patch_size = (ph, pw)
            patch_dim = ph * pw * self.num_channels
        else:
            raise ValueError(f"Strategia di masking '{args.masking_strategy}' non supportata. Scegliere 'patch' o 'frame'.")

        self.unfold = nn.Unfold(kernel_size=self.effective_patch_size, stride=self.effective_patch_size)
        self.proj_patches = nn.Linear(patch_dim, config.enc_embed_dim)

        n_patches_h = self.img_size[0] // self.effective_patch_size[0]
        n_patches_w = self.img_size[1] // self.effective_patch_size[1]
        num_patches = n_patches_h * n_patches_w

        if self.pretraining:
            self.mask = Mask(args, config)

        if self.use_cls:
            self.cls_token_enc = nn.Parameter(torch.zeros(1, 1, self.enc_embed_dim))
            if pretraining:  
                self.cls_token_dec = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))

        enc_pos_embed_size = num_patches + (1 if self.use_cls else 0)
        dec_pos_embed_size = num_patches + (1 if self.use_cls else 0)

        if self.learnable_pos_emb:
            self.pos_embed_enc = nn.Parameter(torch.zeros(1, enc_pos_embed_size, self.enc_embed_dim))
            if pretraining:
                self.pos_embed_dec = nn.Parameter(torch.zeros(1, dec_pos_embed_size, self.dec_embed_dim))
        else:
            # Sinusoidal positional encoding
            self.pos_embed_enc = SinusoidalPositionalEncoding(
                embed_dim=self.enc_embed_dim,
                height=n_patches_h,
                width=n_patches_w,
                cls_token=self.use_cls
            )
            if pretraining:
                self.pos_embed_dec = SinusoidalPositionalEncoding(
                    embed_dim=self.dec_embed_dim,
                    height=n_patches_h,
                    width=n_patches_w,
                    cls_token=self.use_cls
                )

            self.register_buffer('pos_embed_enc_fixed', self.pos_embed_enc)
            if pretraining:
                self.register_buffer('pos_embed_dec_fixed', self.pos_embed_dec)

        self.encoder = MAE_Encoder(config)

        if pretraining:
            # Decoder 
            self.project_into_decoder_space = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
            self.decoder_mask_emb = nn.Parameter(torch.FloatTensor(self.dec_embed_dim).uniform_())
            self.decoder = MAE_Decoder(config)

            self.final_recon_proj = nn.Linear(
                self.dec_embed_dim,
                patch_dim,
                bias=True
            )
            self.final_class_proj = nn.Linear(
                self.dec_embed_dim,
                patch_dim,
                bias=True
            )

        else:
            # Fine-tuning
            if args.dataset == 'voxceleb':
                n_classes = config.n_classes_voxceleb
            elif args.dataset == 'esc':
                n_classes = getattr(config, 'n_classes_esc', config.n_classes_esc)
            else:
                raise ValueError(f"Dataset di fine-tuning '{args.dataset}' non riconosciuto. Configurare n_classes.")

            self.classification_head = nn.Linear(
                self.enc_embed_dim,
                n_classes,
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

        # Initialize learnable positional embeddings if used
        if self.learnable_pos_emb:
            torch.nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)
            if self.pretraining:
                torch.nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

        # Initialize mask embedding
        if self.pretraining:
            torch.nn.init.trunc_normal_(self.decoder_mask_emb, std=0.02)

        # Initialize classification head for fine-tuning
        if not self.pretraining and hasattr(self, 'classification_head'):
            nn.init.xavier_uniform_(self.classification_head.weight)
            if self.classification_head.bias is not None:
                nn.init.constant_(self.classification_head.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device # (B, num_channels, H, W)
        B, C, H, W = x.shape

        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension

        if W != self.img_size[1]:
            x = crop_or_repeat(x.squeeze(1), target_width=self.img_size[1]).unsqueeze(1) 
            
        # Batch Normalization
        x = self.batch_norm(x) * 0.5

        original_patches = self.unfold(x).transpose(-1, -2) # (B, num_patches, patch_dim)

        ph, pw = self.effective_patch_size
        n_patches_h = self.img_size[0] // ph
        n_patches_w = self.img_size[1] // pw

        original_patches_reshaped = original_patches.contiguous().view(B, n_patches_h, n_patches_w, -1)
        original_patches_flipped = torch.flip(original_patches_reshaped, dims=[1]) # Flip along height dimension (index 1)
        original_patches_flat = original_patches_flipped.contiguous().view(B, n_patches_h * n_patches_w, -1)

        patch_embedding = self.proj_patches(original_patches_flat) # (B, num_patches, enc_embed_dim)

        # Add positional embedding
        if self.learnable_pos_emb:
            pos_embed_enc_to_use = self.pos_embed_enc
        else: # Sinusoidal PE
            pos_embed_enc_to_use = self.pos_embed_enc_fixed

        if self.use_cls:
            # Positional embedding for patches (excluding CLS position)
            patch_embedding_with_pe = patch_embedding + pos_embed_enc_to_use[:, 1:, :].expand(B, -1, -1)
        else:
            patch_embedding_with_pe = patch_embedding + pos_embed_enc_to_use.expand(B, -1, -1)

        if self.pretraining:
            (
                unmasked_patches_only,
                bool_mask,
                masked_indices,
                unmasked_indices
            ) = self.mask(patch_embedding_with_pe)

            if self.use_cls:
                cls_token = self.cls_token_enc + pos_embed_enc_to_use[:, :1, :]
                cls_tokens = cls_token.expand(unmasked_patches_only.shape[0], -1, -1)
                enc_input = torch.cat((cls_tokens, unmasked_patches_only), dim=1)
            else:
                enc_input = unmasked_patches_only

            # Encode
            unmasked_embeddings = self.encoder(enc_input)

            # Project to decoder space
            unmasked_embeddings_projected = self.project_into_decoder_space(unmasked_embeddings)

            # Prepare decoder input
            num_patches_total = original_patches_flat.shape[1]

            start_idx_enc_output = 1 if self.use_cls else 0
            
            unmasked_indices_tensor = torch.stack(unmasked_indices)
            full_decoder_input = self.decoder_mask_emb.unsqueeze(0).unsqueeze(0).expand(B, num_patches_total, -1)
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, unmasked_indices_tensor.shape[1])
            unmasked_embeddings_to_place = unmasked_embeddings_projected[:, start_idx_enc_output:, :]
            full_decoder_input = full_decoder_input.scatter(1, unmasked_indices_tensor.unsqueeze(-1).expand(-1, -1, self.dec_embed_dim), unmasked_embeddings_to_place)

            decoder_input = full_decoder_input
            
            # Decoder positional encoding
            if self.learnable_pos_emb:
                pos_embed_dec_to_use = self.pos_embed_dec
            else:
                pos_embed_dec_to_use = self.pos_embed_dec_fixed

            # Add positional encoding to decoder input
            if self.use_cls:
                decoder_input = decoder_input + pos_embed_dec_to_use[:, 1:, :].expand(B, -1, -1)
            else:
                decoder_input = decoder_input + pos_embed_dec_to_use.expand(B, -1, -1)

            if self.use_cls:
                cls_token_dec = self.cls_token_dec + pos_embed_dec_to_use[:, :1, :]
                cls_tokens_dec = cls_token_dec.expand(decoder_input.shape[0], -1, -1)
                decoder_input = torch.cat((cls_tokens_dec, decoder_input), dim=1)

            # Decode
            decoder_output = self.decoder(decoder_input)

            if self.use_cls:
                decoder_output = decoder_output[:, 1:, :] # Rimuovi il token CLS del decoder prima della ricostruzione

            # Prendo solo le patch che erano mascherate
            masked_decoder_output = decoder_output[bool_mask.view(B, -1)].view(B, -1, self.dec_embed_dim)

            recon_logits_masked = self.final_recon_proj(masked_decoder_output)
            class_logits_masked = self.final_class_proj(masked_decoder_output)

            target_patches = original_patches_flat # Use the flat, potentially flipped patches as target
            target_masked = target_patches[bool_mask.view(B, -1)].view(B, -1, recon_logits_masked.shape[-1])

            # visualize_spectrogram_patches_with_masking(
            #     original_patches=original_patches_flat,
            #     spectrogram=x,
            #     patch_size=self.effective_patch_size, 
            #     bool_mask=bool_mask,
            #     masked_indices=masked_indices,
            #     batch_idx=0,
            #     config=self.config
            # )
            # visualize_spectrogram_reconstruction(
            #     original_spectrogram=x,
            #     reconstructed_patches=recon_logits_masked, # Passa i logits ricostruiti
            #     bool_mask=bool_mask,
            #     patch_size=self.effective_patch_size, # Usa la patch size effettiva
            #     batch_idx=0
            # )

            return target_masked, recon_logits_masked, class_logits_masked, bool_mask

        else: # Fine-tuning 
            # Per il fine-tuning, non viene applicato il masking.
            # Tutti gli embedding delle patch passano attraverso l'encoder.
            if self.use_cls:
                cls_token = self.cls_token_enc + pos_embed_enc_to_use[:, :1, :]
                cls_tokens = cls_token.expand(patch_embedding_with_pe.shape[0], -1, -1)
                encoder_input_for_ft = torch.cat((cls_tokens, patch_embedding_with_pe), dim=1)
            else:
                encoder_input_for_ft = patch_embedding_with_pe

            features_from_encoder = self.encoder(encoder_input_for_ft)

            if self.use_cls:
                # Uso token CLS per la classificazione
                features = features_from_encoder[:, 0, :]
            else:
                # Se non c'è il token CLS, usa il pooling medio di tutti gli embedding delle patch
                features = features_from_encoder.mean(dim=1)

            class_logits = self.classification_head(features)
            return class_logits
        
if __name__ == "__main__":

    from mae import MAE  
    from config import Config 
    import argparse
    from torchinfo import summary

    argparser = argparse.ArgumentParser(description="MAE Pre-training Script")
    argparser.add_argument('--masking_strategy', default='patch', type=str, choices=['patch', 'frame'], help='Masking strategy to use: "patch" or "frame"')
    argparser.add_argument('--epochs', type=int, default=16, help='Number of epochs to train the model')
    args = argparser.parse_args()

    config = Config()

    model = MAE(args, config)
    summary(model, input_size=(1, 1, 128, 1024))
