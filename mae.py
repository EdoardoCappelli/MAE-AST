import torch
import torch.nn as nn
from config import Config
from positional_embedding import SinusoidalPositionalEncoding
from timm.models.vision_transformer import Block
from masking import Mask
from losses import infoNCE_loss, mae_loss

# from types import SimpleNamespace 

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
        self.pretraining = pretraining
        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)
        self.unfold = nn.Unfold(kernel_size=(config.patch_size[0], config.patch_size[1]),
                                stride=(config.patch_size[0], config.patch_size[1]))

        self.proj_patches = nn.Linear(config.patch_size[0] * config.patch_size[1], config.enc_embed_dim)

        self.mask = Mask(config) 

        # CLS tokens
        self.cls_token_enc = nn.Parameter(torch.zeros(1, 1, self.enc_embed_dim))
        self.cls_token_dec = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        
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
                self.patch_size[0] * self.patch_size[1] * self.num_channels,
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
        # Initialize CLS tokens
        torch.nn.init.trunc_normal_(self.cls_token_enc, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token_dec, std=0.02)
        
        # Initialize mask embedding
        if self.pretraining:
            torch.nn.init.trunc_normal_(self.decoder_mask_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        # Batch Normalization
        x = x.unsqueeze(1)
        x = self.batch_norm(x) * 0.5 

        # Extract patches
        original_patches = self.unfold(x).transpose(-1, -2)
        patch_embedding = self.proj_patches(original_patches)

        # patch_embedding = self.patch_embedding(x)
        with open("C:/Users/admin/Desktop/VS Code/MAE/log.txt", "a") as f:
            f.write("PatchEmbed:\n")
            f.write(str(patch_embedding))
            f.write(str(patch_embedding.shape))
            f.write("\n---------------------------\n")

        # Encoder positional encoding
        pos_embed_enc = SinusoidalPositionalEncoding(
            embed_dim=self.enc_embed_dim, 
            height=int(self.img_size[0]/self.patch_size[0]), 
            width=int(self.img_size[1]/self.patch_size[1]), 
            cls_token=True).to(device)
        pos_embed_enc = pos_embed_enc.expand(x.shape[0], -1, -1)
        
        # Add positional encoding to patches (excluding CLS position)
        patch_embedding_with_pe = patch_embedding + pos_embed_enc[:, 1:, :]
        
        # Apply masking
        (
            unmasked_patches_only, 
            bool_mask,
            masked_indices, 
            unmasked_indices 
        ) = self.mask(patch_embedding_with_pe)

        # Add CLS token to encoder input
        cls_token = self.cls_token_enc + pos_embed_enc[:, :1, :]
        cls_tokens = cls_token.expand(unmasked_patches_only.shape[0], -1, -1)
        enc_input = torch.cat((cls_tokens, unmasked_patches_only), dim=1)
        
        # Encode
        unmasked_embeddings = self.encoder(enc_input)

        if self.pretraining:
            # Project to decoder space
            unmasked_embeddings_projected = self.project_into_decoder_space(unmasked_embeddings)
            
            # Prepare decoder input
            decoder_input = []
            num_patches = original_patches.shape[1]

            for b in range(x.shape[0]):
                batch_decoder_input = torch.zeros(
                    (num_patches, self.dec_embed_dim), 
                    device=device
                )

                # Place unmasked embeddings (skip CLS token from encoder)
                batch_decoder_input[unmasked_indices[b]] = unmasked_embeddings_projected[b, 1:, :]

                # Place mask embeddings
                mask_emb = self.decoder_mask_emb.unsqueeze(0).expand(len(masked_indices[b]), -1)
                batch_decoder_input[masked_indices[b]] = mask_emb

                decoder_input.append(batch_decoder_input)
            
            decoder_input = torch.stack(decoder_input)
            
            # Decoder positional encoding
            pos_embed_dec = SinusoidalPositionalEncoding(
                embed_dim=self.dec_embed_dim, 
                height=int(self.img_size[0]/self.patch_size[0]), 
                width=int(self.img_size[1]/self.patch_size[1]), 
                cls_token=True).to(device)
            pos_embed_dec = pos_embed_dec.expand(x.shape[0], -1, -1)

            # Add positional encoding to decoder input
            decoder_input = decoder_input + pos_embed_dec[:, 1:, :]

            # Add CLS token to decoder input (FIXED: use pos_embed_dec)
            cls_token = self.cls_token_dec + pos_embed_dec[:, :1, :]
            cls_tokens = cls_token.expand(decoder_input.shape[0], -1, -1)
            decoder_input = torch.cat((cls_tokens, decoder_input), dim=1)
            
            # Decode
            decoder_output = self.decoder(decoder_input)
            decoder_output = decoder_output[:, 1:, :]  # Remove CLS token
            
            # Prendo solo le patch che erano mascherate
            masked_decoder_output = decoder_output[bool_mask]  # (num_total_masked_patches, dec_embed_dim)
            masked_decoder_output = masked_decoder_output.view(x.shape[0], -1, self.dec_embed_dim) # (B, num_masked_per_image, dec_embed_dim)

            # Reconstruct patches
            recon_logits_masked = self.final_recon_proj(masked_decoder_output)
            class_logits_masked = self.final_class_proj(masked_decoder_output)

            # Prepare target patches
            # target_patches = patch_extract(original_x, self.patch_size)
            target_patches = original_patches
            target_masked = target_patches[bool_mask].view(x.shape[0], -1, recon_logits_masked.shape[-1])

            return target_masked, recon_logits_masked, class_logits_masked, bool_mask
            
        else:
            # Fine-tuning: use CLS token for classification
            cls_output = unmasked_embeddings[:, 0, :]  # Extract CLS token
            class_logits = self.classification_head(cls_output)
            
            return class_logits


def run_mae_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    config = Config()

    # --- Test per la Modalità Pre-training ---
    # print("\n## Test MAE in Modalità Pre-training ##")

    mae_pt_model = MAE(config, pretraining=True).to(device)
    mae_pt_model.eval()

    B = 2 # Batch size
    # dummy_images = torch.randn(B, config.num_channels, config.img_size[0], config.img_size[1]).to(device)
    dummy_images1 = torch.arange(
        start=0,
        end=config.img_size[0] * config.img_size[1],
        dtype=torch.float32
    ).view(1, config.img_size[0], config.img_size[1])  
    
    dummy_images2 = torch.arange(
        start=config.img_size[0] * config.img_size[1],
        end=config.img_size[0] * config.img_size[1] * 2,
        dtype=torch.float32
    ).view(1, config.img_size[0], config.img_size[1])  
    
    dummy_images = torch.stack([dummy_images1, dummy_images2])
    
    with torch.no_grad():
        target, recon_logits, class_logits, bool_mask = mae_pt_model(dummy_images)

    print("Forward Pass in Modalità Pre-training Riuscito!")
    print(f"  Shape input image: {dummy_images.shape}")
    print(f"  Shape target: {target.shape}")
    print(f"  Shape logits ricostruzione: {recon_logits.shape}")
    print(f"  Shape bool_mask (pretrain): {bool_mask.shape}")

    # target_w_ex = patch_extract(dummy_images, config.patch_size)
    # print(f"target with patch extractor:\t\t{target_w_ex.shape}")
    # print(f"target with patch extractor:\t\t{target_w_ex}")
    # print(f"Target: {target}")

    loss_recon = mae_loss(target, recon_logits)
    loss_info_nce = infoNCE_loss(target, class_logits)
    loss = loss_recon + loss_info_nce

    print(f"MAE Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_mae_tests()






























