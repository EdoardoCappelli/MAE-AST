import torch
import torch.nn as nn
from config import Config
from patch_extractor import PatchEmbedding
import numpy as np
from typing import Optional, Tuple, Dict
from masking import Mask
from positional_encoding import SinusoidalPositionalEncoding
from utils.visualize import visualize_spectrogram, visualize_patches

class Vit_Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim
        self.num_heads = config.num_enc_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.enc_attention_dropout
        
        assert self.embed_dim % self.num_heads == 0, f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, seq_len, _ = hidden_states.size()
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to [B, num_heads, seq_len, head_dim]
        query = query.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # [B, num_heads, seq_len, head_dim]
        
        # Reshape back to [B, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class Vit_MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.intermediate_size = config.enc_embed_dim * 4  # Standard 4x expansion
        self.linear1 = nn.Linear(config.enc_embed_dim, self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size, config.enc_embed_dim)
        self.dropout = nn.Dropout(config.enc_attention_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states
    
class Vit_EncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.enc_embed_dim
        self.self_attn = Vit_Attention(config)  
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.enc_layer_norm_eps)
        self.mlp = Vit_MLP(config) 
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.enc_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm architecture (like in original ViT/MAE)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(hidden_states)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states

class MAE_Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Vit_EncoderLayer(config) for _ in range(config.num_enc_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.enc_embed_dim, eps=config.enc_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Vit_DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.dec_embed_dim
        
        # Create attention module with decoder dimensions
        decoder_config = config
        decoder_config.enc_embed_dim = config.dec_embed_dim  # Use decoder embed dim
        decoder_config.num_enc_attention_heads = config.num_dec_attention_heads
        
        self.self_attn = Vit_Attention(decoder_config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.dec_layer_norm_eps)
        
        # Create MLP with decoder dimensions
        self.mlp = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        ])
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.dec_layer_norm_eps)
        self.dropout = nn.Dropout(config.dec_attention_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(hidden_states)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp[0](hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.mlp[1](hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class MAE_Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Vit_DecoderLayer(config) for _ in range(config.num_dec_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.dec_embed_dim, eps=config.dec_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class MAE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.enc_embed_dim = config.enc_embed_dim
        self.dec_embed_dim = config.dec_embed_dim

        # Input normalization
        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)
        
        # Patch embedding
        self.patch_embeddings = PatchEmbedding(config)

        # Masking
        self.mask = Mask(config)
        
        # Positional embeddings
        self.pos_embed_encoder = SinusoidalPositionalEncoding(embed_dim=self.enc_embed_dim)
        self.pos_embed_decoder = SinusoidalPositionalEncoding(embed_dim=self.dec_embed_dim)
        
        # Encoder
        self.encoder = MAE_Encoder(config)
        
        # Encoder to decoder projection
        if self.enc_embed_dim != self.dec_embed_dim:
            self.enc_to_dec = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
        else:
            self.enc_to_dec = nn.Identity()
        
        # Mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        
        # Decoder
        self.decoder = MAE_Decoder(config)
        
        # Final projection head
        self.decoder_pred = nn.Linear(
            self.dec_embed_dim,
            config.patch_size[0] * config.patch_size[1],
            bias=True
        )
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(_init_weights)

    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x, x_after_proj = self.patch_embeddings(x)  # [B, N, D]
        
        # Add positional embedding
        x = x_after_proj + self.pos_embed_encoder(x_after_proj)
        
        # Masking: keep only visible patches
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply encoder
        x = self.encoder(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.enc_to_dec(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append cls token
        
        # Add positional embedding
        x = x + self.pos_embed_decoder(x)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        print(f"x_masked:\n{x_masked}")
        print(f"mask:\n{mask}")
        print(f"ids_restore:\n{ids_restore}")

        return x_masked, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.config.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def forward(self, spectrogram_values: torch.Tensor, mask_ratio: float = 0.75) -> Dict[str, torch.Tensor]:
        if len(spectrogram_values.shape) == 5:
            spectrogram_values = spectrogram_values.squeeze(dim=2)
        
        # Normalize input
        spectrogram_values = self.batch_norm(spectrogram_values) * 0.5
        
        # Forward encoder
        latent, mask, ids_restore = self.forward_encoder(spectrogram_values, mask_ratio)
        
        # Forward decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Compute loss
        loss = self.forward_loss(spectrogram_values, pred, mask)
        
        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "ids_restore": ids_restore
        }
    
def test_model():
    config = Config()
    model = MAE(config)
    print(model)

    
    # print(spectrogram.shape) # 999, 128

    #---------------------------------------------
    # spectrogram_np = np.load("C:/Users/admin/Desktop/VS Code/VisualTransformer/datasets/numpy/balanced_train_segments/EX_-_6RxZyi30Q.npy")
    # spectrogram = torch.from_numpy(spectrogram_np)
    # spectrogram = torch.arange(
    #     start=0,
    #     end=spectrogram.shape[0] * spectrogram.shape[1],
    #     dtype=torch.float32
    # ).view(spectrogram.shape)  
    # spectrogram = spectrogram.unsqueeze(0).unsqueeze(0) # [B, C, W, H]
    # print(spectrogram.shape)
    # print(spectrogram)
    #---------------------------------------------

    w = 900
    l = 128
    spectrogram = torch.arange(0, w*l, dtype=torch.float32).reshape(w, l)
    spectrogram = spectrogram.unsqueeze(0).unsqueeze(0) # [B, C, W, H]
    # spectrogram = spectrogram.permute(0, 1, 3, 2) # [B, C, H, W]
    # print(spectrogram.shape)
    # print(spectrogram)

    output = model(spectrogram)
    loss = output["loss"]
    pred = output["pred"]
    mask = output["mask"]

    print(f"loss: {loss}")
    print(f"pred: {pred}")
    print(f"mask: {mask}")
    
if __name__ == "__main__":
    test_model()