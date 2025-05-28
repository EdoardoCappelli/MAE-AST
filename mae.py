import torch
import torch.nn as nn
from config import Config
from positional_embedding import SinusoidalPositionalEncoding
from timm.models.vision_transformer import Attention, Block
from timm.layers.patch_embed import PatchEmbed
from masking import Mask
import math
from types import SimpleNamespace 

'''
FINETUNING encoder only
----------------------------


PRETRAINING encoder-decoder
----------------------------

PIPELINE:

spectrogram
    batch norm
    patch embedding
    patch masking
    positional encoding
    visible patches + pe --> MAE_Encoder 
    
    IF FINETUNING:
        mean pooling
        classification head
    
    IF PRETRAINING:
        linear projection into decoder space
        encoder output + masked patches 
        positional encoding
        visible patches + masked patches + pe --> MAE_Decoder
        linear head for recostruction loss
        linear head for infoNCE loss

----------------------------

'''
def patch_extract(x):
    # x: (B, C, H, W)
    config = Config()
    B, C, H, W = x.shape
    P = config.patch_size
    assert H % P[0] == 0 and W % P[1] == 0, "Image size must be divisible by patch size"

    patches = x.unfold(2, P[0], P[1]).unfold(3, P[0], P[1])  # (B, C, H//P, W//P, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, H//P, W//P, C, P, P)
    patches = patches.reshape(B, -1, C * P[0] * P[1])  # (B, num_patches, patch_dim)

    return patches

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

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

        self.batch_norm = nn.BatchNorm2d(num_features=self.num_channels, affine=False)
        self.patch_embedding = PatchEmbed(
            self.img_size,
            self.patch_size,
            self.num_channels,
            self.enc_embed_dim
        ) # B, num_patch, embed_dim
        self.mask = Mask(config) 
        self.cls_token_enc = nn.Parameter(torch.zeros(1, 1, self.enc_embed_dim))
        self.cls_token_dec = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        self.encoder = MAE_Encoder(config) 
        
        if pretraining:
            self.project_into_decoder_space = nn.Linear(self.enc_embed_dim, self.dec_embed_dim)
            self.decoder_mask_emb = nn.Parameter(torch.FloatTensor(self.dec_embed_dim).uniform_()) # è l'embedding che rappresenta la patch mascherata e dovrebbe essere appresa durante il training
            self.decoder = MAE_Decoder(config)
            
            self.final_proj_reconstruction = nn.Linear(
                self.dec_embed_dim,
                self.patch_size[0] * self.patch_size[1] * self.num_channels,
                bias=True
            )
            self.final_proj_classification = nn.Linear(
                self.dec_embed_dim,
                self.patch_size[0] * self.patch_size[1] * self.num_channels,
                bias=True
            )
        else:
            self.mean_pooling = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
            self.classification_head = nn.Linear(
                self.dec_embed_dim,
                self.patch_size[0] * self.patch_size[1] * self.num_channels,
                bias=True
            )    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        original_x = x.detach().clone()
        # print(f"x:\t\t\t{x.shape}")
        x = self.batch_norm(x)
        patch_embedding = self.patch_embedding(x)
        # print(f"patch_embedding:\t{patch_embedding.shape}")
        # print(f"num_patches:\t{self.patch_embedding.num_patches}")
        
        pos_embed_enc = SinusoidalPositionalEncoding(
            embed_dim=self.enc_embed_dim, 
            height=int(self.img_size[0]/self.patch_size[0]), 
            width=int(self.img_size[1]/self.patch_size[1]), 
            cls_token=True).to(device)  # [num_patches, embed_dim]
        pos_embed_enc = pos_embed_enc.expand(x.shape[0], -1, -1)
        pos_embed_enc = pos_embed_enc.to(device)
        # print(f"pos_embed_enc:\t\t{pos_embed_enc.shape}")
        patch_embedding_with_pe = patch_embedding + pos_embed_enc[:,1:,:] # in prima posiziono cls token
        # print(f"patch_embed_with_pe:\t{patch_embedding_with_pe.shape}")
        
        (
            patch_embeddings_with_mask_embeddings, 
            unmasked_patches_only, 
            bool_mask,
            masked_indices, 
            unmasked_indices 
        ) = self.mask(patch_embedding_with_pe)

        cls_token = self.cls_token_enc + pos_embed_enc[:, :1, :] # sommo positional encoding al cls token
        cls_tokens = cls_token.expand(unmasked_patches_only.shape[0], -1, -1)
        # print(f"cls_tokens:\t\t{cls_tokens.shape}")
        enc_input = torch.cat((cls_tokens, unmasked_patches_only), dim=1)
        # print(f"enc_input:\t\t{enc_input.shape}")

        unmasked_embeddings = self.encoder(enc_input) # il primo è relativo al CLS
        # print(f"unmasked_embeddings:\t{unmasked_embeddings.shape}")

        if self.pretraining:
            unmasked_embeddings_projected = self.project_into_decoder_space(unmasked_embeddings)
            decoder_input = []
            for b in range(x.shape[0]):
                batch_decoder_input = torch.zeros(
                    (self.patch_embedding.num_patches, self.dec_embed_dim), 
                    device=device
                )

                # Inserisci embeddings encoder nei punti non mascherati
                batch_decoder_input[unmasked_indices[b]] = unmasked_embeddings_projected[b, 1:, :]

                mask_emb = self.decoder_mask_emb.unsqueeze(0).expand(len(masked_indices[b]), -1)
                batch_decoder_input[masked_indices[b]] = mask_emb

                decoder_input.append(batch_decoder_input)
            
            decoder_input = torch.stack(decoder_input)
            # for b in range(x.shape[0]):
            #     x = torch.zeros((self.patch_embedding.num_patches, self.dec_embed_dim), device=unmasked_embeddings_projected[b].device)

            #     # Inserisci embeddings encoder nei punti non mascherati
            #     x[unmasked_indices[b]] = unmasked_embeddings_projected[b, 1:, :] 

            #     # Inserisci decoder_mask_emb nei punti mascherati
            #     x[masked_indices[b]] = self.decoder_mask_emb  # broadcast su più righe

            #     decoder_input.append(x)
            # decoder_input = torch.stack(decoder_input)
            
            pos_embed_dec = SinusoidalPositionalEncoding(
                embed_dim=self.dec_embed_dim, 
                height=int(self.img_size[0]/self.patch_size[0]), 
                width=int(self.img_size[1]/self.patch_size[1]), 
                cls_token=True).to(device) # [num_patches, embed_dim]
            pos_embed_dec = pos_embed_dec.expand(x.shape[0], -1, -1)

            cls_token = self.cls_token_dec + pos_embed_enc[:, :1, :] # sommo positional encoding al cls token
            cls_tokens = cls_token.expand(unmasked_patches_only.shape[0], -1, -1)
            
            decoder_input = torch.cat((cls_tokens, decoder_input), dim=1)
            # print(f"decoder_input:\t\t{decoder_input.shape}")

            decoder_output = self.decoder(decoder_input)
            decoder_output = decoder_output[:, 1:, :]
            # decoder_output = decoder_output.unsqueeze(0).transpose(0,1) # B, num_patch, embed_dim --> B, 1, num_patch, embed_dim
            # print(f"decoder_output:\t\t{decoder_output.shape}")

            recon_logits = self.final_proj_reconstruction(decoder_output)
            class_logits = self.final_proj_classification(decoder_output)
            # print(f"recon_logits:\t\t{recon_logits.shape}")
            # print(f"class_logits:\t\t{class_logits.shape}")
            
            target = patch_extract(original_x)
            return target, recon_logits, class_logits, bool_mask
            
        else:
            unmasked_embeddings = self.mean_pooling(unmasked_embeddings[:,1:,:])
            class_logits = self.classification_head(unmasked_embeddings)
            
            return unmasked_embeddings, class_logits
 
# class PatchExtract(nn.Module):
#     def __init__(self, patch_size=16):
#         super().__init__()
#         self.patch_size = patch_size

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape
#         P = self.patch_size
#         assert H % P == 0 and W % P == 0, "Image size must be divisible by patch size"

#         # Usa unfold per dividere in patch
#         patches = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, H//P, W//P, P, P)
#         patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, H//P, W//P, C, P, P)
#         patches = patches.reshape(B, -1, C * P * P)  # (B, num_patches, patch_dim)

#         return patches



def run_mae_tests():
    # print("Avvio dei test per MAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Utilizzo del dispositivo: {device}")

    # --- Test per la Modalità Pre-training ---
    # print("\n## Test MAE in Modalità Pre-training ##")
    config_pt = SimpleNamespace(
        num_channels = 1,
        patch_embedding_dropout = 0.1,
        img_size =  (16*5, 16*3),
            
        # Masking 
        masking_strategy = "random", # tipo di mascheramento
        masking_percentage = 0.75, # percentuale delle patches che vogliamo mascherare
        
        # Encoder 
        enc_embed_dim=768, # dimensione dell'embedding vector
        enc_mlp_layer_dim=3072, # dimensione linear layer del FFN 3072
        enc_hidden_layers = 6, # layers del vision tranformer
        enc_attention_heads = 1, # heads del multi head attention
        patch_size = (16,16), # ogni immagine verrà divisa in patch 16x16
        enc_layer_norm_eps = 1e-6, 
        enc_attention_dropout = 0.0,
        enc_mlp_ratio = 4,
        
        # Decoder
        dec_hidden_layers = 2,
        dec_embed_dim = 768,
        dec_attention_heads = 1, # heads del multi head attention
        dec_layer_norm_eps = 1e-6, 
        dec_attention_dropout = 0.0,
        dec_mlp_ratio = 4,

        # Training
        batch_size = 32,
        initial_lr = 0.0001,
        lambda_recon = 10,
        epochs = 50,
        weight_decay = 0.05,
        sample_size = 4000,
        dataset_dir = "D:/data/spectrograms/balanced_train_segments",
        checkpoints_dir = "D:/checkpoints",
        npy_dir = "MAE-AST/spectrograms/balanced_train_segments",
        tensor_dir = "MAE-AST/tensors/balanced_train_segments",
    )

    mae_pt_model = MAE(config_pt, pretraining=True).to(device)
    mae_pt_model.eval()

    B = 2 # Batch size
    # dummy_images = torch.randn(B, config_pt.num_channels, config_pt.img_size[0], config_pt.img_size[1]).to(device)
    dummy_images1 = torch.arange(
        start=0,
        end=config_pt.img_size[0] * config_pt.img_size[1],
        dtype=torch.float32
    ).view(1, config_pt.img_size[0], config_pt.img_size[1])  
    
    dummy_images2 = torch.arange(
        start=config_pt.img_size[0] * config_pt.img_size[1],
        end=config_pt.img_size[0] * config_pt.img_size[1] * 2,
        dtype=torch.float32
    ).view(1, config_pt.img_size[0], config_pt.img_size[1])  
    
    dummy_images = torch.stack([dummy_images1, dummy_images2])
    
    with torch.no_grad():
        decoder_output, recon_logits, class_logits = mae_pt_model(dummy_images)

    # print("Forward Pass in Modalità Pre-training Riuscito!")
    # print(f"  Shape input image: {dummy_images.shape}")
    # print(f"  Shape output decoder: {decoder_output.shape}")
    # print(f"  Shape logits ricostruzione: {recon_logits.shape}")
    # print(f"  Shape logits classificazione (pretrain): {class_logits.shape}")

    target = patch_extract(dummy_images)
    # print(f"target:\t\t{target.shape}")

    # --- Test per la Modalità Fine-tuning ---
    # # print("\n## Test MAE in Modalità Fine-tuning ##")
    # config_ft = SimpleNamespace(
    #     img_size=(32, 32), patch_size=(8, 8), num_channels=3,
    #     enc_embed_dim=64, dec_embed_dim=32, # dec_embed_dim non usato dalla head di fine-tuning corretta
    #     masking_percentage=0.0, masking_strategy="none", # Nessun masking in fine-tuning
    #     num_classes=10 
    # )
    
    # mae_ft_model = MAE(config_ft, pretraining=False).to(device)
    # mae_ft_model.eval()

    # dummy_images_ft = torch.randn(B, config_ft.num_channels, config_ft.img_size[0], config_ft.img_size[1]).to(device)

    # try:
    #     with torch.no_grad():
    #         cls_features, class_logits_ft = mae_ft_model(dummy_images_ft)
        
    #     # print("Forward Pass in Modalità Fine-tuning Riuscito!")
    #     # print(f"  Shape input image: {dummy_images_ft.shape}")
    #     # print(f"  Shape features CLS: {cls_features.shape}")
    #     # print(f"  Shape logits classificazione (finetune): {class_logits_ft.shape}")

    #     expected_cls_features_shape = (B, config_ft.enc_embed_dim)
    #     expected_class_logits_ft_shape = (B, config_ft.num_classes)
        
    #     assert cls_features.shape == expected_cls_features_shape, \
    #         f"Errore shape features CLS: {cls_features.shape}, Atteso {expected_cls_features_shape}"
    #     assert class_logits_ft.shape == expected_class_logits_ft_shape, \
    #         f"Errore shape logits classificazione (finetune): {class_logits_ft.shape}, Atteso {expected_class_logits_ft_shape}"
    #     # print("Verifiche Shape Modalità Fine-tuning Superate!")

    # except Exception as e:
    #     # print(f"Errore durante il test MAE fine-tuning: {e}")
    #     traceback.# print_exc()

if __name__ == "__main__":
    run_mae_tests()
