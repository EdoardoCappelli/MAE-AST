from typing import Optional, Tuple
import torch
import torch.nn as nn


class Config:

    def __init__(
            self,

            # Spectrogram
            n_mel_bins = 128, # numero di bins del mel spectrogram (asse y)
            num_channels = 1,
            patch_embedding_dropout = 0.1,

            # Masking 
            masking_strategy = "random", # tipo di mascheramento
            masking_percentage = 0.75, # percentuale delle patches che vogliamo mascherare
            
            # Encoder 
            enc_embed_dim=768, # dimensione dell'embedding vector
            enc_mlp_layer_dim=3072, # dimensione linear layer del FFN
            num_enc_hidden_layers = 1, # layers del vision tranformer
            num_enc_attention_heads = 12, # heads del multi head attention
            patch_size = (16,16), # ogni immagine verrà divisa in patch 16x16
            enc_layer_norm_eps = 1e-6, 
            enc_attention_dropout = 0.0,

            # Decoder
            num_dec_hidden_layers = 2,
            dec_embed_dim = 768,

            # Training
            batch_size = 32,
            learning_rate = 0.01,
            lambda_recon = 10,
            epochs = 30,
            
            **kwargs
    ):
        super().__init__()

        # Spectrogram
        self.num_channels = num_channels
        self.n_mel_bins = n_mel_bins
        self.patch_embedding_dropout = patch_embedding_dropout
        
        # Masking 
        self.masking_strategy = masking_strategy
        self.masking_percentage = masking_percentage
        
        # Encoder 
        self.enc_embed_dim = enc_embed_dim
        self.enc_mlp_layer_dim = enc_mlp_layer_dim
        self.num_enc_hidden_layers = num_enc_hidden_layers
        self.num_enc_attention_heads = num_enc_attention_heads  
        self.patch_size = patch_size
        self.enc_layer_norm_eps = enc_layer_norm_eps
        self.enc_attention_dropout = enc_attention_dropout
        
        # Decoder
        self.dec_embed_dim = dec_embed_dim
        self.num_dec_hidden_layers = num_dec_hidden_layers

        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_recon = lambda_recon
        self.epochs = epochs