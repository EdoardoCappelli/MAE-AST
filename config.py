class Config:

    def __init__(
            self,

            # Spectrogram
            n_mel_bins = 128, # numero di bins del mel spectrogram (asse y)
            num_channels = 1,
            patch_embedding_dropout = 0.1,
            img_size = (128, 1024),
            sample_rate = 16000,
            hop_length = 160,
            n_fft = 1024,

            # Masking 
            masking_strategy = "patch", # tipo di mascheramento patch o frame
            random = False, # se False chunk masking
            masking_percentage = 0.75, # percentuale delle patches che vogliamo mascherare
            chunk_size = (4, 4),  # Dimensione del chunk di patch (es. 4x4 patches)
            chunk_len = 10,

            # Encoder 
            enc_embed_dim=768, # dimensione dell'embedding vector
            enc_hidden_layers = 6, # layers del vision tranformer
            enc_attention_heads = 12, # heads del multi head attention
            patch_size = (16,16), # ogni immagine verrà divisa in patch 16x16
            enc_layer_norm_eps = 1e-6, 
            enc_attention_dropout = 0.0,
            enc_mlp_ratio = 4, # mlp hidden dimension 3072

            # Decoder
            dec_hidden_layers = 2,
            dec_embed_dim = 768,
            dec_attention_heads = 12, # heads del multi head attention
            dec_layer_norm_eps = 1e-6, 
            dec_attention_dropout = 0.0,
            dec_mlp_ratio = 4,

            # Training
            use_validation = True,   
            validation_split_ratio = 0.2,
            learnable_pos_emb = False,
            use_cls = False,
            batch_size = 32,
            initial_lr = 1e-4,
            lambda_recon = 10,
            workers = 4,
            pin_memory = True,
            weight_decay = 0.01,
            sample_size = 4000,
            print_freq = 50,
            resume = None,   # Path to a specific checkpoint to resume training
            resume_epoch = None,  # Resume training from a specific epoch number  
            warmup_percentage = 0.1, 

            # Pretraining
            pretraining_dataset = "librispeech&audioset",
            pretraining_checkpoint_epochs = [1, 2, 4, 6, 8, 10, 12, 14, 16], 
            pretraining_checkpoints_dir = './checkpoints/pretraining_schedPoly_lr0.0001_warmup',

            # Finetuning
            n_classes_voxceleb = 1251,
            n_classes_esc = 50,
            pretrained_checkpoint_path = './checkpoints/pretraining_schedPoly_lr0.0001_warmup/librispeech&audioset/checkpoint_epoch_16.pth',
            voxceleb_meta_file = "./data/VoxCeleb/vox1_meta.csv",  
            esc_audio_root= "./data/ESC/audio",
            esc_meta_file= "./data/ESC/meta/esc50.csv",
            finetuning_checkpoint_epochs = [3, 5, 10, 20, 50, 100, 200, 500, 1000],
            finetuning_checkpoints_dir = './checkpoints/finetuning_schedPoly_lr0.0001_warmup',

            # Dataset 
            voxceleb_root = "./data/VoxCeleb",
            audioset_root = "./data/AudioSet",
            librispeech_root = "./data/LibriSpeech",
            librispeech_subset = 'train-clean-100',
            librispeech_percentage = 0.5,
            audioset_percentage = 1,  
            voxceleb_percentage = 0.5,
            esc_percentage = 1,

            # WandB
            use_wandb = False,
            finetuning_wandb_project = "mae-finetuning",
            pretraining_wandb_project = "mae-pretraining",
            wandb_entity = "name_entity",  
            save_artifacts = False, 
            **kwargs
    ):
        super().__init__()

        # Spectrogram
        self.num_channels = num_channels
        self.n_mel_bins = n_mel_bins
        self.patch_embedding_dropout = patch_embedding_dropout
        self.img_size = img_size
        self.sample_rate = sample_rate
        self.n_fft = n_fft  
        self.hop_length = hop_length

        # Masking 
        self.masking_percentage = masking_percentage
        self.chunk_size = chunk_size
        self.chunk_len = chunk_len 

        # Encoder 
        self.enc_embed_dim = enc_embed_dim
        self.enc_hidden_layers = enc_hidden_layers
        self.enc_attention_heads = enc_attention_heads  
        self.patch_size = patch_size
        self.enc_layer_norm_eps = enc_layer_norm_eps
        self.enc_attention_dropout = enc_attention_dropout
        self.enc_mlp_ratio = enc_mlp_ratio
        
        # Decoder
        self.dec_embed_dim = dec_embed_dim
        self.dec_hidden_layers = dec_hidden_layers
        self.dec_hidden_layers= dec_hidden_layers
        self.dec_attention_heads = dec_attention_heads
        self.dec_layer_norm_eps = dec_layer_norm_eps
        self.dec_attention_dropout = dec_attention_dropout
        self.dec_mlp_ratio = dec_mlp_ratio

        # Training
        self.use_validation = use_validation   
        self.validation_split_ratio = validation_split_ratio
        self.learnable_pos_emb = learnable_pos_emb
        self.use_cls = use_cls
        self.print_freq = print_freq
        self.workers = workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.lambda_recon = lambda_recon
        self.weight_decay = weight_decay
        self.sample_size = sample_size
        self.resume = resume
        self.resume_epoch = resume_epoch  # Specific epoch to resume training from
        self.warmup_percentage = warmup_percentage
         
        # Pretraining
        self.pretraining_checkpoint_epochs = pretraining_checkpoint_epochs
        self.pretraining_dataset = pretraining_dataset
        self.pretraining_checkpoints_dir = pretraining_checkpoints_dir

        # Finetuning
        self.n_classes_voxceleb = n_classes_voxceleb
        self.n_classes_esc = n_classes_esc
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.voxceleb_meta_file = voxceleb_meta_file
        self.esc_audio_root = esc_audio_root
        self.esc_meta_file = esc_meta_file
        self.finetuning_checkpoint_epochs = finetuning_checkpoint_epochs  
        self.finetuning_checkpoints_dir = finetuning_checkpoints_dir

        # Dataset
        self.librispeech_root = librispeech_root
        self.voxceleb_root = voxceleb_root
        self.audioset_root = audioset_root
        self.librispeech_subset = librispeech_subset
        self.librispeech_percentage = librispeech_percentage
        self.voxceleb_percentage = voxceleb_percentage
        self.esc_percentage = esc_percentage
        self.audioset_percentage = audioset_percentage  


        # WandB
        self.use_wandb = use_wandb
        self.finetuning_wandb_project = finetuning_wandb_project
        self.pretraining_wandb_project = pretraining_wandb_project
        self.wandb_entity = wandb_entity    
        self.save_artifacts = save_artifacts
