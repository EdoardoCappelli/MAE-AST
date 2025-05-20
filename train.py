import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformer import VisionTransformer
from config import Config
import numpy as np
from typing import Optional, Tuple, Dict, List
import random 
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from pathlib import Path
import random

class BalancedTrainDataset(Dataset):
    def __init__(self, data_dir, sample_size = None):
        
        self.file_paths = []
        
        if isinstance(data_dir, (list, tuple)):
            for dir_path in data_dir:
                self._load_files(Path(dir_path))
        else:
            self._load_files(Path(data_dir))
        
        if sample_size is not None and sample_size < len(self.file_paths):
            self.file_paths = random.sample(self.file_paths, sample_size)
        
        print(f"Dataset creato con {len(self.file_paths)} spettrogrammi")
        
    def _load_files(self, directory):
        if directory.is_dir():
            files = list(directory.glob(f"**/*.pt"))
            self.file_paths.extend(files)
            print(f"Caricati {len(files)} file da {directory}")
        else:
            print(f"Attenzione: {directory} non è una directory valida")
    
 
    def _load_spectrogram(self, file_path):
        """Carica uno spettrogramma da file."""
        try:
            if str(file_path).endswith('.pt'):
                return torch.load(file_path, map_location='cpu')
            else:
                raise ValueError(f"Formato file non supportato: {file_path}")
        except Exception as e:
            print(f"Errore nel caricamento di {file_path}: {e}")
            return torch.zeros((1, 1, 128, 128))  
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Restituisce uno spettrogramma. 
        """
        file_path = self.file_paths[idx]
        spectrogram = self._load_spectrogram(file_path)
        return spectrogram


def infoNCE(class_logits, targets):
    all_dots = torch.matmul(class_logits, targets.transpose(-1, -2))
    log_softmax = F.log_softmax(all_dots, dim=-1)
    loss_info_nce = -torch.mean(
        torch.diagonal(log_softmax, dim1=-2, dim2=-1)
    )

    return loss_info_nce

def train_one_epoch(model: VisionTransformer,
                  dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  config: Config,
                  device: torch.device):
    model.train()
    total_loss_epoch = 0.0
    total_recon_loss_epoch = 0.0
    total_NCE_loss_epoch = 0.0

    for batch_idx, data_in_batch in enumerate(dataloader):
        # Se data_in_batch è una tupla o lista, prendi il primo elemento (lo spectrogramma)
        if isinstance(data_in_batch, (list, tuple)):
            spectrogram_batch_tensor = data_in_batch[0].to(device)
        else:
            spectrogram_batch_tensor = data_in_batch.to(device)

        optimizer.zero_grad()

        outputs = model(spectrogram_batch_tensor)

        targets = outputs["target_patches"]
        
        recon_logits = outputs["recon_logits"]
        loss_recon = F.mse_loss(recon_logits, targets)
        
        class_logits = outputs["class_logits"]
        loss_info_nce = infoNCE(class_logits, targets)
        

        combined_loss = (config.lambda_recon * loss_recon) + loss_info_nce

        combined_loss.backward()
        optimizer.step()

        total_loss_epoch += combined_loss.item()
        total_recon_loss_epoch += loss_recon.item()
        total_NCE_loss_epoch += loss_info_nce.item()

        # if batch_idx % 5 == 0:
        #     print(
        #         f"  Batch {batch_idx}/{len(dataloader)}:\n "
        #         f"\tLoss Tot:\t{combined_loss.item():.4f}\n"
        #         f"\tLoss Recon:\t{loss_recon.item():.4f} (x{config.lambda_recon:.2f})\n"
        #         f"\tInfoNCE Loss:\t{loss_info_nce.item():.4f} (x{1:.2f})\n\n"
        #     )

    # Calcolo delle medie sull’intero epoch
    avg_loss = total_loss_epoch / len(dataloader)
    avg_recon_loss = total_recon_loss_epoch / len(dataloader)
    avg_NCE_loss = total_NCE_loss_epoch / len(dataloader)
    
    curr_lr = optimizer.param_groups[0]['lr']

    print(
        f"  Fine Epoch:\n"
        f"\tAvg Loss Tot:\t{avg_loss:.4f}\n"
        f"\tAvg Recon:\t{avg_recon_loss:.4f}\n"
        f"\tAvg InfoNCE:\t{avg_NCE_loss:.4f}\n"
        f"\tLearning Rate:\t{curr_lr:.6f}\n\n"
    )

    return avg_loss

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Training del MAE"
    )
    parser.add_argument(
        'dataset_dir',
        nargs='?',
        default="/content/drive/MyDrive/Università/DeepLearning/mae_audio/datasets/tensors/balanced_train_segments",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionTransformer(config).to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.initial_lr, 
        weight_decay=0.01
    )

    def polynomial_decay(epoch):
        # Calculate the current decay factor
        max_epochs = config.epochs
        power = 2.0  # Use power=2.0 for polynomial decay
        decay_factor = (1 - epoch / max_epochs) ** power
        return decay_factor
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)

    dataset = BalancedTrainDataset(dataset_dir)
    
    # Creazione del DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,  # Shuffle per training self-supervised
        num_workers=4, 
        pin_memory=True
    )

    # dummy_data = []
    # for _ in range(200): # 20 campioni fittizi
    #     # Calcola altezza e larghezza valide per essere divisibili per patch_size
    #     img_width = 896
    #     img_height = 128
    #     valid_h = (img_height // config.patch_size[0]) * config.patch_size[0]
    #     valid_w = (img_width // config.patch_size[1]) * config.patch_size[1]
    #     if valid_h != img_height or valid_w != img_width:
    #         print(f"Attenzione: img_height/width adattate a {valid_h}x{valid_w} per divisibilità patch.")
    #     dummy_spectrogram = torch.randn(1, valid_h, valid_w)
    #     dummy_data.append(dummy_spectrogram)
    
    # dummy_dataset = torch.utils.data.TensorDataset(torch.stack(dummy_data))
    # dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config.batch_size)

    print("Inizio Training Fittizio...")
    for epoch in range(1, config.epochs + 1):
        print(f"--- Epoch {epoch} ---")
        avg_epoch_loss = train_one_epoch(
            model, 
            dataloader, 
            optimizer, 
            config, 
            device
        )
        scheduler.step()

    print(f"Fine training con una avg loss di {avg_epoch_loss}.")


if __name__ == "__main__":
    main()

    


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time
# import random
# from datetime import datetime

# from config import Config
# from transformer import VisionTransformer

# # Set random seeds for reproducibility
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# class DummySpectrogramDataset(Dataset):
#     """
#     A dummy dataset that generates random spectrogram data to simulate audio spectrograms
#     """
#     def __init__(self, num_samples=1000, height=128, width=1000):
#         self.num_samples = num_samples
#         self.height = height  # Frequency bins
#         self.width = width    # Time steps
        
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # Generate a random spectrogram with some patterns (more realistic than pure random)
#         spec = np.zeros((self.height, self.width), dtype=np.float32)
        
#         # Create some "harmonic" patterns
#         num_harmonics = np.random.randint(2, 6)
#         for i in range(num_harmonics):
#             freq = np.random.randint(10, self.height // 2)
#             harmonic_width = np.random.randint(3, 10)
#             intensity = np.random.uniform(0.5, 1.0)
            
#             # Add the harmonic and blur it slightly
#             for j in range(-harmonic_width, harmonic_width + 1):
#                 if 0 <= freq + j < self.height:
#                     amplitude = intensity * np.exp(-0.5 * (j / (harmonic_width/2))**2)
#                     spec[freq + j, :] += amplitude * np.random.uniform(0.8, 1.0, self.width)
        
#         # Add some temporal structure
#         onset_points = np.random.randint(0, self.width, size=np.random.randint(3, 8))
#         for onset in onset_points:
#             duration = np.random.randint(5, 20)
#             if onset + duration > self.width:
#                 duration = self.width - onset
#             spec[:, onset:onset+duration] *= np.random.uniform(1.2, 2.0)
        
#         # Add some noise
#         spec += np.random.normal(0, 0.05, (self.height, self.width))
        
#         # Make sure values are positive (as real spectrograms would be)
#         spec = np.abs(spec)
        
#         # Convert to tensor with batch and channel dimension [C, H, W]
#         spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        
#         return spec_tensor

# def plot_learning_curve(train_losses, val_losses, save_path='learning_curve.png'):
#     """Plot the learning curves for training and validation losses"""
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Learning Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()

# def train_model(config, model, train_loader, val_loader, num_epochs, learning_rate, device, output_dir):
#     """Train the model and validate it"""
    
#     # Define loss function for reconstruction task
#     criterion_recon = nn.MSELoss()
    
#     # Define optimizer
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # Track training progress
#     train_losses = []
#     val_losses = []
#     best_val_loss = float('inf')
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     print(f"Starting training... Model: {type(model).__name__}")
#     print(f"Training on device: {device}")
    
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
        
#         # Training phase
#         model.train()
#         train_loss = 0.0
#         train_recon_loss = 0.0
        
#         train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
#         for batch_idx, spectrograms in enumerate(train_progress):
#             # Move data to device
#             spectrograms = spectrograms.to(device)
            
#             # Zero gradients
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(spectrograms)
            
#             # Calculate reconstruction loss
#             recon_loss = criterion_recon(outputs["recon_logits"], outputs["target_patches"])
            
#             # Total loss
#             loss = recon_loss
            
#             # Backward pass and optimize
#             loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             # Update metrics
#             train_loss += loss.item()
#             train_recon_loss += recon_loss.item()
            
#             # Update progress bar
#             train_progress.set_postfix({
#                 'loss': f"{loss.item():.4f}",
#                 'recon_loss': f"{recon_loss.item():.4f}"
#             })
        
#         # Average training losses
#         avg_train_loss = train_loss / len(train_loader)
#         avg_train_recon_loss = train_recon_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
        
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         val_recon_loss = 0.0
        
#         with torch.no_grad():
#             val_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
#             for batch_idx, spectrograms in enumerate(val_progress):
#                 # Move data to device
#                 spectrograms = spectrograms.to(device)
                
#                 # Forward pass
#                 outputs = model(spectrograms)
                
#                 # Calculate reconstruction loss
#                 recon_loss = criterion_recon(outputs["recon_logits"], outputs["target_patches"])
                
#                 # Total loss
#                 loss = recon_loss
                
#                 # Update metrics
#                 val_loss += loss.item()
#                 val_recon_loss += recon_loss.item()
                
#                 # Update progress bar
#                 val_progress.set_postfix({
#                     'loss': f"{loss.item():.4f}",
#                     'recon_loss': f"{recon_loss.item():.4f}"
#                 })
        
#         # Average validation losses
#         avg_val_loss = val_loss / len(val_loader)
#         avg_val_recon_loss = val_recon_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
        
#         # Update learning rate
#         scheduler.step()
#         current_lr = scheduler.get_last_lr()[0]
        
#         # Print epoch summary
#         print(f"Epoch {epoch+1}/{num_epochs} - "
#               f"Train Loss: {avg_train_loss:.4f}, "
#               f"Train Recon Loss: {avg_train_recon_loss:.4f}, "
#               f"Val Loss: {avg_val_loss:.4f}, "
#               f"Val Recon Loss: {avg_val_recon_loss:.4f}, "
#               f"LR: {current_lr:.6f}")
        
#         # Save best model
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_model_path = os.path.join(output_dir, 'best_model.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': best_val_loss,
#             }, best_model_path)
#             print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
#         # Save checkpoint every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_losses': train_losses,
#                 'val_losses': val_losses,
#             }, checkpoint_path)
    
#     # Plot and save learning curves
#     plot_learning_curve(train_losses, val_losses, os.path.join(output_dir, 'learning_curve.png'))
    
#     # Save final model
#     final_model_path = os.path.join(output_dir, 'final_model.pth')
#     torch.save({
#         'model_state_dict': model.state_dict(),
#     }, final_model_path)
    
#     return train_losses, val_losses

# def visualize_reconstructions(model, val_loader, device, output_dir, num_samples=5):
#     """Visualize original spectrograms alongside model reconstructions"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     model.eval()
#     with torch.no_grad():
#         # Get a batch of samples
#         spectrograms = next(iter(val_loader))
#         spectrograms = spectrograms.to(device)
        
#         # Forward pass
#         outputs = model(spectrograms)
        
#         # Prepare for visualization
#         recon_patches = outputs["recon_logits"]
#         target_patches = outputs["target_patches"]
        
#         # Create reconstructed spectrograms
#         for i in range(min(num_samples, spectrograms.size(0))):
#             # Extract data for the current sample
#             original_spec = spectrograms[i, 0].cpu().numpy()
            
#             # Get masked indices from model
#             masked_indices = model.patch_embeddings.masked_indices_list[i]
            
#             # Create a blank canvas for the reconstruction
#             patch_size = model.config.patch_size
            
#             # Visualize
#             plt.figure(figsize=(15, 6))
            
#             # Plot original spectrogram
#             plt.subplot(1, 2, 1)
#             plt.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
#             plt.colorbar(format='%+2.0f dB')
#             plt.title('Original Spectrogram')
#             plt.xlabel('Time Frames')
#             plt.ylabel('Frequency Bins')
            
#             # Plot reconstruction
#             plt.subplot(1, 2, 2)
#             plt.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
#             plt.colorbar(format='%+2.0f dB')
#             plt.title('Masked Regions Highlighted')
            
#             # Highlight masked regions
#             height, width = original_spec.shape
#             rows_per_patch = patch_size[0]
#             cols_per_patch = patch_size[1]
#             rows = height // rows_per_patch
#             cols = width // cols_per_patch
            
#             for idx in masked_indices:
#                 # Convert flat index to 2D coordinates
#                 row = idx // cols
#                 col = idx % cols
                
#                 # Calculate spectrogram coordinates
#                 y_start = row * rows_per_patch
#                 x_start = col * cols_per_patch
                
#                 # Add rectangle to highlight masked region
#                 rect = plt.Rectangle((x_start, y_start), cols_per_patch, rows_per_patch, 
#                                       linewidth=1, edgecolor='r', facecolor='none')
#                 plt.gca().add_patch(rect)
            
#             plt.tight_layout()
#             plt.savefig(os.path.join(output_dir, f'reconstruction_{i}.png'))
#             plt.close()

# def main():
#     # Configuration
#     config = Config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create timestamp for output directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = f"outputs/mae_training_{timestamp}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Hyperparameters
#     batch_size = 10
#     num_epochs = 20
#     learning_rate = 1e-4
#     val_split = 0.2
    
#     # Create dataset
#     print("Creating dataset...")
#     dataset = DummySpectrogramDataset(num_samples=1000)
    
#     # Split into train and validation
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(val_split * dataset_size))
    
#     # Shuffle indices
#     np.random.shuffle(indices)
#     train_indices, val_indices = indices[split:], indices[:split]
    
#     # Create samplers
#     train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
#     val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
#     # Create data loaders
#     train_loader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         sampler=train_sampler,
#         num_workers=2,
#         pin_memory=torch.cuda.is_available()
#     )
    
#     val_loader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         sampler=val_sampler,
#         num_workers=2,
#         pin_memory=torch.cuda.is_available()
#     )
    
#     print(f"Train set: {len(train_indices)} samples")
#     print(f"Validation set: {len(val_indices)} samples")
    
#     # Create model
#     print("Creating model...")
#     model = VisionTransformer(config)
#     model = model.to(device)
    
#     # Save config and training parameters
#     with open(os.path.join(output_dir, 'training_config.txt'), 'w') as f:
#         f.write(f"Batch size: {batch_size}\n")
#         f.write(f"Learning rate: {learning_rate}\n")
#         f.write(f"Number of epochs: {num_epochs}\n")
#         f.write(f"Model configuration:\n")
#         # Write model hyperparameters
#         for param, value in vars(config).items():
#             f.write(f"  {param}: {value}\n")
    
#     # Print model summary
#     print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
#     print(f"Model architecture: {model}")
    
#     # Train model
#     print("Starting training...")
#     train_losses, val_losses = train_model(
#         config=config,
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=num_epochs,
#         learning_rate=learning_rate,
#         device=device,
#         output_dir=output_dir
#     )
    
#     # Visualization
#     print("Generating visualizations...")
#     visualize_reconstructions(model, val_loader, device, os.path.join(output_dir, 'reconstructions'))
    
#     print(f"Training complete! All outputs saved to {output_dir}")

# if __name__ == "__main__":
#     main()