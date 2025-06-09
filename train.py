import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from audioset import LibriSpeech, collate_fn_spectrogram, collate_fn_crop, AudioToSpectrogram
from mae import MAE
from torch.utils.data import DataLoader, Subset
import random
from types import SimpleNamespace
import torch.nn.functional as F 
from losses import infoNCE_loss, mae_loss
from colorama import Fore, Style, init

DATASET_NAME = "AudioSet"

# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------
def data_loader_librispeech(root, batch_size=32, workers=4, pin_memory=True, sample_size=None):
    
    train_dataset = LibriSpeech(
        root=root,
        train=True,
        download=True,
        subset='clean-100'  # Usa train-clean-100
    )

    if sample_size is not None:
        train_indices = random.sample(range(len(train_dataset)), min(sample_size, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(sample_size, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader_crop = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn_crop(batch, max_length=80000*3)  # 5*3 secondi
    )
    
    return train_loader_crop

# --------------------------------------------------
# AVERAGE METER (per tempi e loss)
# --------------------------------------------------

# Inizializza colorama  
from colorama import init, Fore, Style
init(autoreset=True)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

def print_header(dataset_name="LibriSpeech", use_validation=True):
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{'MAE Training on ' + dataset_name:^80}{Style.RESET_ALL}")
    if use_validation:
        print(f"{Fore.GREEN}{'Mode: Training + Validation':^80}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}{'Mode: Training Only':^80}{Style.RESET_ALL}")
    print("="*80)

def print_epoch_header(epoch, total_epochs, lr):
    """Stampa l'header dell'epoca"""
    print(f"\n{Fore.YELLOW}┌─ Epoch {epoch+1:3d}/{total_epochs} ")
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} Learning Rate: {Fore.GREEN}{lr:.2e}{Style.RESET_ALL}")

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Decade il learning rate di 0.8 ogni 15 epoche."""
    lr = init_lr * (0.8 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_loader, model, optimizer, epoch, total_epochs, print_freq=100, device='cuda', pretraining=True):
    """Loop di training professionale per un epoch"""
    batch_time = AverageMeter() # batch_time - data_time alto → Riduci model size, aumenta batch size
    data_time = AverageMeter() # data_time alto → Aumenta num_workers, usa SSD, ottimizza dataset
    losses = AverageMeter()
    recon_losses = AverageMeter()
    class_losses = AverageMeter()

    model.train()
    end = time.time()
    
    # Calcola ETA
    total_batches = len(train_loader)
    
    for i, (input, _) in enumerate(train_loader):
        # Misura data loading time
        data_time.update(time.time() - end)

        input = input.to(device, non_blocking=True)

        # Forward pass
        if pretraining:
            target, recon_logits, class_logits, bool_mask = model(input)
            recon_loss = mae_loss(target, recon_logits)
            class_loss = infoNCE_loss(target, class_logits)
            loss = recon_loss * 1 + class_loss * 1
            
            # Update meters
            recon_losses.update(recon_loss.item(), input.size(0))
            class_losses.update(class_loss.item(), input.size(0))

        # Backward + optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Misura batch time
        batch_time.update(time.time() - end)
        end = time.time()

        # Update loss meter
        losses.update(loss.item(), input.size(0))

        # Print progress
        if i % print_freq == 0:
            # Calcola ETA
            remaining_batches = total_batches - i
            eta_seconds = remaining_batches * batch_time.avg
            
            progress = (i + 1) / total_batches * 100
            bar_length = 30
            filled_length = int(bar_length * (i + 1) // total_batches)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"{Fore.YELLOW}│{Style.RESET_ALL} "
                  f"[{bar}] {progress:5.1f}% "
                  f"│ Batch {i:4d}/{total_batches} "
                  f"│ Loss: {Fore.RED}{losses.val:.4f}{Style.RESET_ALL}({losses.avg:.4f}) "
                  f"│ Recon: {Fore.BLUE}{recon_losses.val:.4f}{Style.RESET_ALL} "
                  f"│ Class: {Fore.MAGENTA}{class_losses.val:.4f}{Style.RESET_ALL} "
                  f"│ Time: {batch_time.val:.3f}s "
                  f"│ ETA: {format_time(eta_seconds)}")

    # Final epoch summary
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} {'─'*70}")
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} "
          f" Epoch {epoch+1:3d} Complete │ "
          f"Avg Loss: {Fore.GREEN}{losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Recon: {Fore.BLUE}{recon_losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Class: {Fore.MAGENTA}{class_losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Time: {format_time(batch_time.sum)}")
    print(f"{Fore.YELLOW}└─{Style.RESET_ALL}")
    
    return losses.avg

def validate(val_loader, model, epoch, print_freq=100, device='cuda', pretraining=True):
    """Loop di validazione professionale"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    class_losses = AverageMeter()

    model.eval()
    
    print(f"\n{Fore.CYAN} Validation Phase{Style.RESET_ALL}")
    
    end = time.time()
    
    with torch.no_grad():
        for i, (input, _) in enumerate(val_loader):
            input = input.to(device, non_blocking=True)

            # Forward pass
            if pretraining:
                target, recon_logits, class_logits, bool_mask = model(input)
                recon_loss = mae_loss(target, recon_logits)
                class_loss = infoNCE_loss(target, class_logits)
                loss = recon_loss * 1 + class_loss * 1
                
                # Update meters
                recon_losses.update(recon_loss.item(), input.size(0))
                class_losses.update(class_loss.item(), input.size(0))
            
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress = (i + 1) / len(val_loader) * 100
                print(f"  [{progress:5.1f}%] Batch {i:3d}/{len(val_loader)} │ "
                      f"Loss: {losses.val:.4f}({losses.avg:.4f}) │ "
                      f"Time: {batch_time.val:.3f}s")

    # Validation summary
    print("\n" + "─" * 50)
    print(f" {Fore.GREEN}Validation Results{Style.RESET_ALL}")
    print(f"    Total Loss:    {Fore.CYAN}{losses.avg:.6f}{Style.RESET_ALL}")
    print(f"    Recon Loss:    {Fore.BLUE}{recon_losses.avg:.6f}{Style.RESET_ALL}")
    print(f"    Class Loss:    {Fore.MAGENTA}{class_losses.avg:.6f}{Style.RESET_ALL}")
    print(f"    Avg Time:     {batch_time.avg:.3f}s/batch")
    print("─" * 50)
    
    return losses.avg

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    """Salva checkpoint con stampa professionale"""
    torch.save(state, filename)
    print(f" {Fore.GREEN}Checkpoint saved:{Style.RESET_ALL} {filename}")
    if is_best:
        best_filename = filename.replace('.pth.tar', '_best.pth.tar')
        torch.save(state, best_filename)
        print(f" {Fore.YELLOW}Best model saved:{Style.RESET_ALL} {best_filename}")

def print_training_summary(epoch, total_epochs, train_loss, val_loss=None, best_loss=None, lr=None, elapsed_time=None, use_validation=True):
    """Stampa un summary completo dell'epoca"""
    print(f"\n{Fore.CYAN} Training Summary - Epoch {epoch+1}/{total_epochs}{Style.RESET_ALL}")
    print(f" Train Loss:     {train_loss:.6f}")
    
    if use_validation and val_loss is not None:
        print(f" Val Loss:       {val_loss:.6f}") 
        if best_loss is not None:
            print(f" Best Loss:      {best_loss:.6f}")
    
    if lr is not None:
        print(f" Learning Rate:  {lr:.2e}")
    if elapsed_time is not None:
        print(f" Epoch Time:     {format_time(elapsed_time)}")
    
    # Progress bar per le epoche
    progress = (epoch + 1) / total_epochs
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\n Total Progress: [{bar}] {progress*100:.1f}%")


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():
    import time
    import torch
    import torch.optim as optim
    from types import SimpleNamespace
    import os
    
    # Configurazione base
    config = SimpleNamespace(
        # DATASET SETTINGS
        dataset_name = "LibriSpeech", 
        use_validation = True,      

        # LibriSpeech
        audio_length_seconds = 5,  
        librispeech_root = "C:/Users/admin/Desktop/VS Code/data/",
        librispeech_subset = 'clean-100',  # 'clean-100', 'clean-360', 'other-500'
        
        # MNIST  
        mnist_root = "./data_mnist",
        img_size = (128, 1000),
        patch_size = (16, 16),
        num_channels = 1,
        
        # Parametri Encoder
        enc_embed_dim = 768,
        enc_mlp_layer_dim = 768,
        enc_hidden_layers = 8,
        enc_attention_heads = 8,
        enc_layer_norm_eps = 1e-6,
        enc_attention_dropout = 0.1,
        enc_mlp_ratio = 4,
        
        # Parametri Decoder
        dec_hidden_layers = 6,
        dec_embed_dim = 768,
        dec_attention_heads = 8,
        dec_layer_norm_eps = 1e-6,
        dec_attention_dropout = 0.1,
        dec_mlp_ratio = 4,
        
        # Masking
        masking_strategy = "random",
        masking_percentage = 0.75,
        
        # Training
        batch_size = 32,
        initial_lr = 1e-4,
        weight_decay = 0.01,
        epochs = 100,
        print_freq = 50,
        workers = 0,
        pin_memory = False,

        # Checkpoint
        checkpoints_dir = "./checkpoints_mae"
    )

    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # Istanzio il modello MAE
    model = MAE(config)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(),
                           lr=config.initial_lr,
                           weight_decay=config.weight_decay)

    # DataLoader - scegli il tuo dataset
    if config.dataset_name == "LibriSpeech":
        spectrogram_transform = AudioToSpectrogram(
        n_mels=128,
        sample_rate=16000,
        hop_length=160,  # ~10ms hop
        n_fft=512,       # ~32ms window (più frequenze)
        to_db=True,
        normalize=True
        )
        
        train_dataset_spec = LibriSpeech(
            root=config.librispeech_root,
            train=True,
            download=False,  # Già scaricato
            subset='clean-100',
            transform=spectrogram_transform
        )
        
        # Test DataLoader con spettrogrammi
        train_loader_spec = torch.utils.data.DataLoader(
            train_dataset_spec,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1000)  # 1008 = 63*16 patch perfette
        )
        # Validation loader (solo se richiesto)
        if config.use_validation:
            val_dataset = LibriSpeech(
                root=config.librispeech_root,
                train=False,
                download=True,
                subset='clean'  # usa test-clean per validation
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.workers,
                pin_memory=True,
                collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1000)  # 1008 = 63*16 patch perfette
                # collate_fn=lambda batch: collate_fn_crop(batch, max_length=80000*3)
            )
        else:
            val_loader = None
            
    else:
        # Usa MNIST o altro dataset
        # train_loader, val_loader = data_loader_mnist(...)
        pass
    
    print_header(config.dataset_name, config.use_validation)

    best_loss = float('inf')
    total_epochs = config.epochs
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, config.initial_lr)
        print_epoch_header(epoch, total_epochs, lr)

        # Training
        train_loss = train(train_loader_spec, model, optimizer, epoch, total_epochs, 
                          config.print_freq, device, pretraining=True)
        
        # Validation (solo se richiesta)
        val_loss = None
        if config.use_validation and val_loader is not None:
            val_loss = validate(val_loader, model, epoch, config.print_freq, device, 
                               pretraining=True)
            
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
        else:
            # Se non usi validation, il "best" è basato su train loss
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, filename=f'{config.checkpoints_dir}/checkpoint_epoch_{epoch+1}.pth', is_best=is_best)
        
        # Training summary
        epoch_time = time.time() - epoch_start
        print_training_summary(epoch, total_epochs, train_loss, val_loss, 
                             best_loss, lr, epoch_time, config.use_validation)

if __name__ == '__main__':
    main()