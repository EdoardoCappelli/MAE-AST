import time
import torch
from librispeech import LibriSpeech, collate_fn_spectrogram, collate_fn_crop, AudioToSpectrogram
from mae import MAE
from torch.utils.data import Subset
import random
from losses import infoNCE_loss, mae_loss
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import os 
import wandb
from config import Config 
import glob
import argparse

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
        collate_fn=lambda batch: collate_fn_crop(batch, max_length=80000*2)  # 5*2 secondi
    )
    
    return train_loader_crop

# --------------------------------------------------
# CHECKPOINT UTILITIES
# --------------------------------------------------

def find_checkpoint_by_epoch(checkpoint_dir, epoch):
    """Trova un checkpoint specifico per epoca"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, device='cuda'):
    """Carica un checkpoint e restituisce le informazioni necessarie per riprendere il training"""
    if not os.path.exists(checkpoint_path):
        print(f"{Fore.RED}Checkpoint non trovato: {checkpoint_path}{Style.RESET_ALL}")
        return None
    
    print(f"{Fore.CYAN}Caricamento checkpoint: {checkpoint_path}{Style.RESET_ALL}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Carica lo stato del modello
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Carica lo stato dell'ottimizzatore
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Carica lo stato dello scheduler se presente
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Estrai le informazioni del checkpoint
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        total_iterations = checkpoint.get('total_iterations', 0)
        
        print(f"{Fore.GREEN}Checkpoint caricato con successo!{Style.RESET_ALL}")
        print(f"  Epoca: {start_epoch}")
        print(f"  Best Loss: {best_loss:.6f}")
        print(f"  Total Iterations: {total_iterations}")
        
        return {
            'start_epoch': start_epoch,
            'best_loss': best_loss,
            'total_iterations': total_iterations
        }
        
    except Exception as e:
        print(f"{Fore.RED}Errore nel caricamento del checkpoint: {str(e)}{Style.RESET_ALL}")
        return None

def print_resume_info(resume_from_epoch, checkpoint_dir):
    """Stampa informazioni sul resume"""
    print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}RESUME TRAINING{Style.RESET_ALL}")
    print(f"Riprendendo il training dall'epoca: {Fore.GREEN}{resume_from_epoch}{Style.RESET_ALL}")
    print(f"Directory checkpoint: {checkpoint_dir}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")

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

def print_header(dataset_name="LibriSpeech", use_validation=True, device='cpu', resume_info=None):
    print("\n" + "="*80)
    header_text = f'MAE Training on {dataset_name} using {device}'
    print(f"{Fore.CYAN}{header_text:^80}{Style.RESET_ALL}")
    if use_validation:
        print(f"{Fore.GREEN}{'Mode: Training + Validation':^80}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}{'Mode: Training Only':^80}{Style.RESET_ALL}")
    
    if resume_info:
        print(f"{Fore.MAGENTA}{'RESUMING from Epoch ' + str(resume_info['start_epoch']):^80}{Style.RESET_ALL}")
    
    print("="*80)

def print_epoch_header(epoch, total_epochs, lr):
    """Stampa l'header dell'epoca"""
    print(f"\n{Fore.YELLOW}┌─ Epoch {epoch+1:3d}/{total_epochs} ")
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} Learning Rate: {Fore.GREEN}{lr:.2e}{Style.RESET_ALL}")

def train(train_loader, model, optimizer, epoch, total_epochs, print_freq=100, device='cuda', pretraining=True, use_wandb=False):
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
            recon_loss = mae_loss(target, recon_logits) * 10
            class_loss = infoNCE_loss(target, class_logits) * 1
            loss = recon_loss + class_loss 
            
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
            
            step = epoch * total_batches + i
            if use_wandb:
                wandb.log({
                    "train/step_loss": losses.val,
                    "train/step_recon_loss": recon_losses.val,
                    "train/step_class_loss": class_losses.val,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": step,
                    "epoch": epoch + 1
                })

    # Final epoch summary
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} {'─'*70}")
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} "
          f" Epoch {epoch+1:3d} Complete │ "
          f"Avg Loss: {Fore.GREEN}{losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Recon: {Fore.BLUE}{recon_losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Class: {Fore.MAGENTA}{class_losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Time: {format_time(batch_time.sum)}")
    print(f"{Fore.YELLOW}└─{Style.RESET_ALL}")
    
    return losses.avg, recon_losses.avg, class_losses.avg

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
                recon_loss = mae_loss(target, recon_logits) * 10
                class_loss = infoNCE_loss(target, class_logits) * 1
                loss = recon_loss + class_loss 
                
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
    
    return losses.avg, recon_losses.avg, class_losses.avg

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False, save_to_wandb=False, use_wandb=False):
    """Salva checkpoint con stampa professionale e opzione per wandb artifact"""
    torch.save(state, filename)
    print(f" {Fore.GREEN}Checkpoint saved:{Style.RESET_ALL} {filename}")
    if is_best:
        best_filename = filename.replace('.pth.tar', '_best.pth.tar')
        torch.save(state, best_filename)
        print(f" {Fore.YELLOW}Best model saved:{Style.RESET_ALL} {best_filename}")
        # Salva il modello migliore come artefatto
        if save_to_wandb and use_wandb:
            artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
            artifact.add_file(best_filename, name='model_best.pth.tar')
            wandb.log_artifact(artifact)
            print(f" {Fore.BLUE}Best model saved as wandb artifact.{Style.RESET_ALL}")

def print_training_summary(epoch, total_epochs, total_iterations, train_loss, val_loss=None, best_loss=None, lr=None, elapsed_time=None, use_validation=True):
    """Stampa un summary completo dell'epoca"""
    print(f"\n{Fore.CYAN} Training Summary - Epoch {epoch+1}/{total_epochs}{Style.RESET_ALL}")
    print(f" Total iterations:     {total_iterations}")
    print(f" Train Loss:     {train_loss:.6f}")
    
    if use_validation and val_loss is not None:
        print(f" Val Loss:       {val_loss:.6f}") 
        if best_loss is not None:
            print(f" Best Loss:        {best_loss:.6f}")
    
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

def plot_training_graph(iterations, train_losses):
    save_path = "./"
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.xlabel("Pretraining Iteration")
    plt.ylabel("Normalized Performance")
    plt.title("Training Loss vs. Iterations")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n{Fore.GREEN}Grafico salvato in:{Style.RESET_ALL} {save_path}")
    plt.show()
    plt.close() 

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():   
    import time
    import torch
    import torch.optim as optim
    from types import SimpleNamespace
    import os
    
    config = Config()  
    
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # Istanzio il modello MAE
    model = MAE(config)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(),
                           lr=config.initial_lr,
                           weight_decay=config.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)

    # Variabili per il resume
    start_epoch = 0
    best_loss = float('inf')
    total_iterations_offset = 0
    resume_info = None
    
    # Logica di resume
    checkpoint_to_load = None
    
    if config.resume:
        # Resume da un checkpoint specifico
        checkpoint_to_load = config.resume
    elif config.resume_epoch is not None:
        # Resume da un'epoca specifica
        checkpoint_to_load = find_checkpoint_by_epoch(config.checkpoints_dir, config.resume_epoch)
        if checkpoint_to_load is None:
            print(f"{Fore.RED}Checkpoint per l'epoca {config.resume_epoch} non trovato!{Style.RESET_ALL}")
            return
    
    # Carica il checkpoint se specificato
    if checkpoint_to_load:
        resume_info = load_checkpoint(checkpoint_to_load, model, optimizer, scheduler, device)
        if resume_info:
            start_epoch = resume_info['start_epoch']
            best_loss = resume_info['best_loss']
            total_iterations_offset = resume_info['total_iterations']
            print_resume_info(start_epoch, config.checkpoints_dir)
        else:
            print(f"{Fore.RED}Impossibile caricare il checkpoint, iniziando da zero{Style.RESET_ALL}")

    # Inizializza wandb solo se abilitato
    
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config),
            resume="allow" if resume_info else False,
            id=wandb.util.generate_id() if not resume_info else None
        )
        
        run_name = f'{config.dataset_name}-epochs_{config.epochs}-lr_{config.initial_lr}'
        if resume_info:
            run_name += f'-resumed_from_{start_epoch}'
        run_name += f'-{wandb.run.id}'
        
        wandb.run.name = run_name
        # wandb.run.save()

        wandb.watch(model, log='gradients', log_freq=config.print_freq * 5)
    else:
        print(f"{Fore.YELLOW}WandB logging disabilitato{Style.RESET_ALL}")

    # DataLoader 
    if config.dataset_name == "LibriSpeech":
        spectrogram_transform = AudioToSpectrogram(
            n_mels=128,
            sample_rate=16000,
        )
        
        train_dataset_spec = LibriSpeech(
            root=config.librispeech_root,
            train=True,
            download=config.download_librispeech,  # Già scaricato
            subset=config.librispeech_subset,
            transform=spectrogram_transform
        )
        
        # Test DataLoader con spettrogrammi
        train_loader = torch.utils.data.DataLoader(
            train_dataset_spec,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1024)  
        )

        # Validation loader 
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
                collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1024)
            )
        else:
            val_loader = None
            
    else:
        # Usa MNIST o altro dataset
        pass
    
    print_header(config.dataset_name, config.use_validation, device, resume_info)

    total_epochs = config.epochs
    
    checkpoint_epochs = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    iterations_per_epoch = len(train_loader)

    all_iterations = []
    all_train_losses = []

    # Training loop modificato per supportare il resume
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        
        lr = optimizer.param_groups[0]['lr']
        print_epoch_header(epoch, total_epochs, lr)

        # Training
        train_loss, train_recon_loss, train_class_loss = train(train_loader, model, optimizer, epoch, total_epochs, config.print_freq, device, pretraining=True, use_wandb=config.use_wandb)
        
        total_iterations = total_iterations_offset + (epoch - start_epoch + 1) * iterations_per_epoch
        all_iterations.append(total_iterations)
        all_train_losses.append(train_loss)

        # Validation (solo se richiesta)
        val_loss, val_recon_loss, val_class_loss = None, None, None
        if config.use_validation and val_loader is not None:
            val_loss, val_recon_loss, val_class_loss = validate(val_loader, model, epoch, config.print_freq, device, 
                                    pretraining=True)
            
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
        else:
            # Se non usi validation, il "best" è basato su train loss
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)
        
        epoch_log_dict = {
            'epoch/train_loss': train_loss,
            'epoch/train_recon_loss': train_recon_loss,
            'epoch/train_class_loss': train_class_loss,
            'epoch/epoch': epoch + 1,
            'epoch/total_iterations': total_iterations,
        }
        if val_loss is not None:
            epoch_log_dict.update({
                'epoch/val_loss': val_loss,
                'epoch/val_recon_loss': val_recon_loss,
                'epoch/val_class_loss': val_class_loss,
            })
        
        if use_wandb:
            wandb.log(epoch_log_dict)

        if (epoch + 1) in checkpoint_epochs or (epoch + 1) % 5 == 0:  # Salva anche ogni 5 epoche
            # Definisci lo stato da salvare (includendo anche lo scheduler)
            state = {
                'epoch': epoch + 1,
                'total_iterations': total_iterations,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,  
                'best_loss': best_loss,
                'config': vars(config)  # Salva anche la configurazione
            }
            filename = f'{config.checkpoints_dir}/checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(state, filename=filename, is_best=is_best, save_to_wandb=False, use_wandb=use_wandb)

        # Training summary
        epoch_time = time.time() - epoch_start

        scheduler.step()
        
    if use_wandb:
        wandb.finish()
    else:
        print(f"\n{Fore.GREEN}Training completato!{Style.RESET_ALL}")

if __name__ == '__main__':
    main()