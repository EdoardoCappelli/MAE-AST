import time
import torch
import torch.optim as optim
from types import SimpleNamespace
import os
import random
from torch.utils.data import Subset, DataLoader
import torch.nn as nn 
from pathlib import Path 
from config import Config 
from mae import MAE 
from dataloader.dataloader_finetuning import AudioToSpectrogram, VoxCelebGenderDataset, ESCAudioDataset, collate_fn_spectrogram
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import wandb
import glob
import argparse
from transformers import get_polynomial_decay_schedule_with_warmup
from utils.plot import plot_normalized_loss

# Per colorama
init(autoreset=True)

# --------------------------------------------------
# UTILITIES CHECKPOINT
# --------------------------------------------------

def find_checkpoint_by_epoch(checkpoint_dir, epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, device='cuda'):
    if not os.path.exists(checkpoint_path):
        print(f"{Fore.RED}Checkpoint non trovato: {checkpoint_path}{Style.RESET_ALL}")
        return None
    
    print(f"{Fore.CYAN}Caricamento checkpoint: {checkpoint_path}{Style.RESET_ALL}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        total_iterations = checkpoint.get('total_iterations', 0)
        
        print(f"{Fore.GREEN}Checkpoint caricato con successo!{Style.RESET_ALL}")
        print(f"   Epoca: {start_epoch}")
        print(f"   Best Loss: {best_loss:.6f}")
        print(f"   Total Iterations: {total_iterations}")
        
        return {
            'start_epoch': start_epoch,
            'best_loss': best_loss,
            'total_iterations': total_iterations,
            'checkpoint_data': checkpoint
        }
        
    except Exception as e:
        print(f"{Fore.RED}Errore nel caricamento del checkpoint: {str(e)}{Style.RESET_ALL}")
        return None

def print_resume_info(resume_from_epoch, checkpoint_dir):
    print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}RIPRENDENDO IL TRAINING{Style.RESET_ALL}")
    print(f"Riprendendo il training dall'epoca: {Fore.GREEN}{resume_from_epoch}{Style.RESET_ALL}")
    print(f"Directory checkpoint: {checkpoint_dir}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")

# --------------------------------------------------
# AVERAGE METER (per tempi e loss)
# --------------------------------------------------

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

def print_header(dataset_name="Fine-tuning", use_validation=True, device='cpu', resume_info=None):
    print("\n" + "="*80)
    header_text = f'MAE Fine-tuning on {dataset_name} using {device}'
    print(f"{Fore.CYAN}{header_text:^80}{Style.RESET_ALL}")
    if use_validation:
        print(f"{Fore.GREEN}{'Modalità: Training + Validation':^80}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}{'Modalità: Solo Training':^80}{Style.RESET_ALL}")
    
    if resume_info:
        print(f"{Fore.MAGENTA}{'RIPRENDENDO dall\'Epoca ' + str(resume_info['start_epoch']):^80}{Style.RESET_ALL}")
    
    print("="*80)

def print_epoch_header(epoch, total_epochs, lr):
    """Stampa l'header dell'epoca"""
    print(f"\n{Fore.YELLOW}┌─ Epoca {epoch+1:3d}/{total_epochs} ")
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} Learning Rate: {Fore.GREEN}{lr:.2e}{Style.RESET_ALL}")

def train(train_loader, model, optimizer, criterion, epoch, total_epochs, print_freq=100, device='cuda', use_wandb=False, initial_step=0):
    batch_time = AverageMeter() 
    data_time = AverageMeter() 
    losses = AverageMeter()
    
    correct_predictions = 0
    total_samples = 0

    model.train()
    end = time.time()
    
    total_batches = len(train_loader)
    
    for i, batch_data in enumerate(train_loader): 
        data_time.update(time.time() - end)

        input_waveform = batch_data['waveform'].to(device, non_blocking=True)
        labels = batch_data['label'].to(device, non_blocking=True)

        logits = model(input_waveform) 
        loss = criterion(logits, labels)
        
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Backward + optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Misura batch time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), input_waveform.size(0))

        if i % print_freq == 0:
            remaining_batches = total_batches - i
            eta_seconds = remaining_batches * batch_time.avg
            
            progress = (i + 1) / total_batches * 100
            bar_length = 30
            filled_length = int(bar_length * (i + 1) // total_batches)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            current_train_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
            
            print(f"{Fore.YELLOW}│{Style.RESET_ALL} "
                  f"[{bar}] {progress:5.1f}% "
                  f"│ Batch {i:4d}/{total_batches} "
                  f"│ Loss: {Fore.RED}{losses.val:.4f}{Style.RESET_ALL}({losses.avg:.4f}) "
                  f"│ Acc: {Fore.GREEN}{current_train_accuracy:.2f}%{Style.RESET_ALL} " 
                  f"│ Time: {batch_time.val:.3f}s "
                  f"│ ETA: {format_time(eta_seconds)}")
            
            step = initial_step + i

            if use_wandb:
                wandb.log({
                    "train/step_loss": losses.val,
                    "train/avg_loss": losses.avg,
                    "train/step_accuracy": current_train_accuracy, 
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": step, 
                    "epoch": epoch + 1
                })

    # Riassunto epoca
    final_train_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} {'─'*70}")
    print(f"{Fore.YELLOW}│{Style.RESET_ALL} "
          f" Epoca {epoch+1:3d} Completata │ "
          f"Loss Media: {Fore.GREEN}{losses.avg:.6f}{Style.RESET_ALL} │ "
          f"Accuracy Training: {Fore.CYAN}{final_train_accuracy:.2f}%{Style.RESET_ALL} │ "  
          f"Tempo: {format_time(batch_time.sum)}")
    print(f"{Fore.YELLOW}└─{Style.RESET_ALL}")
    
    return losses.avg, final_train_accuracy, total_batches  

def validate(val_loader, model, criterion, epoch, print_freq=100, device='cuda', use_wandb=False, initial_step=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    correct_predictions = 0
    total_samples = 0

    model.eval()
    
    print(f"\n{Fore.CYAN} Fase di Validation{Style.RESET_ALL}")
    
    end = time.time()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader): 
            input_waveform = batch_data['waveform'].to(device, non_blocking=True)
            labels = batch_data['label'].to(device, non_blocking=True)

            logits = model(input_waveform)
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            losses.update(loss.item(), input_waveform.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress = (i + 1) / len(val_loader) * 100
                current_val_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
                print(f"   [{progress:5.1f}%] Batch {i:3d}/{len(val_loader)} │ "
                      f"Loss: {losses.val:.4f}({losses.avg:.4f}) │ "
                      f"Acc: {Fore.GREEN}{current_val_accuracy:.2f}%{Style.RESET_ALL} " 
                      f"│ Time: {batch_time.val:.3f}s")

            step = initial_step + i  
            if use_wandb:
                wandb.log({
                    "val/step_loss": losses.val,
                    "val/avg_loss": losses.avg,
                    "val/step_accuracy": current_val_accuracy,  
                    "step": step,
                    "epoch": epoch + 1
                })

    final_val_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print("\n" + "─" * 50)
    print(f" {Fore.GREEN}Risultati Validation{Style.RESET_ALL}")
    print(f"    Loss Totale:    {Fore.CYAN}{losses.avg:.6f}{Style.RESET_ALL}")
    print(f"    Accuracy:    {Fore.MAGENTA}{final_val_accuracy:.2f}%{Style.RESET_ALL}")  
    print(f"    Tempo Medio:    {batch_time.avg:.3f}s/batch")
    print("─" * 50)
    
    return losses.avg, final_val_accuracy 

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False, save_to_wandb=False, use_wandb=False):
    torch.save(state, filename)
    print(f" {Fore.GREEN}Checkpoint salvato:{Style.RESET_ALL} {filename}")
    if is_best:
        best_filename = filename.replace('.pth.tar', '_best.pth.tar')
        torch.save(state, best_filename)
        print(f" {Fore.YELLOW}Modello migliore salvato:{Style.RESET_ALL} {best_filename}")
        
        if save_to_wandb and use_wandb:
            artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
            artifact.add_file(best_filename, name='model_best.pth.tar')
            wandb.log_artifact(artifact)
            print(f" {Fore.BLUE}Modello migliore salvato come artefatto wandb.{Style.RESET_ALL}")

def print_training_summary(epoch, total_epochs, total_iterations, train_loss, train_accuracy, val_loss=None, val_accuracy=None, best_loss=None, lr=None, elapsed_time=None, use_validation=True):
    print(f"\n{Fore.CYAN} Riepilogo Training - Epoca {epoch+1}/{total_epochs}{Style.RESET_ALL}")
    print(f" Iterazioni totali:       {total_iterations}")
    print(f" Loss di Training:        {train_loss:.6f}")
    print(f" Accuracy Training:    {train_accuracy:.2f}%") 
    
    if use_validation and val_loss is not None:
        print(f" Loss di Validation:     {val_loss:.6f}") 
        print(f" Accuracy Validation: {val_accuracy:.2f}%") 
        if best_loss is not None:
            print(f" Migliore Loss:          {best_loss:.6f}")
    
    if lr is not None:
        print(f" Learning Rate:            {lr:.2e}")
    if elapsed_time is not None:
        print(f" Tempo Epoca:              {format_time(elapsed_time)}")
    
    progress = (epoch + 1) / total_epochs
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\n Progresso Totale: [{bar}] {progress*100:.1f}%")

# --------------------------------------------------
# FUNZIONE MAIN
# --------------------------------------------------

def main():   
    config = Config() 
    
    argparser = argparse.ArgumentParser(description="MAE Fine-tuning Script")
    argparser.add_argument('--dataset', type=str, choices=['esc', 'voxceleb'], help='Dataset to use for fine-tuning: "esc" or "voxceleb"')
    argparser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    argparser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for early stopping without improvement')
    argparser.add_argument('--initial_lr', type=float, default=0.0001, help='Initial learning rate for the optimizer')
    argparser.add_argument('--masking_strategy', type=str, choices=['patch', 'frame'], help='Masking strategy to use: "patch" or "frame"')
    argparser.add_argument('--audio_length_seconds', type=int, default=10, help='Length of audio segments in seconds for fine-tuning')
    argparser.add_argument('--pretrained_checkpoint_path', type=str, default=config.pretrained_checkpoint_path, help='Path to the pre-trained checkpoint to load for fine-tuning')
    args = argparser.parse_args()

    if args.dataset == "esc":
        args.audio_length_seconds = 5  
    elif args.dataset == "voxceleb":
        args.audio_length_seconds = 10
    else:
        raise ValueError("dataset deve essere 'esc' o 'voxceleb'")

    checkpoint_dir = config.finetuning_checkpoints_dir + '_' + args.masking_strategy + '/' +  args.dataset
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./plots", exist_ok=True) # Assicurati che la cartella plots esista

    model = MAE(args, config, pretraining=False) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=config.initial_lr,
                           weight_decay=config.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # Variabili per il resume
    start_epoch = 0
    best_loss = float('inf')
    best_accuracy = 0.0 
    total_iterations_offset = 0
    resume_info = None
    checkpoint_to_load = None
    
    # Logica di resume
    if config.resume:
        checkpoint_to_load = config.resume
    elif config.resume_epoch is not None:
        checkpoint_to_load = find_checkpoint_by_epoch(checkpoint_dir, config.resume_epoch)
        if checkpoint_to_load is None:
            print(f"{Fore.RED}Checkpoint per l'epoca {config.resume_epoch} non trovato! Iniziare da zero.{Style.RESET_ALL}")
    
    temp_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1, 
        num_training_steps=2, 
        lr_end=0.0,  
        power=2.0,   
        last_epoch=-1
    )

    if checkpoint_to_load:
        resume_info = load_checkpoint(checkpoint_to_load, model, optimizer, scheduler=temp_scheduler, device=device)
        if resume_info:
            start_epoch = resume_info['start_epoch']
            best_loss = resume_info['best_loss']
            total_iterations_offset = resume_info['total_iterations']
            best_accuracy = resume_info.get('best_accuracy', 0.0)
            print_resume_info(start_epoch, checkpoint_dir)
        else:
            print(f"{Fore.RED}Impossibile caricare il checkpoint di fine-tuning, iniziando da zero.{Style.RESET_ALL}")

    if args.pretrained_checkpoint_path and not resume_info:
        print(f"{Fore.CYAN}Caricamento del modello pre-addestrato da: {args.pretrained_checkpoint_path}{Style.RESET_ALL}")
        try:
            pretrained_checkpoint = torch.load(args.pretrained_checkpoint_path, map_location=device)
            # MAE in fine-tuning avrà un classifier head in più, quindi strict=False
            model.load_state_dict(pretrained_checkpoint['model_state_dict'], strict=False) 
            print(f"{Fore.GREEN}Modello pre-addestrato caricato con successo.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Errore nel caricamento del checkpoint pre-addestrato: {str(e)}{Style.RESET_ALL}")

    if config.use_wandb:
        wandb.init(
            project=config.finetuning_wandb_project+ f"-{args.masking_strategy}",
            entity=config.wandb_entity,
            config=vars(config),
            resume="allow" if resume_info else False, 
            id=wandb.util.generate_id() if not resume_info else None
        )
        
        run_name = f'finetune-{args.dataset}-epochs_{args.epochs}-lr_{config.initial_lr}'
        if resume_info:
            run_name += f'-resumed_from_{start_epoch}'
        run_name += f'-{wandb.run.id}'
        
        wandb.run.name = run_name
        wandb.watch(model, log='gradients', log_freq=config.print_freq * 5)
    else:
        print(f"{Fore.YELLOW}WandB logging disabilitato.{Style.RESET_ALL}")

    print(f"\n=== Inizializzazione Dataloader per il Fine-tuning ({args.dataset}) ===")
    
    spectrogram_transform = AudioToSpectrogram(
        n_mels=config.n_mel_bins,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length
    )
    
    current_dataset_raw = None
    if args.dataset == 'voxceleb':
        try:
            current_dataset_raw = VoxCelebGenderDataset(
                root=config.voxceleb_root,
                meta_file=config.voxceleb_meta_file,
                transform=spectrogram_transform
            )
        except RuntimeError as e:
            print(f"{Fore.RED}Errore nel caricamento del dataset VoxCeleb: {e}. Uscita.{Style.RESET_ALL}")
            return
        
        if config.voxceleb_percentage < 1.0:
            num_samples = int(len(current_dataset_raw) * config.voxceleb_percentage)
            indices = random.sample(range(len(current_dataset_raw)), num_samples)
            current_dataset = Subset(current_dataset_raw, indices)
            print(f"Selezionati {len(current_dataset)} campioni da VoxCeleb ({config.voxceleb_percentage*100:.1f}%).")
        else:
            current_dataset = current_dataset_raw
            print(f"Utilizzati tutti i {len(current_dataset)} campioni da VoxCeleb.")

    elif args.dataset == 'esc':
        try:
            current_dataset_raw = ESCAudioDataset(
                root=config.esc_audio_root,
                meta_file=config.esc_meta_file,
                transform=spectrogram_transform
            )
        except RuntimeError as e:
            print(f"{Fore.RED}Errore nel caricamento del dataset ESC-50: {e}. Uscita.{Style.RESET_ALL}")
            return
        
        if config.esc_percentage < 1.0:
            num_samples = int(len(current_dataset_raw) * config.esc_percentage)
            indices = random.sample(range(len(current_dataset_raw)), num_samples)
            current_dataset = Subset(current_dataset_raw, indices)
            print(f"Selezionati {len(current_dataset)} campioni da ESC-50 ({config.esc_percentage*100:.1f}%).")
        else:
            current_dataset = current_dataset_raw
            print(f"Utilizzati tutti i {len(current_dataset)} campioni da ESC-50.")

    else:
        print(f"{Fore.RED}Dataset di fine-tuning '{args.dataset}' non riconosciuto. Uscita.{Style.RESET_ALL}")
        return

    train_dataset, val_dataset = None, None
    if config.use_validation:
        dataset_size = len(current_dataset)
        val_size = int(config.validation_split_ratio * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            current_dataset, [train_size, val_size], generator=torch.Generator()
        )
        print(f"Dataset suddiviso: {len(train_dataset)} campioni per il training, {len(val_dataset)} per la validation.")
    else:
        train_dataset = current_dataset
        print("Validation disabilitata. L'intero dataset verrà usato per il training.")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
        collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1024) 
    )

    val_loader = None
    if config.use_validation and val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False, 
            num_workers=config.workers,
            pin_memory=config.pin_memory,
            collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1024)
        )
            
    print_header(args.dataset, config.use_validation, device, resume_info)
    
    # Scheduler
    total_epochs = args.epochs
    num_training_steps = len(train_loader) * total_epochs
    if num_training_steps == 0:
        print(f"{Fore.RED}AVVISO: num_training_steps è zero. Lo scheduler potrebbe non funzionare come previsto. Controlla la dimensione del dataset e il batch_size.{Style.RESET_ALL}")
        num_training_steps = 1
    num_warmup_steps = int(config.warmup_percentage * num_training_steps)
    
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=0.0,  
        power=1.0,   
        last_epoch=-1
    )

    if resume_info and 'checkpoint_data' in resume_info and 'scheduler_state_dict' in resume_info['checkpoint_data']:
        scheduler.load_state_dict(resume_info['checkpoint_data']['scheduler_state_dict'])

    checkpoint_epochs = config.finetuning_checkpoint_epochs

    all_iterations = []
    all_train_losses = []
    all_train_accuracies = []  
    all_val_losses = []     
    all_val_accuracies = []  
        
    patience_counter = 0 

    current_total_iterations = total_iterations_offset

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        
        lr = optimizer.param_groups[0]['lr']
        print_epoch_header(epoch, total_epochs, lr)

        # Training
        train_loss, train_accuracy, iters_in_epoch = train(  
            train_loader, model, optimizer, criterion, epoch, total_epochs, 
            config.print_freq, device, use_wandb=config.use_wandb,
            initial_step=current_total_iterations
        )
        
        current_total_iterations += iters_in_epoch

        all_iterations.append(current_total_iterations)
        all_train_losses.append(train_loss)
        all_train_accuracies.append(train_accuracy)  

        # Validation 
        val_loss, val_accuracy = None, None
        
        if config.use_validation and val_loader is not None:
            val_loss, val_accuracy = validate(val_loader, model, criterion, epoch, config.print_freq, device, use_wandb=config.use_wandb, initial_step=current_total_iterations) 
            
            all_val_losses.append(val_loss)  
            all_val_accuracies.append(val_accuracy)  

            is_best = val_accuracy > best_accuracy
            if is_best:
                best_loss = val_loss
                best_accuracy = val_accuracy 
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"{Fore.YELLOW}Early stopping attivato dopo {patience_counter} epoche senza miglioramenti.{Style.RESET_ALL}")
                    break
        else:
            is_best = val_accuracy > best_accuracy
            best_loss = min(train_loss, best_loss)
            best_accuracy = max(train_accuracy, best_accuracy)

        epoch_log_dict = {
            'epoch/train_loss': train_loss,
            'epoch/train_accuracy': train_accuracy, 
            'epoch/epoch': epoch + 1,
            'epoch/total_iterations': current_total_iterations,
        }
        if val_loss is not None:
            epoch_log_dict.update({
                'epoch/val_loss': val_loss,
                'epoch/val_accuracy': val_accuracy, 
            })
        
        if config.use_wandb:
            wandb.log(epoch_log_dict, step=current_total_iterations) 

        # Salvataggio checkpoint
        if (epoch + 1) in checkpoint_epochs or (epoch + 1) % 5 == 0: 
            state = {
                'epoch': epoch + 1,
                'total_iterations': current_total_iterations,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss, 
                'best_loss': best_loss, 
                'accuracy': train_accuracy,
                'val_loss': val_loss,      
                'val_accuracy': val_accuracy,  
                'best_accuracy': best_accuracy,  
                'config': vars(config) 
            }
            filename = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(state, filename=filename, is_best=is_best, save_to_wandb=False, use_wandb=config.use_wandb)

        epoch_time = time.time() - epoch_start
        print_training_summary(epoch, total_epochs, current_total_iterations, train_loss, train_accuracy, 
                               val_loss, val_accuracy, best_loss, lr, epoch_time, config.use_validation)

        scheduler.step() 

    plot_normalized_loss(args, all_iterations, all_train_losses, train=True, save_path=f'./plots/finetuning_loss_train_{args.masking_strategy}_{args.dataset}.png')  
    plot_normalized_loss(args, all_iterations, all_val_losses, train=False, save_path=f'./plots/finetuning_loss_val_{args.masking_strategy}_{args.dataset}.png') if config.use_validation else None

    if config.use_wandb:
        wandb.finish()
    else:
        print(f"\n{Fore.GREEN}Training completato!{Style.RESET_ALL}")
if __name__ == '__main__':
    main()
