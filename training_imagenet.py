import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mae import MAE
from torch.utils.data import Subset
import random

# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------
def data_loader(root_train, root_val, batch_size=256, workers=4, pin_memory=True, sample_size=None):
    """
    Carica immagini da due cartelle (train e val), applica transform e restituisce due DataLoader.
    Il target restituito (la classe) verrà ignorato perché il MAE non usa label di classificazione.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        root_train,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        root_val,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    if sample_size is not None:
        train_indices = random.sample(range(len(train_dataset)), min(sample_size, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(sample_size, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

# --------------------------------------------------
# AVERAGE METER (per tempi e loss)
# --------------------------------------------------
class AverageMeter(object):
    """Tiene traccia di media e valore corrente."""
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

# --------------------------------------------------
# ADJUST LEARNING RATE
# --------------------------------------------------
def adjust_learning_rate(optimizer, epoch, init_lr):
    """
    Decade il learning rate di 0.1 ogni 30 epoche.
    """
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------
def train(train_loader, model, criterion, optimizer, epoch, print_freq=100, device='cuda', pretraining=True):
    """
    Loop di training per un epoch:
    - esegue forward del MAE, prende recon_logits e target_patches
    - calcola MSE loss
    - aggiorna i pesi
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.to(device, non_blocking=True)

        # Forward
        if pretraining:
            target, recon_logits, class_logits = model(input)
            loss = criterion(recon_logits, target)

        # Backward + optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggiorna misure
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Epoch [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
    return losses.avg

# --------------------------------------------------
# VALIDATE FUNCTION
# --------------------------------------------------
def validate(val_loader, model, criterion, print_freq=100, device='cuda', pretraining=True):
    """
    Loop di validazione:
    - forward del MAE su batch di validazione
    - calcola solamente MSE loss sui patch (nessuna accuracy)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, _) in enumerate(val_loader):
            input = input.to(device, non_blocking=True)

            # Forward
            if pretraining:
                target, recon_logits, class_logits = model(input)
                loss = criterion(recon_logits, target)

            # Aggiorna misure
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(f'Validate [{i}/{len(val_loader)}]\t'
                      f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})')
    print(f' * Validation MSE Loss (avg) {losses.avg:.4f}')
    return losses.avg

# --------------------------------------------------
# MAIN
# --------------------------------------------------
from types import SimpleNamespace
import shutil

def main():
    # Configurazione base
    config = SimpleNamespace(
        # Dimensioni immagine / patch
        img_size = (224, 224),
        patch_size = (16, 16),
        channels = 3,
        n_mel_bins = 224,
        patch_embedding_dropout = 0.0,
        num_channels = 3,

        # MAE Encoder
        enc_embed_dim = 768,
        enc_mlp_layer_dim = 3072,
        enc_hidden_layers = 6,
        enc_attention_heads = 12,
        enc_layer_norm_eps = 1e-6,
        enc_attention_dropout = 0.0,
        enc_mlp_ratio = 4,

        # MAE Decoder
        dec_hidden_layers = 2,
        dec_embed_dim = 768,
        dec_attention_heads = 12,
        dec_layer_norm_eps = 1e-6,
        dec_attention_dropout = 0.0,
        dec_mlp_ratio = 4,
        
        # Masking
        masking_strategy = "random",
        masking_percentage = 0.75,

        # Training
        batch_size = 32,
        initial_lr = 1e-4,
        weight_decay = 0.05,
        epochs = 50,
        print_freq = 50,

        # Cartelle dataset
        train_dir = r"tiny-imagenet-200/train",
        val_dir   = r"tiny-imagenet-200/val",
        test_dir = r"tiny-imagenet-200/test",
        # Checkpoint
        checkpoints_dir = "C:/Users/admin/Desktop/VS Code/MAE/checkpoints_mae_imagenet"
    )

    # Crea cartella checkpoint se non esiste
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # Istanzio il modello MAE
    model = MAE(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ottimizzatore e criterio (MSE)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.initial_lr,
                           weight_decay=config.weight_decay)
    criterion = torch.nn.MSELoss()

    # DataLoader
    train_loader, val_loader = data_loader(
        config.train_dir, 
        config.val_dir,
        batch_size=config.batch_size, 
        workers=4, 
        pin_memory=True,
        sample_size=100
    )

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        adjust_learning_rate(optimizer, epoch, config.initial_lr)

        # TRAIN
        avg_train_loss = train(train_loader, model, criterion, optimizer,
                               epoch, print_freq=config.print_freq, device=device, pretraining=True)

        # VALIDATE
        avg_val_loss = validate(val_loader, model, criterion,
                                print_freq=config.print_freq, device=device, pretraining=True)

        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict()
        }
        filename = os.path.join(config.checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(config.checkpoints_dir, "model_best.pth"))

        print(f"Epoch {epoch+1}/{config.epochs} ready. "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"({'*' if is_best else ''} best: {best_val_loss:.4f})")

if __name__ == '__main__':
    main()
