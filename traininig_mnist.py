import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mae import MAE
from torch.utils.data import DataLoader, Subset
import random
from types import SimpleNamespace
import shutil
import torch.nn.functional as F 
from losses import infoNCE_loss, mae_loss

# --------------------------------------------------
# DATA LOADER per MNIST
# --------------------------------------------------
def data_loader_mnist(root, batch_size=256, workers=4, pin_memory=True, sample_size=None):
    """
    Restituisce train_loader e val_loader per MNIST.
    Se sample_size è specificato, sottocampiona i dataset.
    """
    # Normalize con i parametri standard di MNIST
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # Trasformazioni per train e validation (uguali, dato che MNIST è semplice)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Dataset MNIST
    train_dataset = datasets.MNIST(root=root,
                                   train=True,
                                   download=True,
                                   transform=transform)

    val_dataset = datasets.MNIST(root=root,
                                 train=False,
                                 download=True,
                                 transform=transform)

    # Se vogliamo usare un sottoinsieme
    if sample_size is not None:
        train_indices = random.sample(range(len(train_dataset)), min(sample_size, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(sample_size, len(val_dataset)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
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
    Decade il learning rate di 0.8 ogni 30 epoche.
    """
    lr = init_lr * (0.8 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Actual lr: {lr}")

# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------
def train(train_loader, model, optimizer, epoch, print_freq=100, device='cuda', pretraining=True):
    """
    Loop di training per un epoch su MNIST:
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
            # Il modello MAE restituisce: target_patches, recon_logits, class_logits (non usato), bool_mask
            target, recon_logits, class_logits = model(input)
            recon_loss = mae_loss(target, recon_logits)
            class_loss = infoNCE_loss(target, class_logits)
            loss = recon_loss * 1 + class_loss * 1

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
def validate(val_loader, model, print_freq=100, device='cuda', pretraining=True):
    """
    Loop di validazione su MNIST:
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
                recon_loss = mae_loss(target, recon_logits)
                class_loss = infoNCE_loss(target, class_logits)
                loss = recon_loss * 1 + class_loss * 1
            
            # Aggiorna misure
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(f'Validate [{i}/{len(val_loader)}]\t'
                      f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})'
                      f'Recon: {recon_loss}, infoNCE: {class_loss}')
    print(f' * Validation MSE Loss (avg) {losses.avg:.4f}')
    return losses.avg

# --------------------------------------------------
# MAIN per MNIST
# --------------------------------------------------

def main():
    # Configurazione base per MNIST
    config = SimpleNamespace(
        # Dimensioni immagine / patch per MNIST (28×28 grayscale)
        img_size = (28, 28),
        patch_size = (4, 4),       # es. patch 7×7 → 4×4 patches (28/7=4)
        num_channels = 1,
        # Parametri Encoder (modifica se necessario)
        enc_embed_dim = 768,
        enc_mlp_layer_dim = 512,
        enc_hidden_layers = 12,
        enc_attention_heads = 12,
        enc_layer_norm_eps = 1e-6,
        enc_attention_dropout = 0.0,
        enc_mlp_ratio = 4,
        # Parametri Decoder (modifica se necessario)
        dec_hidden_layers = 8,
        dec_embed_dim = 512,
        dec_attention_heads = 16,
        dec_layer_norm_eps = 1e-6,
        dec_attention_dropout = 0.0,
        dec_mlp_ratio = 4,
        # Masking
        masking_strategy = "random",
        masking_percentage = 0.75,
        # Training
        batch_size = 32,
        initial_lr = 1e-3,
        weight_decay = 0.01,
        epochs = 100,
        print_freq = 100,
        # Cartella dataset
        mnist_root = "./data_mnist",
        # Checkpoint
        checkpoints_dir = "./checkpoints_mae_mnist"
    )

    print(config)

    # Crea cartella checkpoint se non esiste
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # Istanzio il modello MAE (adattato a 1 canale e input 28×28)
    model = MAE(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ottimizzatore e criterio (MSE)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.initial_lr,
                           weight_decay=config.weight_decay)

    # DataLoader MNIST
    train_loader, val_loader = data_loader_mnist(
        root=config.mnist_root,
        batch_size=config.batch_size,
        workers=2,
        pin_memory=True,
        sample_size=1000  # metti un numero se vuoi limitare il numero di esempi
    )

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        adjust_learning_rate(optimizer, epoch, config.initial_lr)

        # TRAIN
        avg_train_loss = train(train_loader, model, optimizer,
                               epoch, print_freq=config.print_freq, device=device, pretraining=True)

        # VALIDATE
        avg_val_loss = validate(val_loader, model,
                                print_freq=config.print_freq, device=device, pretraining=True)

        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        # Salva checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict()
        }
        
        filename = os.path.join(config.checkpoints_dir, f"checkpoint_last_epoch.pth")
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(config.checkpoints_dir, "model_best.pth"))

        print(f"Epoch {epoch+1}/{config.epochs} fatto. "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"({'*' if is_best else ''} best: {best_val_loss:.4f})")

if __name__ == '__main__':
    main()
