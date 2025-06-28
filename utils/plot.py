import os
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
import torch 
from typing import Optional, Callable, Tuple, List
from config import Config 

def plot_normalized_loss(args, steps, loss, train=True, save_path = None):
    if args.epochs >= 8:
        steps = np.array(steps)
        loss = np.array(loss, dtype=float)

        # Normalize loss
        loss_min = loss.min()
        loss_max = loss.max()
        loss_norm = (loss - loss_min) / (loss_max - loss_min)

        # Sample n_points equispaced
        indices = np.linspace(0, len(steps) - 1, 8, dtype=int)
        steps_s = steps[indices]
        loss_s = loss_norm[indices]

        # Plot
        plt.figure(figsize=(12, 3))
        plt.plot(steps_s, loss_s, marker='o', linestyle='-')
        plt.locator_params(axis='y', nbins=5)
        plt.xlabel('Step')
        plt.ylabel('Normalized Loss')
        if train:
            plt.title("Normalized Training Loss vs. Steps")
        else:
            plt.title("Normalized Validation Loss vs. Steps")
        plt.grid(True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    else:
        print(f"{Fore.RED}Warning: Not enough epochs to plot loss.{Style.RESET_ALL}")
        print(f"{Fore.RED}You need at least 8 epochs to plot the loss.{Style.RESET_ALL}")
        print(f"{Fore.RED}Current epochs: {args.epochs}.{Style.RESET_ALL}")
        print(f"{Fore.RED}Plotting skipped.{Style.RESET_ALL}")

def visualize_spectrogram_patches_with_masking(
        original_patches: torch.Tensor,
        spectrogram: torch.Tensor,
        patch_size: Tuple[int, int] = (32, 32),
        bool_mask: torch.Tensor = None,
        masked_indices: List[torch.Tensor] = None,
        batch_idx: int = 0,
        figsize: Tuple[int, int] = (18, 6),
        cmap: str = 'viridis',
        config: Config = None,
    ) -> None:
    
    if spectrogram.dim() == 3:
        spectrogram = spectrogram.unsqueeze(0)

    spec_batch = spectrogram[batch_idx]  # (C, H, W)
    patches_batch = original_patches[batch_idx]  # (num_patches, patch_dim)

    ph, pw = patch_size
    num_channels = spec_batch.shape[0]
    num_patches = patches_batch.shape[0]

    patches = patches_batch.view(num_patches, num_channels, ph, pw)

    C, H, W = spec_batch.shape
    n_patches_h = H // ph
    n_patches_w = W // pw
    total_patches = n_patches_h * n_patches_w

    plt.figure(figsize=(12, 3))
    img = spec_batch[0].detach().cpu().detach().numpy()
    plt.imshow(img, cmap=cmap, aspect='auto', origin='lower')
    plt.title('Spettrogramma Originale')
    plt.axis('off')
    plt.tight_layout()
    os.makedirs("./plots/", exist_ok=True)
    if config.random:
        plt.savefig(f'./plots/spectrogram_batch_{batch_idx}_random_{args.masking_strategy}.png', bbox_inches='tight', dpi=150)
    else:
        plt.savefig(f'./plots/spectrogram_batch_{batch_idx}_chunked_{args.masking_strategy}.png', bbox_inches='tight', dpi=150)
    plt.close() 


    patch_display_size = 0.8  # Ridotto da 1.5 a 0.8 pollici per patch
    fig_width = min(n_patches_w * patch_display_size, 12)
    fig_height = min(n_patches_h * patch_display_size, 3)

    fig, axes = plt.subplots(n_patches_h, n_patches_w,
                             figsize=(fig_width, fig_height))


    if total_patches == 1:
        axes = np.array([axes]) # Ensure it's iterable
    else:
        axes = axes.flatten()

    for i in range(total_patches):
        ax = axes[i]

        if i < patches.shape[0]:
            # Usa il primo canale della patch
            patch_img = patches[i, 0].detach().cpu().detach().numpy()

            # Imposta le dimensioni esatte del plot per rispettare patch_size
            im = ax.imshow(patch_img, cmap=cmap, aspect='auto', origin='lower')

        else:
            # Patch vuote per completare la griglia
            ax.set_visible(False)

        ax.axis('off')


    plt.subplots_adjust(left=0.05, bottom=0.3, right=0.9, top=0.95, wspace=0.05, hspace=0.05)

    os.makedirs("./plots/", exist_ok=True)
    if config.random:
        plt.savefig(f'plots/patches_batch_{batch_idx}_random_{args.masking_strategy}.png', bbox_inches='tight', dpi=150)
    else:
        plt.savefig(f'plots/patches_batch_{batch_idx}_chunked_{args.masking_strategy}.png', bbox_inches='tight', dpi=150)
    plt.close()

    if bool_mask is not None or masked_indices is not None:
        # Determine masked indices
        if masked_indices is not None and batch_idx < len(masked_indices):
            masked_idx = masked_indices[batch_idx].cpu().detach().numpy()
        elif bool_mask is not None:
            batch_mask = bool_mask.view(spectrogram.shape[0], -1)[batch_idx]
            masked_idx = torch.where(batch_mask)[0].cpu().detach().numpy()
        else:
            masked_idx = []

        fig, axes = plt.subplots(n_patches_h, n_patches_w,
                                       figsize=(min(fig_width, 12), min(fig_height, 3)))

        if total_patches == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for i in range(total_patches):
            ax = axes[i]

            if i < patches.shape[0]:
                patch_img = patches[i, 0].detach().cpu().detach().numpy()

                alpha = 0.3 if i in masked_idx else 1.0

                ax.imshow(patch_img, aspect='auto',
                                  alpha=alpha, origin='lower')

                title_color = 'red' if i in masked_idx else 'black'
            else:
                ax.set_visible(False)

            ax.axis('off')

        plt.subplots_adjust(left=0.05, bottom=0.3, right=0.9, top=0.95, wspace=0.05, hspace=0.05)
        os.makedirs("./plots/", exist_ok=True)
        if config.random:
            plt.savefig(f'plots/masking_batch_{batch_idx}_random_{args.masking_strategy}.png', bbox_inches='tight', dpi=150)
        else:
            plt.savefig(f'plots/masking_batch_{batch_idx}_chunked_{args.masking_strategy}.png', bbox_inches='tight', dpi=150)

        plt.close()
        

def visualize_spectrogram_reconstruction(
        original_spectrogram: torch.Tensor,
        reconstructed_patches: torch.Tensor,
        bool_mask: torch.Tensor,
        patch_size: Tuple[int, int] = (32, 32),
        batch_idx: int = 0,
        cmap: str = 'viridis'
    ) -> None:

    if original_spectrogram.dim() == 3:
        original_spectrogram = original_spectrogram.unsqueeze(0)

    spec = original_spectrogram[batch_idx]  # (C, H, W)
    ph, pw = patch_size
    C, H, W = spec.shape
    n_patches_h = H // ph
    n_patches_w = W // pw

    recon = spec.clone()
    mask = bool_mask.view(original_spectrogram.shape[0], -1)[batch_idx]
    masked_idx = torch.where(mask)[0].cpu().detach().numpy()
    for i, pidx in enumerate(masked_idx):
        row, col = divmod(int(pidx), n_patches_w)
        y0, x0 = row * ph, col * pw
        if i < reconstructed_patches.shape[1]:
            patch = reconstructed_patches[batch_idx, i].view(C, ph, pw)
            recon[:, y0:y0+ph, x0:x0+pw] = patch

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    images = [
        (spec[0].cpu().detach().numpy(), 'Originale', cmap),
        (recon[0].cpu().detach().numpy(), 'Ricostruito', cmap),
        (abs(spec - recon)[0].cpu().detach().numpy(), 'Differenza', 'hot')
    ]

    for ax, (img, title, cmap_i) in zip(axs, images):
        ax.imshow(img, origin='lower', aspect='auto', cmap=cmap_i)
        ax.set_title(title)
        ax.axis('off')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)

    os.makedirs("./plots/", exist_ok=True)
    
    plt.savefig(f'plots/spectrogram_reconstruction_batch_{batch_idx}.png',
                bbox_inches='tight', dpi=150)
    plt.close()  

def crop_or_repeat(spec: torch.Tensor, target_width: int):
    """
    Applica un crop dalla coda o ripete lo spettrogramma dall'inizio per raggiungere la target_width.
    """
    current_width = spec.shape[-1]

    if current_width > target_width:
        return spec[..., -target_width:]

    elif current_width < target_width:
        repeat_times = target_width // current_width
        remainder = target_width % current_width

        if repeat_times > 1:
            repeated_spec = spec.repeat(1, repeat_times)
        else:
            repeated_spec = spec

        if remainder > 0:
            repeated_spec = torch.cat([repeated_spec, spec[..., :remainder]], dim=-1)

        return repeated_spec
    else:
        return spec
    
