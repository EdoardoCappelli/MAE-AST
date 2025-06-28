import os
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

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
    
