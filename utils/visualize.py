import torch
import matplotlib.pyplot as plt
import os

def visualize_fbank(fbank, output_path=None):
    if isinstance(fbank, torch.Tensor):
        fbank = fbank.numpy()

    if fbank.ndim == 1:
        print(f"Nota: Trasformazione di fbank da forma {fbank.shape} a (1, {len(fbank)})")
        fbank = fbank.reshape(1, -1)

    plt.figure(figsize=(12, 8))
    plt.imshow(fbank, aspect='auto', origin='lower')
    plt.title('Fbank (Mel Filterbank Features)')
    plt.xlabel('Time frame')
    plt.ylabel('Mel bin')
    plt.colorbar(format='%+2.0f dB')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if output_path:
        plt.savefig(output_path+"/fbank.png")
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def visualize_spectrogram(spectrogram, output_path=None):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.numpy()
    
    # Se spectrogram o fbank sono 1D, devono essere trasformati in 2D per la visualizzazione
    if spectrogram.ndim == 1:
        print(f"Nota: Trasformazione di spectrogram da forma {spectrogram.shape} a (1, {len(spectrogram)})")
        spectrogram = spectrogram.reshape(1, -1)
    
        
    plt.figure(figsize=(12, 8))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.ylabel('Frequency bin')
    plt.colorbar(format='%+2.0f dB')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if output_path:
        plt.savefig(output_path+"/spectrogram.png")
        print(f"Saved visualization to {output_path}")
    
    plt.show()

    
def visualize_patches(patches, title=""):
    print(f"Visualizing {title}") 
    print(patches.shape)
    print(patches)