import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import Config

def convert_npy_to_tensors(input_dir, output_dir):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Ottieni tutti i file .npy nella cartella di input
    input_path = Path(input_dir)
    npy_files = list(input_path.glob("*.npy"))
    
    print(f"Trovati {len(npy_files)} file .npy in {input_dir}")
    
    # Processa ogni file
    for npy_file in tqdm(npy_files, desc="Conversione file"):
        try:
            # Carica il file .npy
            spectrogram_np = np.load(str(npy_file))
            print(str(npy_file))
            # Converti in tensor e applica le trasformazioni
            spectrogram = torch.from_numpy(spectrogram_np)
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # [B, C, W, H]
            spectrogram = spectrogram.permute(0, 1, 3, 2)  # [B, C, H, W]
            
            # Crea il path di output mantenendo lo stesso nome file ma in una cartella diversa
            output_file = Path(output_dir) / f"{npy_file.stem}.pt"
            
            # Salva il tensore
            torch.save(spectrogram, output_file)
            
        except Exception as e:
            print(f"Errore durante l'elaborazione del file {npy_file}: {e}")
    
    print(f"Conversione completata. Risultati salvati in {output_dir}")


def main():

    config = Config()

    convert_npy_to_tensors(config.npy_dir, config.tensor_dir)

if __name__ == '__main__':
    main()