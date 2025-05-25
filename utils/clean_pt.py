import os
import torch
from pathlib import Path
from config import Config

def elimina_pt_invalidi(root_dir, target_shape=(1, 1, 128, 998)):
    """
    Scorre ricorsivamente tutti i file .pt sotto root_dir e
    rimuove quelli la cui shape != target_shape.

    Parametri:
        root_dir (str o Path): cartella di partenza (può contenere sottocartelle)
        target_shape (tuple): forma esatta che ci si aspetta dal tensore
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        print(f"[ERROR] '{root_dir}' non è una directory valida.")
        return

    # Conta quanti file .pt vengono controllati e quanti eliminati
    tot_controllati = 0
    tot_eliminati = 0

    # Scansione ricorsiva di tutti i file .pt
    for file_path in root_dir.rglob("*.pt"):
        tot_controllati += 1
        try:
            # Carico il tensore (in CPU per sicurezza)
            tensore = torch.load(file_path, map_location="cpu")
            
            # Controllo che abbia attributo .shape (es. torch.Tensor)
            if not hasattr(tensore, "shape"):
                raise ValueError("L'oggetto caricato non è un tensore con .shape")

            if tuple(tensore.shape) != target_shape:
                # Se la shape non coincide, elimino
                file_path.unlink()
                tot_eliminati += 1
                print(f"[ELIMINATO] {file_path} → shape={tuple(tensore.shape)}")
        except Exception as e:
            # Se c'è un errore durante il caricamento o shape, elimino comunque per sicurezza
            try:
                file_path.unlink()
                tot_eliminati += 1
                print(f"[ELIMINATO (errore caricamento)] {file_path} → {e}")
            except Exception as e2:
                print(f"[ERRORE] Impossibile eliminare {file_path}: {e2}")

    print(f"\nControllo completato.\nTotale .pt controllati: {tot_controllati}\nTotale .pt eliminati: {tot_eliminati}")

if __name__ == "__main__":

    config = Config()

    elimina_pt_invalidi(config.tensor_dir)
    