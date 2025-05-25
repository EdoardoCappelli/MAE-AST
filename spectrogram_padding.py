import torch
import torch.nn.functional as F
import os
import glob

# --- Configurazione ---
INPUT_DIR = "C:/Users/admin/Desktop/VS Code/MAE-AST/dataset"
OUTPUT_DIR = "C:/Users/admin/Desktop/VS Code/MAE-AST/spectrograms_padded/"

# Specifica quale dimensione del tensore rappresenta la "lunghezza" da paddare.
# Esempi comuni per spettrogrammi 2D:
# - Se la forma è (Tempo, Frequenze), allora LENGTH_DIM_INDEX = 0
# - Se la forma è (Frequenze, Tempo), allora LENGTH_DIM_INDEX = 1
# Modifica questo valore in base alla forma dei tuoi tensori.
LENGTH_DIM_INDEX = 3

PADDING_VALUE = 0.0  # Valore usato per il padding

# --- 1. Trova la lunghezza massima ---
max_length = 0
# Cerca tutti i file .pt nella directory di input e sottodirectory
file_paths = glob.glob(os.path.join(INPUT_DIR, "**", "*.pt"), recursive=True)

if not file_paths:
    print(f"Nessun file .pt trovato in '{INPUT_DIR}' o nelle sue sottodirectory.")
    exit()

print(f"Trovati {len(file_paths)} file .pt. Inizio Pass 1: Ricerca lunghezza massima...")

for f_path in file_paths:
    try:
        tensor = torch.load(f_path, map_location='cpu') # Carica su CPU per evitare problemi di memoria GPU
        print(tensor.shape)
        if not isinstance(tensor, torch.Tensor):
            print(f"Attenzione: Il file {f_path} non contiene un tensore PyTorch. Saltato.")
            continue

        if tensor.ndim <= LENGTH_DIM_INDEX:
            print(f"Attenzione: Il tensore in {f_path} ha solo {tensor.ndim} dimensioni, "
                  f"impossibile accedere alla dimensione {LENGTH_DIM_INDEX}. Saltato.")
            continue

        current_length = tensor.shape[LENGTH_DIM_INDEX]
        if current_length > max_length:
            max_length = current_length
    except Exception as e:
        print(f"Errore durante il caricamento o l'analisi di {f_path}: {e}")
        continue

if max_length == 0 and file_paths: # Se ci sono file ma non si è trovata una lunghezza valida
    print("Non è stato possibile determinare una lunghezza massima valida dai tensori analizzati.")
    exit()
elif not file_paths: # Già gestito, ma per sicurezza
    exit()


print(f"Lunghezza massima trovata nella dimensione {LENGTH_DIM_INDEX}: {max_length}")

# --- 2. Applica il padding e salva i tensori ---
print(f"\nInizio Pass 2: Applicazione del padding ai tensori (target: {max_length}) e salvataggio in '{OUTPUT_DIR}'...")

# Crea la directory di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

for f_path in file_paths:
    try:
        tensor = torch.load(f_path, map_location='cpu')
        if not isinstance(tensor, torch.Tensor) or tensor.ndim <= LENGTH_DIM_INDEX:
            # Questo controllo è ridondante se il Pass 1 ha funzionato, ma è una buona pratica
            print(f"Saltato file non valido o con dimensioni errate nel Pass 2: {f_path}")
            continue

        current_length = tensor.shape[LENGTH_DIM_INDEX]
        padding_needed = max_length - current_length

        padded_tensor = tensor # Inizia assumendo che non sia necessario il padding

        if padding_needed > 0:
            # Costruisci la tupla di padding per F.pad
            # La tupla di padding ha 2*N elementi per un tensore N-dimensionale.
            # Gli elementi sono (pad_ultima_dim_inizio, pad_ultima_dim_fine, pad_penultima_dim_inizio, ...)
            
            # Inizializza la tupla di padding con tutti zeri
            padding_tuple_list = [0] * (tensor.ndim * 2)
            
            # Calcola l'indice della dimensione da paddare partendo dalla fine (come richiesto da F.pad)
            # Se LENGTH_DIM_INDEX è 0 (prima dimensione), e il tensore ha N dimensioni,
            # per F.pad questa è la (N-1)-esima dimensione da destra (0-indexed).
            dim_to_pad_idx_from_right = tensor.ndim - 1 - LENGTH_DIM_INDEX
            
            # La posizione nella tupla di padding per il "padding alla fine" di questa dimensione
            # è 2 * dim_to_pad_idx_from_right + 1
            padding_tuple_list[2 * dim_to_pad_idx_from_right + 1] = padding_needed
            
            padded_tensor = F.pad(tensor, tuple(padding_tuple_list), mode="constant", value=PADDING_VALUE)
            print(f"Applicato padding a {f_path}. Shape originale: {tensor.shape}, Nuova shape: {padded_tensor.shape}")
        elif padding_needed < 0:
            print(f"Attenzione: Il tensore {f_path} (lunghezza {current_length}) è più lungo della lunghezza massima ({max_length}) trovata nel primo passaggio. Questo non dovrebbe accadere. Il tensore verrà copiato senza modifiche.")
        else: # padding_needed == 0
            print(f"Nessun padding necessario per {f_path}. Lunghezza: {current_length}.")

        # Ricrea la struttura di cartelle dell'input nell'output
        relative_path = os.path.relpath(os.path.dirname(f_path), INPUT_DIR)
        current_output_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        file_name = os.path.basename(f_path)
        output_path = os.path.join(current_output_dir, file_name)
        
        torch.save(padded_tensor, output_path)

    except Exception as e:
        print(f"Errore durante il padding o il salvataggio di {f_path}: {e}")

print("\nElaborazione completata.")