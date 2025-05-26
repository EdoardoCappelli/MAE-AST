#!/usr/bin/env python3
import os
import shutil
import zipfile
import urllib.request

def download_file(url, dest_path):
    """
    Scarica un file da URL e lo salva in dest_path.
    Se il file esiste già, non lo riscarica.
    """
    if os.path.isfile(dest_path):
        print(f"Il file '{dest_path}' esiste già, skip download.")
        return

    print(f"Scarico da {url} → {dest_path} ...")
    # Crea la cartella di destinazione solo se dirname non è vuoto
    parent_dir = os.path.dirname(dest_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Scaricamento con progress bar minima
    def _progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(int(downloaded * 100 / total_size), 100)
            print(f"\r{percent}% ({downloaded // 1024} KB di {total_size // 1024} KB)", end='')
        else:
            print(f"\r{downloaded // 1024} KB scaricati", end='')

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
    print("\nDownload completato.")

def unzip_dataset(zip_path, extract_to='.'):
    """
    Estrae lo ZIP del Tiny ImageNet nella cartella extract_to.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file non trovato: {zip_path}")
    print(f"Decomprimo '{zip_path}' in '{extract_to}' ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path=extract_to)
    print("Decompressione completata.")

def process_training_data(base_dir):
    """
    Per ogni cartella di classe in train/, rimuove i .txt,
    sposta le immagini dalla sottocartella images/ alla cartella di classe,
    e infine rimuove la cartella images/.
    """
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Directory di training non trovata: {train_dir}")

    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # ignora eventuali file non-directory

        # 1) Rimuovi tutti i file .txt nella directory della classe
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.txt'):
                os.remove(os.path.join(class_dir, fname))

        # 2) Sposta tutte le immagini da images/ alla cartella della classe
        images_subdir = os.path.join(class_dir, 'images')
        if os.path.isdir(images_subdir):
            for img_fname in os.listdir(images_subdir):
                src_path = os.path.join(images_subdir, img_fname)
                dst_path = os.path.join(class_dir, img_fname)
                shutil.move(src_path, dst_path)
            # 3) Rimuovi la cartella images/ vuota
            shutil.rmtree(images_subdir)

    print("Elaborazione dati di training completata.")

def process_validation_data(base_dir):
    """
    Legge val/val_annotations.txt, crea una cartella per ogni classe,
    sposta ciascuna immagine nella cartella corrispondente, 
    poi rimuove la directory val/images/.
    """
    val_dir = os.path.join(base_dir, 'val')
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Directory di validation non trovata: {val_dir}")

    annotate_file = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.isfile(annotate_file):
        raise FileNotFoundError(f"File di annotazioni non trovato: {annotate_file}")

    images_subdir = os.path.join(val_dir, 'images')
    if not os.path.isdir(images_subdir):
        raise FileNotFoundError(f"Directory delle immagini di validation non trovata: {images_subdir}")

    # Leggi tutte le righe
    with open(annotate_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        filename, cls = parts[0], parts[1]
        src_path = os.path.join(images_subdir, filename)
        if not os.path.isfile(src_path):
            # se il file non esiste, saltalo
            continue

        # Crea la cartella di classe dentro val/, se non esiste
        class_folder = os.path.join(val_dir, cls)
        os.makedirs(class_folder, exist_ok=True)

        # Sposta l'immagine nella cartella di classe
        dst_path = os.path.join(class_folder, filename)
        shutil.move(src_path, dst_path)

    # Rimuovi la cartella images/ ora vuota
    shutil.rmtree(images_subdir)
    print("Elaborazione dati di validation completata.")

if __name__ == '__main__':
    # URL del dataset Tiny ImageNet
    tiny_imagenet_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_filename = 'tiny-imagenet-200.zip'  # nome del file locale

    # 1) Scarica lo ZIP se non esiste
    download_file(tiny_imagenet_url, zip_filename)

    # 2) Decomprimi lo ZIP
    unzip_dataset(zip_filename, extract_to='.')

    # 3) Definisci il percorso principale del dataset estratto
    current_dir = os.path.abspath('tiny-imagenet-200')

    # 4) Elabora i dati di training
    process_training_data(current_dir)

    # 5) Elabora i dati di validation
    process_validation_data(current_dir)

    print("Tutte le operazioni sono terminate con successo.")
