import os
import requests
import zipfile
from io import BytesIO

def download_and_extract_esc50(dest_dir: str = "./data/ESC",
                               url: str = "https://github.com/karoldvl/ESC-50/archive/master.zip"):
    # Crea la directory di destinazione se non esiste
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Scaricando da {url} …")
    resp = requests.get(url)
    resp.raise_for_status()

    print("Estraendo i file …")
    # Apri l’zip in memoria
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        # Estrarre in una cartella temporanea interna
        temp_root = zf.namelist()[0].rstrip("/")  # es. "ESC-50-master"
        for member in zf.infolist():
            # rimuove il prefisso "ESC-50-master/"
            rel_path = os.path.relpath(member.filename, temp_root)
            # percorso completo di destinazione
            target_path = os.path.join(dest_dir, rel_path)
            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                # crea la cartella padre se necessario
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    print(f"Tutti i file sono stati estratti in `{dest_dir}`.")

if __name__ == "__main__":
    download_and_extract_esc50()
