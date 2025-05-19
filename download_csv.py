import os
import argparse
from urllib.parse import urljoin
import requests
from pathlib import Path
import subprocess
import pandas as pd

def download_segment(youtube_id: str, start: float, end: float, out_path: Path):
    """
    Usa yt-dlp + ffmpeg per scaricare ed estrarre il segmento audio [start,end]
    e salvarlo in WAV.
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    # Costruiamo la sezione nel formato ffmpeg/yt-dlp
    section = f"*{start:.3f}-{end:.3f}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        url,
        "--quiet",
        "--no-warnings",
        "--extract-audio",
        "--audio-format", "wav",
        "--output", str(out_path),
        "--external-downloader", "ffmpeg",
        "--download-sections", section
    ]
    subprocess.run(cmd, check=True)

def process_csv(csv_path: Path, wav_dir: Path):
    """
    Legge un CSV, itera sulle righe e chiama download_segment.
    Il file wav viene nominato YTID_start_end.wav
    """
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        ytid = row['YTID']
        start = float(row['start_seconds'])
        end   = float(row['end_seconds'])
        # es. outputs/audioset201906/wavs/eval_segments/--PJHxphWEs_30.000_40.000.wav
        subdir = csv_path.stem  # es. 'eval_segments'
        filename = f"{ytid}_{start:06.3f}_{end:06.3f}.wav"
        out_path = wav_dir / subdir / filename
        print(f"Scarico {ytid} [{start:.1f}s–{end:.1f}s] → {out_path}")
        try:
            download_segment(ytid, start, end, out_path)
        except subprocess.CalledProcessError:
            print(f"⚠️ Errore con {ytid}, salto.")

def download_file(url: str, dest: Path):
    """Scarica un file da URL e lo salva in dest."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    parser = argparse.ArgumentParser(
        description="Scarica i csv di AudioSet nella cartella desiderata"
    )
    parser.add_argument(
        'dataset_dir',
        nargs='?',
        default='C:/Users/admin/Desktop/VS Code/VisualTranformer/datasets/csv',
    )
    args = parser.parse_args()
    base_dir = Path(args.dataset_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("------ Download metadata ------")

    # Base URL per i CSV
    base_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/"

    files = {
        "eval_segments.csv": "csv/eval_segments.csv",
        "balanced_train_segments.csv":   "csv/balanced_train_segments.csv",
        "unbalanced_train_segments.csv": "csv/unbalanced_train_segments.csv",
        "class_labels_indices.csv": "csv/class_labels_indices.csv",
        # QA counts in percorso a sé stante
        "qa_true_counts.csv": "qa/qa_true_counts.csv",
    }

    for filename, path in files.items():
        url = urljoin(base_url, path)
        dest = base_dir / filename
        print(f"Downloading {filename} …")
        download_file(url, dest)

    print(f"Download completato in {base_dir.resolve()}")


if __name__ == "__main__":
    main()
