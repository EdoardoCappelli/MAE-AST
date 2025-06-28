#!/usr/bin/env python3
"""
Script per scaricare e processare gli audio di AudioSet dal CSV balanced_train_segments.csv
Versione che NON richiede ffmpeg - usa solo yt-dlp per tutto il processing
"""

import pandas as pd
import os
import subprocess
import sys
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioSetDownloader:
    def __init__(self, csv_path, output_dir, max_workers=4):
        """
        Inizializza il downloader AudioSet
        
        Args:
            csv_path (str): Percorso al file CSV di AudioSet
            output_dir (str): Directory di output per gli audio
            max_workers (int): Numero di thread per il download parallelo
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verifica che yt-dlp sia installato
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verifica che yt-dlp sia installato"""
        try:
            subprocess.run(['yt-dlp', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("yt-dlp non è installato. Installa con: pip install yt-dlp")
            sys.exit(1)
    
    def load_csv(self):
        """Carica e processa il CSV di AudioSet"""
        logger.info(f"Caricamento CSV: {self.csv_path}")
        
        # Leggi il CSV saltando le righe di commento
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()
        
        # Trova la prima riga che non inizia con #
        data_start = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                data_start = i
                break
        
        # Crea DataFrame con colonne appropriate
        data_lines = lines[data_start:]
        data = []
        
        for line in data_lines:
            if line.strip():
                parts = line.strip().split(', ')
                if len(parts) >= 4:
                    ytid = parts[0]
                    if ytid.startswith('"'):
                        ytid = ytid[1:]
                    start_time = float(parts[1])
                    end_time = float(parts[2])
                    labels = parts[3]
                    data.append({
                        'ytid': ytid,
                        'start_seconds': start_time,
                        'end_seconds': end_time,
                        'positive_labels': labels
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Caricati {len(df)} segmenti audio")
        return df
    
    def download_segment(self, row):
        """
        Scarica un singolo segmento audio usando solo yt-dlp
        
        Args:
            row: Riga del DataFrame con le informazioni del segmento
            
        Returns:
            tuple: (success, ytid, error_message)
        """
        ytid = row['ytid']
        start_time = row['start_seconds']
        end_time = row['end_seconds']
        
        # Crea il nome del file di output
        output_filename = f"{ytid}.wav"
        output_path = self.output_dir / output_filename
        
        # Salta se il file esiste già
        if output_path.exists():
            logger.debug(f"File già esistente: {output_filename}")
            return True, ytid, None
        
        # URL del video YouTube
        video_url = f"https://www.youtube.com/watch?v={ytid}"
        
        try:
            # Usa yt-dlp con --download-sections per estrarre solo il segmento desiderato
            # Questo metodo NON richiede ffmpeg!
            cmd = [
                'yt-dlp',
                '--format', 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio',
                '--output', str(output_path.with_suffix('.%(ext)s')),
                '--download-sections', f'*{start_time}-{end_time}',
                '--force-keyframes-at-cuts',
                '--no-playlist',
                '--ignore-errors',
                '--quiet',
                '--no-warnings',
                video_url
            ]
            
            # Esegui il comando
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=180
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                
                # Gestisci errori specifici
                if any(phrase in error_msg for phrase in ["Video unavailable", "Private video", "removed"]):
                    logger.debug(f"Video non disponibile: {ytid}")
                    return False, ytid, "Video non disponibile"
                elif "download-sections" in error_msg or "force-keyframes" in error_msg:
                    # Se il metodo con --download-sections fallisce, prova il metodo alternativo
                    logger.debug(f"Fallback per {ytid}")
                    return self._download_with_postprocess(ytid, start_time, end_time, output_path, video_url)
                else:
                    logger.debug(f"Errore scaricando {ytid}: {error_msg}")
                    return False, ytid, error_msg
            
            # Trova il file scaricato (potrebbe avere un'estensione diversa)
            downloaded_file = None
            base_path = output_path.with_suffix('')
            
            # Cerca file con varie estensioni
            for ext in ['.m4a', '.webm', '.opus', '.mp3', '.wav']:
                potential_file = base_path.with_suffix(ext)
                if potential_file.exists():
                    downloaded_file = potential_file
                    break
            
            if not downloaded_file:
                logger.debug(f"File scaricato non trovato per {ytid}")
                return False, ytid, "File scaricato non trovato"
            
            # Se il file non è già in formato WAV, rinominalo con l'estensione corretta
            if downloaded_file.suffix != '.wav':
                final_path = output_path.with_suffix(downloaded_file.suffix)
                if downloaded_file != final_path:
                    downloaded_file.rename(final_path)
                logger.debug(f"Scaricato segmento: {final_path.name}")
            else:
                logger.debug(f"Scaricato segmento: {downloaded_file.name}")
            
            return True, ytid, None
                
        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout scaricando {ytid}")
            return False, ytid, "Timeout"
        except Exception as e:
            logger.error(f"Errore inaspettato scaricando {ytid}: {str(e)}")
            return False, ytid, str(e)
    
    def _download_with_postprocess(self, ytid, start_time, end_time, output_path, video_url):
        """
        Metodo di fallback che usa yt-dlp con post-processing
        """
        try:
            # Calcola la durata
            duration = end_time - start_time
            
            cmd = [
                'yt-dlp',
                '--format', 'bestaudio',
                '--output', str(output_path.with_suffix('.%(ext)s')),
                '--postprocessor-args', f'-ss {start_time} -t {duration}',
                '--no-playlist',
                '--ignore-errors',
                '--quiet',
                '--no-warnings',
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                return False, ytid, "Fallback fallito"
            
            # Trova il file scaricato
            base_path = output_path.with_suffix('')
            for ext in ['.m4a', '.webm', '.opus', '.mp3']:
                potential_file = base_path.with_suffix(ext)
                if potential_file.exists():
                    logger.debug(f"Scaricato con fallback: {potential_file.name}")
                    return True, ytid, None
            
            return False, ytid, "File fallback non trovato"
            
        except Exception as e:
            return False, ytid, f"Errore fallback: {str(e)}"
    
    def download_all(self, limit=None):
        """
        Scarica tutti i segmenti audio
        
        Args:
            limit (int): Limite sul numero di segmenti da scaricare (per test)
        """
        df = self.load_csv()
        
        if limit:
            df = df.head(limit)
            logger.info(f"Limitando il download a {limit} segmenti")
        
        logger.info(f"Inizio download di {len(df)} segmenti con {self.max_workers} thread")
        
        successful_downloads = 0
        failed_downloads = 0
        unavailable_videos = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Sottometti tutti i task
            futures = {executor.submit(self.download_segment, row): row 
                      for _, row in df.iterrows()}
            
            # Processa i risultati man mano che completano
            for future in as_completed(futures):
                success, ytid, error = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                    if error and "non disponibile" in error:
                        unavailable_videos += 1
                
                # Stampa progresso ogni 50 download
                total_processed = successful_downloads + failed_downloads
                if total_processed % 50 == 0:
                    logger.info(f"Progresso: {total_processed}/{len(df)} - "
                              f"Successi: {successful_downloads}, "
                              f"Fallimenti: {failed_downloads} "
                              f"(di cui {unavailable_videos} video non disponibili)")
        
        logger.info(f"Download completato: {successful_downloads} successi, "
                   f"{failed_downloads} fallimenti ({unavailable_videos} video non disponibili)")
        
        return successful_downloads, failed_downloads
    
    def get_downloaded_files(self):
        """Restituisce la lista dei file audio scaricati"""
        audio_extensions = ['*.wav', '*.m4a', '*.webm', '*.opus', '*.mp3']
        files = []
        for ext in audio_extensions:
            files.extend(self.output_dir.glob(ext))
        return files


def main():
    parser = argparse.ArgumentParser(description='Scarica segmenti audio da AudioSet')
    parser.add_argument('--csv_path', 
                       default='/home/ing2025edocap/snap/snapd-desktop-integration/253/Scrivania/DeepLearning/data/AudioSet/balanced_train_segments.csv', 
                       help='Percorso al file CSV di AudioSet')
    parser.add_argument('--output-dir', 
                       default='/home/ing2025edocap/snap/snapd-desktop-integration/253/Scrivania/DeepLearning/data/AudioSet/balanced_train', 
                       help='Directory di output (default: balanced_train)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Numero di thread per download parallelo (default: 4)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limita il numero di download (per test)')
    
    args = parser.parse_args()
    
    # Verifica che yt-dlp supporti --download-sections
    try:
        result = subprocess.run(['yt-dlp', '--help'], capture_output=True, text=True)
        if '--download-sections' not in result.stdout:
            logger.warning("La tua versione di yt-dlp potrebbe non supportare --download-sections")
            logger.warning("Aggiorna con: pip install --upgrade yt-dlp")
    except:
        pass
    
    # Crea il downloader
    downloader = AudioSetDownloader(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Avvia il download
    start_time = time.time()
    successful, failed = downloader.download_all(limit=args.limit)
    end_time = time.time()
    
    logger.info(f"Tempo totale: {end_time - start_time:.2f} secondi")
    logger.info(f"File scaricati: {len(downloader.get_downloaded_files())}")


if __name__ == "__main__":
    main()