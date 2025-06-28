import os
import tarfile
import urllib.request
from pathlib import Path
import random
from typing import Optional, Callable, Tuple, List
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd


class AudioToSpectrogram:
    """
    Transform per convertire audio raw in mel-spectrogram per MAE-AST.
    """
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        to_db: bool = True,
        normalize: bool = True
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.to_db = to_db
        self.normalize = normalize
        
        # Mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=0.0,
            f_max=sample_rate // 2,
            power=2.0
        )
        
        # DB conversion
        if to_db:
            self.db_transform = T.AmplitudeToDB()
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel-spectrogram.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [time] -> [1, time]
        elif waveform.dim() == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0) # [1, C, T] -> [C, T]
        
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True) 
            
        mel_spec = self.mel_transform(waveform)
        
        if self.to_db:
            mel_spec = self.db_transform(mel_spec)
        
        mel_spec = mel_spec.squeeze(0) 
        
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec


class LibriSpeechAudioDataset(Dataset):
    """
    LibriSpeech dataset per il pre-training.
    """
    
    MIRRORS = {
        'train-clean-100': "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
        'test-clean': "https://openslr.elda.org/resources/12/test-clean.tar.gz",
    }
    
    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Optional[Callable] = None,
        subset: str = 'train-clean-100',
    ):
        self.root = Path(root)
        self.transform = transform
        self.subset = subset
        self.dataset_name = "LibriSpeech"
        self.subset_name = subset  
        self.max_retries = 5

        self.root.mkdir(parents=True, exist_ok=True)
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.root / self.subset_name}. "
                "You can use download=True to download it"
            )
        
        self.data = self._load_all_audio_paths()
        
        print(f"Caricati {len(self.data)} campioni da {self.dataset_name} ({self.subset_name})")
    
    def _check_exists(self) -> bool:
        """Verifica se il dataset esiste nella directory root."""
        subset_dir = self.root / self.subset_name
        return subset_dir.exists() and len(list(subset_dir.rglob("*.flac"))) > 0
    
    def download(self):
        """Scarica ed estrae il dataset."""
        if self._check_exists():
            print(f"Dataset {self.subset_name} già esistente, salto il download.")
            return
        
        url = self.MIRRORS[self.subset_name]
        filename = url.split('/')[-1]
        filepath = self.root / filename
        
        print(f"Scaricando {self.subset_name} da {url}")
        
        def progress_hook(block_num, block_size, total_size):
            if not hasattr(progress_hook, 'pbar'):
                progress_hook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
            progress_hook.pbar.update(block_size)
        
        try:
            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.close()
        except Exception as e:
            if filepath.exists():
                filepath.unlink()
            raise RuntimeError(f"Download del dataset fallito: {e}")
        
        print(f"Estrazione di {filename}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.root)
        except Exception as e:
            raise RuntimeError(f"Estrazione del dataset fallita: {e}")
        finally:
            if filepath.exists():
                filepath.unlink()
        
        print(f"Dataset {self.subset_name} scaricato ed estratto con successo!")
    
    def _load_all_audio_paths(self) -> List[Tuple[str, int]]:
        """Carica tutti i percorsi dei file audio ed estrae lo speaker_id."""
        subset_dir = self.root / self.subset_name # Percorso corretto
        data = []
        
        audio_files = list(subset_dir.rglob("*.flac"))
        
        for audio_path in tqdm(audio_files, desc=f"Verifica e caricamento dati {self.dataset_name}"):
            try:
                torchaudio.load(audio_path, frame_offset=0, num_frames=1000) 
            except Exception as e:
                continue # Skip this file

            try:
                speaker_id = int(audio_path.parent.parent.name)
            except ValueError:
                file_name_parts = audio_path.name.split('-')
                if len(file_name_parts) > 0:
                    try:
                        speaker_id = int(file_name_parts[0])
                    except ValueError:
                        speaker_id = -1 # Indica speaker_id sconosciuto
                else:
                    speaker_id = -1 # Indica speaker_id sconosciuto

            data.append((str(audio_path), speaker_id))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        retries = 0
        while retries < self.max_retries:
            try:
                audio_path, speaker_id = self.data[idx]
                
                waveform, sample_rate = torchaudio.load(audio_path)
                
                if self.transform is not None:
                    waveform = self.transform(waveform)
                
                result = {
                    'waveform': waveform,
                    'speaker_id': speaker_id,
                    'path': audio_path,
                    'dataset': self.dataset_name
                }
                
                return result
            except Exception as e:
                print(f"AVVISO: Errore nel caricamento di {self.data[idx][0]} in __getitem__. Tentativo {retries + 1}/{self.max_retries}. Errore: {e}")
                retries += 1
                idx = random.randint(0, len(self.data) - 1) 
        
        # Se tutti i tentativi falliscono, solleva un errore
        raise RuntimeError(f"Impossibile caricare un campione dopo {self.max_retries} tentativi per il dataset {self.dataset_name}. Considera di pulire il dataset o aumentare max_retries.")


class AudioSetAudioDataset(Dataset):
    """
    AudioSet dataset per il pre-training.
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.dataset_name = "AudioSet"
        self.max_retries = 5 

        if not self.root.exists():
            raise RuntimeError(f"Directory root di AudioSet non trovata a {self.root}")
            
        self.data = self._load_all_audio_paths()
        
        print(f"Caricati {len(self.data)} campioni da {self.dataset_name}")

    def _load_all_audio_paths(self) -> List[Tuple[str, str]]:
        """Scansiona la directory per trovare tutti i file audio .flac."""
        data = []
        audio_files = list(self.root.rglob("*.flac"))
        
        for audio_path in tqdm(audio_files, desc=f"Verifica e caricamento dati {self.dataset_name}"):
            try:
                torchaudio.load(audio_path, frame_offset=0, num_frames=1000)
            except Exception as e:
                continue # Skip this file

            pseudo_speaker_id = audio_path.name 
            data.append((str(audio_path), pseudo_speaker_id))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        retries = 0
        while retries < self.max_retries:
            try:
                audio_path, pseudo_speaker_id = self.data[idx]
                
                waveform, sample_rate = torchaudio.load(audio_path)
                
                if sample_rate != 16000:
                    resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)

                if self.transform:
                    waveform = self.transform(waveform)
                
                result = {
                    'waveform': waveform,
                    'speaker_id': pseudo_speaker_id, 
                    'path': audio_path,
                    'dataset': self.dataset_name
                }
                
                return result
            except Exception as e:
                retries += 1
                idx = random.randint(0, len(self.data) - 1) # Prova un indice casuale diverso
        
        raise RuntimeError(f"Impossibile caricare un campione dopo {self.max_retries} tentativi per il dataset {self.dataset_name}. Considera di pulire il dataset o aumentare max_retries.")


class PretrainingDataset(Dataset):
    """
    Dataset combinato per il pre-training, che mischia campioni da LibriSpeech e AudioSet
    in percentuali specificate.
    """
    def __init__(
        self,
        librispeech_dataset: LibriSpeechAudioDataset,
        audioset_dataset: AudioSetAudioDataset, 
        librispeech_percentage: float = 0.5,
        audioset_percentage: float = 0.5, 
        seed: int = 42 
    ):
        if not (0 <= librispeech_percentage <= 1 and 0 <= audioset_percentage <= 1):
            raise ValueError("Le percentuali devono essere tra 0 e 1.")
        if librispeech_percentage + audioset_percentage == 0:
            raise ValueError("Almeno una percentuale deve essere maggiore di zero.")
        
        random.seed(seed)
        
        combined_datasets = []

        if librispeech_dataset and librispeech_percentage > 0:
            num_librispeech_samples = int(len(librispeech_dataset) * librispeech_percentage)
            indices = random.sample(range(len(librispeech_dataset)), num_librispeech_samples)
            subset_librispeech = Subset(librispeech_dataset, indices)
            combined_datasets.append(subset_librispeech)
            print(f"Selezionati {len(subset_librispeech)} campioni da LibriSpeech.")
        else:
            print("Saltando il dataset LibriSpeech.")

        if audioset_dataset and audioset_percentage > 0:
            num_audioset_samples = int(len(audioset_dataset) * audioset_percentage)
            indices = random.sample(range(len(audioset_dataset)), num_audioset_samples)
            subset_audioset = Subset(audioset_dataset, indices)
            combined_datasets.append(subset_audioset)
            print(f"Selezionati {len(subset_audioset)} campioni da AudioSet.")
        else:
            print("Saltando il dataset AudioSet.")
        
        if not combined_datasets:
            raise RuntimeError("Nessun dataset è stato selezionato per la combinazione.")

        self.combined_dataset = ConcatDataset(combined_datasets)
        print(f"Totale campioni nel dataset combinato di pre-training: {len(self.combined_dataset)}")

    def __len__(self) -> int:
        return len(self.combined_dataset)

    def __getitem__(self, idx: int):
        return self.combined_dataset[idx]


def collate_fn_spectrogram(batch, target_time_frames=1024): 
    """
    Padda/croppa alla dimensione temporale desiderata e prepara il batch.
    """
    spectrograms = [item['waveform'] for item in batch] 
    speaker_ids = [item['speaker_id'] for item in batch]
    paths = [item['path'] for item in batch]
    datasets = [item['dataset'] for item in batch]
    
    processed_specs = []
    
    for spec in spectrograms:
        if spec.dim() == 3:
            spec = spec.squeeze(0)
            
        current_time = spec.shape[1]
        
        if current_time > target_time_frames:
            # Cropping casuale
            start_idx = torch.randint(0, current_time - target_time_frames + 1, (1,)).item()
            spec = spec[:, start_idx:start_idx + target_time_frames]
        elif current_time < target_time_frames:
            # Padding con zeri
            padding = target_time_frames - current_time
            spec = torch.nn.functional.pad(spec, (0, padding))
        
        processed_specs.append(spec)
    
    stacked_specs = torch.stack(processed_specs)
    stacked_specs = stacked_specs.unsqueeze(1)
    
    result = {
        'waveform': stacked_specs, 
        'speaker_id': speaker_ids,
        'path': paths,
        'dataset': datasets
    }
    
    return result
