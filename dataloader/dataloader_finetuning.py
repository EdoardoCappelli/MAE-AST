import os
import tarfile
import urllib.request
from pathlib import Path
import random
from typing import Optional, Callable, Tuple, List
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
        
        Args:
            waveform: Tensor of shape [channels, time] or [time]
        
        Returns:
            spectrogram: Tensor of shape [n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dim
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)   
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB if requested
        if self.to_db:
            mel_spec = self.db_transform(mel_spec)
        
        # Remove channel dimension if mono (e.g., [1, n_mels, time] -> [n_mels, time])
        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.squeeze(0)
        
        # Normalize if requested
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec


class VoxCelebGenderDataset(Dataset):
    """
    VoxCeleb1 dataset per la classificazione di genere.
    """
    def __init__(
        self,
        root: str,
        meta_file: str,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.meta_file = Path(meta_file)
        self.transform = transform
        self.dataset_name = "VoxCeleb1"
        
        if not self.root.exists():
            raise RuntimeError(f"Directory root di VoxCeleb non trovata a {self.root}")
        if not self.meta_file.exists():
            raise RuntimeError(f"File meta di VoxCeleb non trovato a {self.meta_file}")
            
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['f', 'm'])

        self.speaker_to_gender = self._load_meta()
        
        # Carica i percorsi dei file audio con i relativi metadati
        self.data = self._load_data()
        
        print(f"Caricati {len(self.data)} campioni da {self.dataset_name}")
        
        gender_counts = pd.Series([d[1] for d in self.data]).value_counts().to_dict()
        print(f"Distribuzione di genere per {self.dataset_name}: {gender_counts}")

    def _load_meta(self) -> dict:
        df = pd.read_csv(self.meta_file, sep='\t')
        df.columns = df.columns.str.strip()
        speaker_map = {row['VoxCeleb1 ID']: row['Gender'].lower() for _, row in df.iterrows()}
        return speaker_map

    def _load_data(self) -> List[Tuple[str, str, str]]:
        data = []
        audio_files = list(self.root.rglob("*.wav"))
        
        for audio_path in tqdm(audio_files, desc=f"Caricamento dati {self.dataset_name}"):
            speaker_id = audio_path.parent.parent.name
            
            if speaker_id in self.speaker_to_gender:
                gender = self.speaker_to_gender[speaker_id]
                data.append((str(audio_path), gender, speaker_id))
            else:
                print(f"Avviso: Genere non trovato per speaker ID {speaker_id} in {audio_path}")
                pass

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        audio_path, gender, speaker_id = self.data[idx]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        gender_label = self.label_encoder.transform([gender])[0]
        
        result = {
            'waveform': waveform, 
            'label': torch.tensor(gender_label, dtype=torch.long), 
            'speaker_id': speaker_id,
            'path': audio_path,
            'dataset': self.dataset_name
        }
        
        return result
    
    def get_label_encoder(self):
        return self.label_encoder

class ESCAudioDataset(Dataset):
    """
    Dataset ESC-50 per la classificazione dei suoni ambientali.
    Permette di selezionare un sottoinsieme casuale di classi.
    """
    def __init__(
        self,
        root: str, 
        meta_file: str, 
        transform: Optional[Callable] = None,
        num_classes: Optional[int] = None
    ):
        self.root = Path(root)
        self.meta_file = Path(meta_file)
        self.transform = transform
        self.dataset_name = "ESC-50"
        self.num_classes = num_classes
        
        if not self.root.exists():
            raise RuntimeError(f"Directory root di ESC-50 non trovata a {self.root}")
        if not self.meta_file.exists():
            raise RuntimeError(f"File meta di ESC-50 non trovato a {self.meta_file}")
            
        # Carica i dati e il label encoder appropriato
        self.data, self.label_encoder = self._load_data()
        
        num_actual_classes = len(self.label_encoder.classes_)
        print(f"Caricati {len(self.data)} campioni da {self.dataset_name} usando {num_actual_classes} classi.")
        
        category_counts = pd.Series([d[2] for d in self.data]).value_counts().to_dict()
        print(f"Distribuzione delle categorie per {self.dataset_name} (prime 10): {dict(list(category_counts.items())[:10])}")

    def _load_data(self) -> Tuple[List[Tuple[str, int, str]], LabelEncoder]:
        """
        Carica i percorsi dei file audio e le relative label.
        Filtra le classi se num_classes è specificato.
        """
        df = pd.read_csv(self.meta_file)
        
        # --- LOGICA DI SELEZIONE CLASSI ---
        if self.num_classes is not None:
            if not 1 <= self.num_classes <= 50:
                raise ValueError("Il numero di classi deve essere compreso tra 1 e 50.")
            
            unique_categories = df['category'].unique()
            random.shuffle(unique_categories)
            
            selected_categories = unique_categories[:self.num_classes]
            print(f"Selezionate casualmente {self.num_classes} classi: {selected_categories.tolist()}")
            
            df = df[df['category'].isin(selected_categories)].reset_index(drop=True)

        label_encoder = LabelEncoder()
        df['target'] = label_encoder.fit_transform(df['category'])
        
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Caricamento dati {self.dataset_name}"):
            filename = row['filename']
            target_label = row['target']    
            category_name = row['category']  
            
            audio_path = self.root / filename
            
            if audio_path.exists():
                data.append((str(audio_path), target_label, category_name))
            else:
                print(f"Avviso: File audio non trovato: {audio_path}. Ignorato.")

        return data, label_encoder

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        audio_path, target_label, category_name = self.data[idx]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample a 16kHz se necessario, come richiesto da molti modelli pre-addestrati
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)
        
        result = {
            'waveform': waveform, 
            'label': torch.tensor(target_label, dtype=torch.long), 
            'category_name': category_name,
            'path': audio_path,
            'dataset': self.dataset_name
        }
        
        return result

    def get_label_encoder(self):
        """Restituisce il LabelEncoder utilizzato, utile per decodificare le predizioni."""
        return self.label_encoder


def collate_fn_spectrogram(batch, target_time_frames=1024): 
    """
    Collate function per spettrogrammi per il fine-tuning.
    Padda/croppa alla dimensione temporale desiderata e prepara il batch.
    """
    spectrograms = [item['waveform'] for item in batch] 
    labels = torch.stack([item['label'] for item in batch]) 
    
    speaker_ids = [item.get('speaker_id', None) for item in batch] 
    category_names = [item.get('category_name', None) for item in batch] 
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
        'label': labels,          
        'speaker_id': speaker_ids, 
        'category_name': category_names, 
        'path': paths,
        'dataset': datasets
    }
    
    return result
