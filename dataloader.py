import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class AudioToSpectrogram:
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
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dim
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)   # Remove batch dim if present
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB if requested
        if self.to_db:
            mel_spec = self.db_transform(mel_spec)
        
        # Remove channel dimension if mono
        if mel_spec.shape[0] == 1:
            mel_spec = mel_spec.squeeze(0)
        
        # Normalize if requested
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec


class LibriSpeechGender(Dataset):
    
    MIRRORS = {
        'train-clean-100': "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
        'test-clean': "https://openslr.elda.org/resources/12/test-clean.tar.gz",
    }
    
    def __init__(
        self,
        root: str,
        speakers_file: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        subset: str = 'train-clean-100',
        return_transcription: bool = False
    ):
        self.root = Path(root)
        self.speakers_file = speakers_file
        self.train = train
        self.transform = transform
        self.subset = subset
        self.return_transcription = return_transcription
        self.dataset_name = "LibriSpeech"
        
        self.subset_name = subset
        
        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download if requested
        if download:
            self.download()
        
        # Check if dataset exists
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.root / 'LibriSpeech' / self.subset_name}. "
                "You can use download=True to download it"
            )
        
        # Load speaker gender information
        self.speakers_info = self._load_speakers_info()
        self.speaker_to_gender = dict(zip(
            self.speakers_info['ID'], 
            self.speakers_info['SEX']
        ))
        
        # Create label encoder (F=0, M=1)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['F', 'M'])
        
        # Load audio files with gender labels
        self.data = self._load_data_with_gender()
        
        print(f"Loaded {len(self.data)} samples from {self.dataset_name} ({self.subset_name})")
        
        # Print gender distribution
        gender_counts = pd.Series([d[2] for d in self.data]).value_counts().to_dict()
        print(f"Gender distribution for {self.dataset_name}: {gender_counts}")
    
    def _check_exists(self) -> bool:
        subset_dir = self.root / "LibriSpeech" / self.subset_name
        return subset_dir.exists() and len(list(subset_dir.rglob("*.flac"))) > 0
    
    def download(self):
        if self._check_exists():
            print(f"Dataset {self.subset_name} already exists, skipping download.")
            return
        
        url = self.MIRRORS[self.subset_name]
        filename = url.split('/')[-1]
        filepath = self.root / filename
        
        print(f"Downloading {self.subset_name} from {url}")
        
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
            raise RuntimeError(f"Failed to download dataset: {e}")
        
        print(f"Extracting {filename}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.root)
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")
        finally:
            if filepath.exists():
                filepath.unlink()
        
        print(f"Dataset {self.subset_name} downloaded and extracted successfully!")
    
    def _load_speakers_info(self) -> pd.DataFrame:
        df = pd.read_csv(self.speakers_file, sep='|', comment=';', header=None,
                         names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
        
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Filter for the current subset
        df_filtered = df[df['SUBSET'] == self.subset_name].copy()
        
        if len(df_filtered) == 0:
            raise ValueError(f"No speakers found for subset {self.subset_name} in {self.speakers_file}")
        
        return df_filtered
    
    def _load_data_with_gender(self) -> List[Tuple[str, str, str, int]]:
        subset_dir = self.root / "LibriSpeech" / self.subset_name
        data = []
        
        trans_files = list(subset_dir.rglob("*.trans.txt"))
        
        for trans_file in tqdm(trans_files, desc=f"Loading {self.dataset_name} data"):
            chapter_dir = trans_file.parent
            
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            file_id, transcription = parts
                            speaker_id = int(file_id.split('-')[0])
                            
                            if speaker_id in self.speaker_to_gender:
                                audio_path = chapter_dir / f"{file_id}.flac"
                                if audio_path.exists():
                                    gender = self.speaker_to_gender[speaker_id]
                                    data.append((str(audio_path), transcription, gender, speaker_id))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        audio_path, transcription, gender, speaker_id = self.data[idx]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        gender_label = self.label_encoder.transform([gender])[0]
        
        result = {
            'waveform': waveform,
            'label': torch.tensor(gender_label, dtype=torch.long),
            'speaker_id': speaker_id,
            'path': audio_path,
            'dataset': self.dataset_name
        }
        
        if self.return_transcription:
            result['transcription'] = transcription
        
        return result
    
    def get_label_encoder(self):
        return self.label_encoder

class VoxCelebGender(Dataset):
    """
    VoxCeleb1 dataset for gender classification.
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
            raise RuntimeError(f"VoxCeleb root directory not found at {self.root}")
        if not self.meta_file.exists():
            raise RuntimeError(f"VoxCeleb meta file not found at {self.meta_file}")
            
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['f', 'm'])

        self.speaker_to_gender = self._load_meta()
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} samples from {self.dataset_name}")
        
        gender_counts = pd.Series([d[1] for d in self.data]).value_counts().to_dict()
        print(f"Gender distribution for {self.dataset_name}: {gender_counts}")

    def _load_meta(self) -> dict:
        """Loads speaker metadata from the csv file."""
        df = pd.read_csv(self.meta_file, sep='\t')
        df.columns = df.columns.str.strip()
        speaker_map = {row['VoxCeleb1 ID']: row['Gender'].lower() for _, row in df.iterrows()}
        return speaker_map

    def _load_data(self) -> List[Tuple[str, str, str]]:
        data = []
        audio_files = list(self.root.rglob("*.wav"))
        
        for audio_path in tqdm(audio_files, desc=f"Loading {self.dataset_name} data"):
            speaker_id = audio_path.parent.parent.name
            
            if speaker_id in self.speaker_to_gender:
                gender = self.speaker_to_gender[speaker_id]
                data.append((str(audio_path), gender, speaker_id))

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

def collate_fn_spectrogram(batch, target_time_frames=1024): # Approx 10.24 seconds
    spectrograms = [item['waveform'] for item in batch] # Actually spectrograms after transform
    labels = torch.stack([item['label'] for item in batch])
    speaker_ids = [item['speaker_id'] for item in batch]
    paths = [item['path'] for item in batch]
    datasets = [item['dataset'] for item in batch]
    
    processed_specs = []
    
    for spec in spectrograms:
        if spec.dim() == 3:
            spec = spec.squeeze(0)
            
        current_time = spec.shape[1]
        
        if current_time > target_time_frames:
            start_idx = torch.randint(0, current_time - target_time_frames + 1, (1,)).item()
            spec = spec[:, start_idx:start_idx + target_time_frames]
        elif current_time < target_time_frames:
            padding = target_time_frames - current_time
            spec = torch.nn.functional.pad(spec, (0, padding))
        
        processed_specs.append(spec)
    
    stacked_specs = torch.stack(processed_specs)
    stacked_specs = stacked_specs.unsqueeze(1)
    
    result = {
        'waveform': stacked_specs, 
        'label': labels,
        'speaker_id': speaker_ids,
        'path': paths,
        'dataset': datasets
    }
    
    return result
