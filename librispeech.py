import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class AudioToSpectrogram:
    """
    Transform per convertire audio raw in mel-spectrogram per MAE-AST.
    Compatible con il formato richiesto dal paper (128 x time_frames).
    """
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        n_fft: int = 512,  # Aumentato per avere più frequenze
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


class LibriSpeech(Dataset):
    """
    LibriSpeech dataset for speech recognition tasks.
    Compatible with torchvision.datasets style.
    
    Args:
        root (str): Root directory where the dataset will be stored
        train (bool): If True, use training set, otherwise use test set
        download (bool): If True, download the dataset if not present
        transform (callable, optional): Optional transform to be applied on audio
        target_transform (callable, optional): Optional transform to be applied on transcription
        subset (str): Which subset to use ('clean-100', 'clean-360', 'other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other')
    """
    
    # URL mirrors for different subsets
    MIRRORS = {
        'train-clean-100': "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
        'train-clean-360': "https://openslr.elda.org/resources/12/train-clean-360.tar.gz", 
        'train-other-500': "https://openslr.elda.org/resources/12/train-other-500.tar.gz",
        'dev-clean': "https://openslr.elda.org/resources/12/dev-clean.tar.gz",
        'dev-other': "https://openslr.elda.org/resources/12/dev-other.tar.gz",
        'test-clean': "https://openslr.elda.org/resources/12/test-clean.tar.gz",
        'test-other': "https://openslr.elda.org/resources/12/test-other.tar.gz"
    }
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subset: str = 'clean-100'
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset
        
        # Determine which subset to use
        if train:
            if subset == 'clean-100':
                self.subset_name = 'train-clean-100'
            elif subset == 'clean-360':
                self.subset_name = 'train-clean-360'
            elif subset == 'other-500':
                self.subset_name = 'train-other-500'
            elif subset == 'clean-100-small':
                self.subset_name = 'train-clean-100-small'
            else:
                raise ValueError(f"Invalid train subset: {subset}")
        else:
            if subset == 'clean':
                self.subset_name = 'test-clean'
            elif subset == 'other':
                self.subset_name = 'test-other'
            elif subset == 'dev-clean':
                self.subset_name = 'dev-clean'
            elif subset == 'dev-other':
                self.subset_name = 'dev-other'
            else:
                self.subset_name = f'test-{subset}'
        
        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download if requested
        if download:
            self.download()
        
        # Check if dataset exists
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.root}. "
                "You can use download=True to download it"
            )
        
        # Load file paths and transcriptions
        self.data = self._load_data()
    
    def _check_exists(self) -> bool:
        """Check if the dataset exists in the root directory."""
        subset_dir = self.root / "LibriSpeech" / self.subset_name
        return subset_dir.exists() and len(list(subset_dir.rglob("*.flac"))) > 0
    
    def download(self):
        """Download and extract the dataset."""
        if self._check_exists():
            print(f"Dataset {self.subset_name} already exists, skipping download.")
            return
        
        url = self.MIRRORS[self.subset_name]
        filename = url.split('/')[-1]
        filepath = self.root / filename
        
        print(f"Downloading {self.subset_name} from {url}")
        
        # Download with progress bar
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
        
        # Extract tar.gz file
        print(f"Extracting {filename}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.root)
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")
        finally:
            # Clean up downloaded archive
            if filepath.exists():
                filepath.unlink()
        
        print(f"Dataset {self.subset_name} downloaded and extracted successfully!")
    
    def _load_data(self) -> List[Tuple[str, str]]:
        """Load file paths and transcriptions."""
        subset_dir = self.root / "LibriSpeech" / self.subset_name
        data = []
        
        # Find all .trans.txt files
        trans_files = list(subset_dir.rglob("*.trans.txt"))
        
        for trans_file in trans_files:
            chapter_dir = trans_file.parent
            
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            file_id = parts[0]
                            transcription = parts[1]
                            audio_path = chapter_dir / f"{file_id}.flac"
                            
                            if audio_path.exists():
                                data.append((str(audio_path), transcription))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get item by index.
        
        Returns:
            tuple: (audio_tensor, transcription) where audio_tensor is a torch.Tensor
                   and transcription is a string
        """
        audio_path, transcription = self.data[idx]
        
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Apply transforms if provided
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        if self.target_transform is not None:
            transcription = self.target_transform(transcription)
        
        return waveform, transcription
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the audio files (LibriSpeech uses 16kHz)."""
        return 16000
    
    def get_subset_info(self) -> dict:
        """Get information about the current subset."""
        return {
            'subset_name': self.subset_name,
            'num_samples': len(self.data),
            'sample_rate': self.get_sample_rate(),
            'is_train': self.train
        }


class LibriSpeechGender(Dataset):
    """
    LibriSpeech dataset esteso per classificazione di genere.
    Carica audio e associa le label di genere dal file SPEAKERS.txt.
    
    Args:
        root (str): Root directory del dataset LibriSpeech
        speakers_file (str): Path al file SPEAKERS.txt
        train (bool): Se True usa training set, altrimenti test set
        download (bool): Se True scarica il dataset se non presente
        transform (callable, optional): Transform da applicare all'audio
        subset (str): Subset da usare ('clean-100', 'clean-360', etc.)
        return_transcription (bool): Se True restituisce anche la trascrizione
    """
    
    MIRRORS = {
        'train-clean-100': "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
        'train-clean-360': "https://openslr.elda.org/resources/12/train-clean-360.tar.gz", 
        'train-other-500': "https://openslr.elda.org/resources/12/train-other-500.tar.gz",
        'dev-clean': "https://openslr.elda.org/resources/12/dev-clean.tar.gz",
        'dev-other': "https://openslr.elda.org/resources/12/dev-other.tar.gz",
        'test-clean': "https://openslr.elda.org/resources/12/test-clean.tar.gz",
        'test-other': "https://openslr.elda.org/resources/12/test-other.tar.gz"
    }
    
    def __init__(
        self,
        root: str,
        speakers_file: str,
        train: bool = True,
        download: bool = False,
        transform: Optional[Callable] = None,
        subset: str = 'clean-100',
        return_transcription: bool = False
    ):
        self.root = Path(root)
        self.speakers_file = speakers_file
        self.train = train
        self.transform = transform
        self.subset = subset
        self.return_transcription = return_transcription
        
        # Determine subset name
        if train:
            if subset == 'clean-100':
                self.subset_name = 'train-clean-100'
            elif subset == 'clean-360':
                self.subset_name = 'train-clean-360'
            elif subset == 'other-500':
                self.subset_name = 'train-other-500'
            else:
                raise ValueError(f"Invalid train subset: {subset}")
        else:
            if subset == 'clean':
                self.subset_name = 'test-clean'
            elif subset == 'other':
                self.subset_name = 'test-other'
            elif subset == 'dev-clean':
                self.subset_name = 'dev-clean'
            elif subset == 'dev-other':
                self.subset_name = 'dev-other'
            else:
                self.subset_name = f'test-{subset}'
        
        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download if requested
        if download:
            self.download()
        
        # Check if dataset exists
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found at {self.root}. "
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
        
        print(f"Loaded {len(self.data)} samples from {self.subset_name}")
        
        # Print gender distribution
        gender_counts = {}
        for _, _, gender, _ in self.data:
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        print(f"Gender distribution: {gender_counts}")
    
    def _check_exists(self) -> bool:
        """Check if the dataset exists in the root directory."""
        subset_dir = self.root / "LibriSpeech" / self.subset_name
        return subset_dir.exists() and len(list(subset_dir.rglob("*.flac"))) > 0
    
    def download(self):
        """Download and extract the dataset."""
        if self._check_exists():
            print(f"Dataset {self.subset_name} already exists, skipping download.")
            return
        
        url = self.MIRRORS[self.subset_name]
        filename = url.split('/')[-1]
        filepath = self.root / filename
        
        print(f"Downloading {self.subset_name} from {url}")
        
        # Download with progress bar
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
        
        # Extract tar.gz file
        print(f"Extracting {filename}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.root)
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")
        finally:
            # Clean up downloaded archive
            if filepath.exists():
                filepath.unlink()
        
        print(f"Dataset {self.subset_name} downloaded and extracted successfully!")
    
    def _load_speakers_info(self) -> pd.DataFrame:
        """Load and filter speaker information from SPEAKERS.txt"""
        # Read the file skipping the first row (header with ;)
        df = pd.read_csv(self.speakers_file, sep='|', skiprows=1, 
                        names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
        
        # Clean whitespace
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Filter for the current subset
        df_filtered = df[df['SUBSET'] == self.subset_name].copy()
        
        if len(df_filtered) == 0:
            raise ValueError(f"No speakers found for subset {self.subset_name} in {self.speakers_file}")
        
        print(f"Found {len(df_filtered)} speakers in {self.subset_name}")
        print(f"Gender distribution: {df_filtered['SEX'].value_counts().to_dict()}")
        
        return df_filtered
    
    def _load_data_with_gender(self) -> List[Tuple[str, str, str, int]]:
        """Load audio files with associated gender labels"""
        subset_dir = self.root / "LibriSpeech" / self.subset_name
        data = []
        
        # Find all .trans.txt files
        trans_files = list(subset_dir.rglob("*.trans.txt"))
        
        for trans_file in trans_files:
            chapter_dir = trans_file.parent
            
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) >= 2:
                            file_id = parts[0]
                            transcription = parts[1]
                            
                            # Extract speaker ID from file_id (format: speaker_id-chapter_id-utterance_id)
                            speaker_id = int(file_id.split('-')[0])
                            
                            # Check if we have gender info for this speaker
                            if speaker_id in self.speaker_to_gender:
                                audio_path = chapter_dir / f"{file_id}.flac"
                                
                                if audio_path.exists():
                                    gender = self.speaker_to_gender[speaker_id]
                                    data.append((str(audio_path), transcription, gender, speaker_id))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        """
        Get item by index.
        
        Returns:
            dict: {
                'waveform': torch.Tensor,
                'label': torch.Tensor (gender label),
                'speaker_id': int,
                'transcription': str (if return_transcription=True),
                'path': str
            }
        """
        audio_path, transcription, gender, speaker_id = self.data[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Apply transform if provided
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        # Convert gender to numerical label
        gender_label = self.label_encoder.transform([gender])[0]
        
        result = {
            'waveform': waveform,
            'label': torch.tensor(gender_label, dtype=torch.long),
            'speaker_id': speaker_id,
            'path': audio_path
        }
        
        if self.return_transcription:
            result['transcription'] = transcription
        
        return result
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the audio files (LibriSpeech uses 16kHz)."""
        return 16000
    
    def get_gender_distribution(self) -> dict:
        """Get gender distribution in the dataset"""
        gender_counts = {}
        for _, _, gender, _ in self.data:
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        return gender_counts
    
    def get_label_encoder(self):
        """Get the label encoder for gender labels"""
        return self.label_encoder


def collate_fn_pad(batch):
    """
    Collate function per gestire audio di lunghezze diverse.
    Padding degli audio alla lunghezza massima del batch.
    """
    if isinstance(batch[0], dict):
        # For LibriSpeechGender dataset
        waveforms = [item['waveform'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        speaker_ids = [item['speaker_id'] for item in batch]
        paths = [item['path'] for item in batch]
        
        # Handle transcriptions if present
        transcriptions = None
        if 'transcription' in batch[0]:
            transcriptions = [item['transcription'] for item in batch]
        
        # Process waveforms
        waveforms = [w.squeeze(0) if w.dim() > 1 else w for w in waveforms]
        lengths = torch.tensor([w.shape[-1] for w in waveforms])
        padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        
        if padded_waveforms.dim() == 2:
            padded_waveforms = padded_waveforms.unsqueeze(1)
        
        result = {
            'waveform': padded_waveforms,
            'label': labels,
            'speaker_id': speaker_ids,
            'path': paths,
            'lengths': lengths
        }
        
        if transcriptions is not None:
            result['transcription'] = transcriptions
        
        return result
    else:
        # For original LibriSpeech dataset
        waveforms, transcriptions = zip(*batch)
        waveforms = [w.squeeze(0) if w.dim() > 1 else w for w in waveforms]
        lengths = torch.tensor([w.shape[-1] for w in waveforms])
        padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        
        if padded_waveforms.dim() == 2:
            padded_waveforms = padded_waveforms.unsqueeze(1)
        
        return padded_waveforms, list(transcriptions), lengths


def collate_fn_crop(batch, max_length=160000):  # 10 secondi a 16kHz
    """
    Collate function che taglia o padda gli audio a una lunghezza fissa.
    Utile per MAE dove vuoi dimensioni consistenti.
    """
    if isinstance(batch[0], dict):
        # For LibriSpeechGender dataset
        waveforms = [item['waveform'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        speaker_ids = [item['speaker_id'] for item in batch]
        paths = [item['path'] for item in batch]
        
        transcriptions = None
        if 'transcription' in batch[0]:
            transcriptions = [item['transcription'] for item in batch]
        
        processed_waveforms = []
        
        for w in waveforms:
            if w.dim() > 1:
                w = w.squeeze(0)
            
            if w.shape[-1] > max_length:
                start_idx = torch.randint(0, w.shape[-1] - max_length + 1, (1,))
                w = w[start_idx:start_idx + max_length]
            elif w.shape[-1] < max_length:
                padding = max_length - w.shape[-1]
                w = torch.nn.functional.pad(w, (0, padding))
            
            processed_waveforms.append(w)
        
        stacked_waveforms = torch.stack(processed_waveforms)
        
        if stacked_waveforms.dim() == 2:
            stacked_waveforms = stacked_waveforms.unsqueeze(1)
        
        result = {
            'waveform': stacked_waveforms,
            'label': labels,
            'speaker_id': speaker_ids,
            'path': paths
        }
        
        if transcriptions is not None:
            result['transcription'] = transcriptions
        
        return result
    else:
        # For original LibriSpeech dataset
        waveforms, transcriptions = zip(*batch)
        processed_waveforms = []
        
        for w in waveforms:
            if w.dim() > 1:
                w = w.squeeze(0)
            
            if w.shape[-1] > max_length:
                start_idx = torch.randint(0, w.shape[-1] - max_length + 1, (1,))
                w = w[start_idx:start_idx + max_length]
            elif w.shape[-1] < max_length:
                padding = max_length - w.shape[-1]
                w = torch.nn.functional.pad(w, (0, padding))
            
            processed_waveforms.append(w)
        
        stacked_waveforms = torch.stack(processed_waveforms)
        
        if stacked_waveforms.dim() == 2:
            stacked_waveforms = stacked_waveforms.unsqueeze(1)
        
        return stacked_waveforms, list(transcriptions)


def collate_fn_spectrogram(batch, target_time_frames=None):
    """
    Collate function per spettrogrammi. Padda/croppa alla dimensione temporale desiderata.
    """
    if isinstance(batch[0], dict):
        # For LibriSpeechGender dataset with spectrograms
        spectrograms = [item['waveform'] for item in batch]  # Actually spectrograms after transform
        labels = torch.stack([item['label'] for item in batch])
        speaker_ids = [item['speaker_id'] for item in batch]
        paths = [item['path'] for item in batch]
        
        transcriptions = None
        if 'transcription' in batch[0]:
            transcriptions = [item['transcription'] for item in batch]
        
        processed_specs = []
        
        if target_time_frames is None:
            max_time = max(spec.shape[-1] for spec in spectrograms)
            target_time_frames = max_time
        
        for spec in spectrograms:
            current_time = spec.shape[-1]
            
            if current_time > target_time_frames:
                start_idx = torch.randint(0, current_time - target_time_frames + 1, (1,))
                spec = spec[:, start_idx:start_idx + target_time_frames]
            elif current_time < target_time_frames:
                padding = target_time_frames - current_time
                spec = torch.nn.functional.pad(spec, (0, padding))
            
            processed_specs.append(spec)
        
        stacked_specs = torch.stack(processed_specs)
        stacked_specs = stacked_specs.unsqueeze(1)  # Add channel dimension
        
        result = {
            'waveform': stacked_specs,  # Actually spectrograms
            'label': labels,
            'speaker_id': speaker_ids,
            'path': paths
        }
        
        if transcriptions is not None:
            result['transcription'] = transcriptions
        
        return result
    else:
        # For original LibriSpeech dataset
        spectrograms, transcriptions = zip(*batch)
        processed_specs = []
        
        if target_time_frames is None:
            max_time = max(spec.shape[-1] for spec in spectrograms)
            target_time_frames = max_time
        
        for spec in spectrograms:
            current_time = spec.shape[-1]
            
            if current_time > target_time_frames:
                start_idx = torch.randint(0, current_time - target_time_frames + 1, (1,))
                spec = spec[:, start_idx:start_idx + target_time_frames]
            elif current_time < target_time_frames:
                padding = target_time_frames - current_time
                spec = torch.nn.functional.pad(spec, (0, padding))
            
            processed_specs.append(spec)
        
        stacked_specs = torch.stack(processed_specs)
        stacked_specs = stacked_specs.unsqueeze(1)
        
        return stacked_specs, list(transcriptions)


if __name__ == "__main__":
    root = r"C:\Users\admin\Desktop\VS Code\data"
    speakers_file = r"C:\Users\admin\Desktop\VS Code\data\LibriSpeech\SPEAKERS.TXT" 
    
    # Test 1: Dataset originale
    # print("=== Test Dataset Audio Raw ===")
    # train_dataset = LibriSpeech(
    #     root=root,
    #     train=True,
    #     download=True,
    #     subset='clean-100'
    # )
    
    # print(f"Original dataset loaded with {len(train_dataset)} samples")
    
    # Test 2: Dataset con gender labels
    print("\n=== Test Dataset Gender Classification ===")
    train_dataset_gender = LibriSpeechGender(
        root=root,
        speakers_file=speakers_file,
        train=True,
        download=False,  # Già scaricato
        subset='clean-100',
        return_transcription=True
    )
    
    print(f"Gender dataset loaded with {len(train_dataset_gender)} samples")
    print(f"Gender distribution: {train_dataset_gender.get_gender_distribution()}")
    
    # Test primo sample
    sample = train_dataset_gender[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Waveform shape: {sample['waveform'].shape}")
    print(f"Label: {sample['label']} (speaker_id: {sample['speaker_id']})")
    if 'transcription' in sample:
        print(f"Transcription: {sample['transcription'][:100]}...")
        


# class AudioToSpectrogram:
#     """
#     Transform per convertire audio raw in mel-spectrogram per MAE-AST.
#     Compatible con il formato richiesto dal paper (128 x time_frames).
#     """
#     def __init__(
#         self,
#         n_mels: int = 128,
#         sample_rate: int = 16000,
#         n_fft: int = 512,  # Aumentato per avere più frequenze
#         hop_length: int = 160,
#         to_db: bool = True,
#         normalize: bool = True
#     ):
#         self.n_mels = n_mels
#         self.sample_rate = sample_rate
#         self.to_db = to_db
#         self.normalize = normalize
        
#         # Mel-spectrogram transform
#         self.mel_transform = T.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_mels=n_mels,
#             hop_length=hop_length,
#             n_fft=n_fft,
#             f_min=0.0,
#             f_max=sample_rate // 2,
#             power=2.0
#         )
        
#         # DB conversion
#         if to_db:
#             self.db_transform = T.AmplitudeToDB()
    
#     def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Convert waveform to mel-spectrogram.
        
#         Args:
#             waveform: Tensor of shape [channels, time] or [time]
        
#         Returns:
#             spectrogram: Tensor of shape [n_mels, time_frames]
#         """
#         # Ensure correct shape
#         if waveform.dim() == 1:
#             waveform = waveform.unsqueeze(0)  # Add channel dim
#         elif waveform.dim() == 3:
#             waveform = waveform.squeeze(0)   # Remove batch dim if present
        
#         # Convert to mel-spectrogram
#         mel_spec = self.mel_transform(waveform)
        
#         # Convert to dB if requested
#         if self.to_db:
#             mel_spec = self.db_transform(mel_spec)
        
#         # Remove channel dimension if mono
#         if mel_spec.shape[0] == 1:
#             mel_spec = mel_spec.squeeze(0)
        
#         # Normalize if requested
#         if self.normalize:
#             mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
#         return mel_spec


# class LibriSpeech(Dataset):
#     """
#     LibriSpeech dataset for speech recognition tasks.
#     Compatible with torchvision.datasets style.
    
#     Args:
#         root (str): Root directory where the dataset will be stored
#         train (bool): If True, use training set, otherwise use test set
#         download (bool): If True, download the dataset if not present
#         transform (callable, optional): Optional transform to be applied on audio
#         target_transform (callable, optional): Optional transform to be applied on transcription
#         subset (str): Which subset to use ('clean-100', 'clean-360', 'other-500', 'dev-clean', 'dev-other', 'test-clean', 'test-other')
#     """
    
#     # URL mirrors for different subsets
#     MIRRORS = {
#         'train-clean-100': "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
#         'train-clean-360': "https://openslr.elda.org/resources/12/train-clean-360.tar.gz", 
#         'train-other-500': "https://openslr.elda.org/resources/12/train-other-500.tar.gz",
#         'dev-clean': "https://openslr.elda.org/resources/12/dev-clean.tar.gz",
#         'dev-other': "https://openslr.elda.org/resources/12/dev-other.tar.gz",
#         'test-clean': "https://openslr.elda.org/resources/12/test-clean.tar.gz",
#         'test-other': "https://openslr.elda.org/resources/12/test-other.tar.gz"
#     }
    
#     def __init__(
#         self,
#         root: str,
#         train: bool = True,
#         download: bool = False,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         subset: str = 'clean-100'
#     ):
#         self.root = Path(root)
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.subset = subset
        
#         # Determine which subset to use
#         if train:
#             if subset == 'clean-100':
#                 self.subset_name = 'train-clean-100'
#             elif subset == 'clean-360':
#                 self.subset_name = 'train-clean-360'
#             elif subset == 'other-500':
#                 self.subset_name = 'train-other-500'
#             elif subset == 'clean-100-small':
#                 self.subset_name = 'train-clean-100-small'
#             else:
#                 raise ValueError(f"Invalid train subset: {subset}")
#         else:
#             if subset == 'clean':
#                 self.subset_name = 'test-clean'
#             elif subset == 'other':
#                 self.subset_name = 'test-other'
#             elif subset == 'dev-clean':
#                 self.subset_name = 'dev-clean'
#             elif subset == 'dev-other':
#                 self.subset_name = 'dev-other'
#             else:
#                 self.subset_name = f'test-{subset}'
        
#         # Create root directory
#         self.root.mkdir(parents=True, exist_ok=True)
        
#         # Download if requested
#         if download:
#             self.download()
        
#         # Check if dataset exists
#         if not self._check_exists():
#             raise RuntimeError(
#                 f"Dataset not found at {self.root}. "
#                 "You can use download=True to download it"
#             )
        
#         # Load file paths and transcriptions
#         self.data = self._load_data()
    
#     def _check_exists(self) -> bool:
#         """Check if the dataset exists in the root directory."""
#         subset_dir = self.root / "LibriSpeech" / self.subset_name
#         return subset_dir.exists() and len(list(subset_dir.rglob("*.flac"))) > 0
    
#     def download(self):
#         """Download and extract the dataset."""
#         if self._check_exists():
#             print(f"Dataset {self.subset_name} already exists, skipping download.")
#             return
        
#         url = self.MIRRORS[self.subset_name]
#         filename = url.split('/')[-1]
#         filepath = self.root / filename
        
#         print(f"Downloading {self.subset_name} from {url}")
        
#         # Download with progress bar
#         def progress_hook(block_num, block_size, total_size):
#             if not hasattr(progress_hook, 'pbar'):
#                 progress_hook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
#             progress_hook.pbar.update(block_size)
        
#         try:
#             urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
#             if hasattr(progress_hook, 'pbar'):
#                 progress_hook.pbar.close()
#         except Exception as e:
#             if filepath.exists():
#                 filepath.unlink()
#             raise RuntimeError(f"Failed to download dataset: {e}")
        
#         # Extract tar.gz file
#         print(f"Extracting {filename}...")
#         try:
#             with tarfile.open(filepath, 'r:gz') as tar:
#                 tar.extractall(self.root)
#         except Exception as e:
#             raise RuntimeError(f"Failed to extract dataset: {e}")
#         finally:
#             # Clean up downloaded archive
#             if filepath.exists():
#                 filepath.unlink()
        
#         print(f"Dataset {self.subset_name} downloaded and extracted successfully!")
    
#     def _load_data(self) -> List[Tuple[str, str]]:
#         """Load file paths and transcriptions."""
#         subset_dir = self.root / "LibriSpeech" / self.subset_name
#         data = []
        
#         # Find all .trans.txt files
#         trans_files = list(subset_dir.rglob("*.trans.txt"))
        
#         for trans_file in trans_files:
#             chapter_dir = trans_file.parent
            
#             with open(trans_file, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     line = line.strip()
#                     if line:
#                         parts = line.split(' ', 1)
#                         if len(parts) >= 2:
#                             file_id = parts[0]
#                             transcription = parts[1]
#                             audio_path = chapter_dir / f"{file_id}.flac"
                            
#                             if audio_path.exists():
#                                 data.append((str(audio_path), transcription))
        
#         return data
    
#     def __len__(self) -> int:
#         return len(self.data)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
#         """
#         Get item by index.
        
#         Returns:
#             tuple: (audio_tensor, transcription) where audio_tensor is a torch.Tensor
#                    and transcription is a string
#         """
#         audio_path, transcription = self.data[idx]
        
#         # Load audio using torchaudio
#         waveform, sample_rate = torchaudio.load(audio_path)
        
#         # Apply transforms if provided
#         if self.transform is not None:
#             waveform = self.transform(waveform)
        
#         if self.target_transform is not None:
#             transcription = self.target_transform(transcription)
        
#         return waveform, transcription
    
#     def get_sample_rate(self) -> int:
#         """Get the sample rate of the audio files (LibriSpeech uses 16kHz)."""
#         return 16000
    
#     def get_subset_info(self) -> dict:
#         """Get information about the current subset."""
#         return {
#             'subset_name': self.subset_name,
#             'num_samples': len(self.data),
#             'sample_rate': self.get_sample_rate(),
#             'is_train': self.train
#         }


# def collate_fn_pad(batch):
#     """
#     Collate function per gestire audio di lunghezze diverse.
#     Padding degli audio alla lunghezza massima del batch.
    
#     Args:
#         batch: Lista di tuple (waveform, transcription)
    
#     Returns:
#         tuple: (padded_waveforms, transcriptions, lengths)
#     """
#     waveforms, transcriptions = zip(*batch)
    
#     # Converti in lista e ottieni le lunghezze originali
#     waveforms = [w.squeeze(0) if w.dim() > 1 else w for w in waveforms]  # Rimuovi dim canali se presente
#     lengths = torch.tensor([w.shape[-1] for w in waveforms])
    
#     # Padding alla lunghezza massima
#     padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
#     # Aggiungi dimensione canali se necessario (per compatibilità)
#     if padded_waveforms.dim() == 2:
#         padded_waveforms = padded_waveforms.unsqueeze(1)  # [batch, 1, time]
    
#     return padded_waveforms, list(transcriptions), lengths


# def collate_fn_crop(batch, max_length=160000):  # 10 secondi a 16kHz
#     """
#     Collate function che taglia o padda gli audio a una lunghezza fissa.
#     Utile per MAE dove vuoi dimensioni consistenti.
    
#     Args:
#         batch: Lista di tuple (waveform, transcription)
#         max_length: Lunghezza target per tutti gli audio
    
#     Returns:
#         tuple: (cropped_waveforms, transcriptions)
#     """
#     waveforms, transcriptions = zip(*batch)
#     processed_waveforms = []
    
#     for w in waveforms:
#         # Rimuovi dimensione canali se presente
#         if w.dim() > 1:
#             w = w.squeeze(0)
        
#         # Crop o pad alla lunghezza desiderata
#         if w.shape[-1] > max_length:
#             # Crop random (per data augmentation)
#             start_idx = torch.randint(0, w.shape[-1] - max_length + 1, (1,))
#             w = w[start_idx:start_idx + max_length]
#         elif w.shape[-1] < max_length:
#             # Pad con zeri
#             padding = max_length - w.shape[-1]
#             w = torch.nn.functional.pad(w, (0, padding))
        
#         processed_waveforms.append(w)
    
#     # Stack tutti i tensor (ora hanno tutti la stessa dimensione)
#     stacked_waveforms = torch.stack(processed_waveforms)
    
#     # Aggiungi dimensione canali
#     if stacked_waveforms.dim() == 2:
#         stacked_waveforms = stacked_waveforms.unsqueeze(1)  # [batch, 1, time]
    
#     return stacked_waveforms, list(transcriptions)


# def collate_fn_spectrogram(batch, target_time_frames=None):
#     """
#     Collate function per spettrogrammi. Padda/croppa alla dimensione temporale desiderata.
    
#     Args:
#         batch: Lista di tuple (spectrogram, transcription) 
#         target_time_frames: Numero target di frame temporali (None = padding al max)
    
#     Returns:
#         tuple: (spectrograms, transcriptions)
#     """
#     spectrograms, transcriptions = zip(*batch)
#     processed_specs = []
    
#     if target_time_frames is None:
#         # Padding al massimo nel batch
#         max_time = max(spec.shape[-1] for spec in spectrograms)
#         target_time_frames = max_time
    
#     for spec in spectrograms:
#         # spec shape: [n_mels, time_frames]
#         current_time = spec.shape[-1]
        
#         if current_time > target_time_frames:
#             # Crop random
#             start_idx = torch.randint(0, current_time - target_time_frames + 1, (1,))
#             spec = spec[:, start_idx:start_idx + target_time_frames]
#         elif current_time < target_time_frames:
#             # Pad con zeri
#             padding = target_time_frames - current_time
#             spec = torch.nn.functional.pad(spec, (0, padding))
        
#         processed_specs.append(spec)
    
#     # Stack tutti gli spettrogrammi
#     stacked_specs = torch.stack(processed_specs)  # [batch, n_mels, time_frames]
    
#     # Aggiungi dimensione canali per compatibilità con modelli CNN/Vision
#     stacked_specs = stacked_specs.unsqueeze(1)  # [batch, channels=1, n_mels, time_frames]
    
#     return stacked_specs, list(transcriptions)


# if __name__ == "__main__":
#     root = "./data"
    
#     # Test 1: Dataset normale (audio raw)
#     print("=== Test Dataset Audio Raw ===")
#     train_dataset = LibriSpeech(
#         root=root,
#         train=True,
#         download=True,
#         subset='clean-100'
#     )
    
#     print(f"Dataset loaded with {len(train_dataset)} samples")
    
#     # Test 2: Dataset con trasform per spettrogrammi
#     print("\n=== Test Dataset con Spectrogram Transform ===")
#     spectrogram_transform = AudioToSpectrogram(
#         n_mels=64,
#         sample_rate=16000,
#         hop_length=160,  # ~10ms hop
#         # n_fft=512,       # ~32ms window (più frequenze)
#         to_db=True,
#         normalize=True
#     )
    
#     train_dataset_spec = LibriSpeech(
#         root=root,
#         train=True,
#         download=False,  # Già scaricato
#         subset='clean-100',
#         transform=spectrogram_transform
#     )
    
#     # Test DataLoader con spettrogrammi
#     batch_size = 4
#     train_loader_spec = torch.utils.data.DataLoader(
#         train_dataset_spec,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         collate_fn=lambda batch: collate_fn_spectrogram(batch, target_time_frames=1000)  # 1008 = 63*16 patch perfette
#     )
    
#     # Test primo batch
#     for batch_idx, (spectrograms, transcriptions) in enumerate(train_loader_spec):
#         print(f"Batch {batch_idx}:")
#         print(f"Spectrograms shape: {spectrograms.shape}")  # Dovrebbe essere [batch_size, 1, 128, 1008]
#         print(f"Min/Max values: {spectrograms.min():.3f} / {spectrograms.max():.3f}")
#         print(f"Sample transcription: {transcriptions[0]}")
        
#         # Calcola quante patch 16x16 ottieni
#         batch_size, channels, n_mels, time_frames = spectrograms.shape
#         patches_freq = n_mels // 16
#         patches_time = time_frames // 16
#         total_patches = patches_freq * patches_time
        
#         print(f"Shape: [batch={batch_size}, channels={channels}, freq={n_mels}, time={time_frames}]")
#         print(f"Patches: {patches_freq} x {patches_time} = {total_patches} patches totali")
#         print(f"Perfect patch alignment: freq={n_mels % 16 == 0}, time={time_frames % 16 == 0}")
        
#         # Info sulla durata audio
#         duration_seconds = time_frames * 160 / 16000  # hop_length / sample_rate
#         print(f"Audio duration: {duration_seconds:.1f} seconds")
#         break