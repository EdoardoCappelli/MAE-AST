import torch
import torch.nn.functional as F
from config import Config as config
import matplotlib.pyplot as plt
import argparse
import soundfile as sf
import av
import os 
import numpy as np

def convert_wav_to_mel_spectrogram(wav_path, mel_path):
    import torchaudio

    wav, cur_sample_rate = sf.read(wav_path)
    wav = torch.from_numpy(wav).float()

    if wav.dim() == 2:
        wav = wav.mean(dim=-1)
    assert wav.dim() == 1, "Audio must be mono"

    if cur_sample_rate != 32000:
        wav = torchaudio.functional.resample(wav, cur_sample_rate, 32000)
    
    with torch.no_grad():
        wav = F.layer_norm(wav, wav.shape)

    wav = wav.view(1, -1)

    spectrogram = torchaudio.compliance.kaldi.spectrogram(
        waveform=wav,
        sample_frequency=32000,
    )  

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform=wav,
        sample_frequency=32000,
        use_energy=False,
        num_mel_bins=128
    ) # (time, frequency)

    spectrogram = spectrogram[:, :128]
    fbank = fbank[:, :128]

    if not os.path.exists(mel_path):
        os.makedirs(mel_path)
        print(f"Directory di output '{mel_path}' creata.")

    path = mel_path + "/" + "SPEC_" + wav_path.split("/")[-1].split(".")[0][3:]
    np.save(path, spectrogram.numpy())
    print(f"Saved spectrogram to {mel_path}")

    return spectrogram, fbank

def visualize_fbank(fbank, output_path=None):
    if isinstance(fbank, torch.Tensor):
        fbank = fbank.numpy()

    if fbank.ndim == 1:
        print(f"Nota: Trasformazione di fbank da forma {fbank.shape} a (1, {len(fbank)})")
        fbank = fbank.reshape(1, -1)

    plt.figure(figsize=(12, 8))
    plt.imshow(fbank, aspect='auto', origin='lower')
    plt.title('Fbank (Mel Filterbank Features)')
    plt.xlabel('Time frame')
    plt.ylabel('Mel bin')
    plt.colorbar(format='%+2.0f dB')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if output_path:
        plt.savefig(output_path+"/fbank.png")
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def visualize_spectrogram(spectrogram, output_path=None):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.numpy()
    
    # Se spectrogram o fbank sono 1D, devono essere trasformati in 2D per la visualizzazione
    if spectrogram.ndim == 1:
        print(f"Nota: Trasformazione di spectrogram da forma {spectrogram.shape} a (1, {len(spectrogram)})")
        spectrogram = spectrogram.reshape(1, -1)
    
        
    plt.figure(figsize=(12, 8))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.ylabel('Frequency bin')
    plt.colorbar(format='%+2.0f dB')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if output_path:
        plt.savefig(output_path+"/spectrogram.png")
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Convert .wav to mel spectrogram')
    parser.add_argument('--wav_dir', type=str, help='Path to the directory containing .wav files')
    parser.add_argument('--out_dir', type=str, help='Path to save the spectrograms')
    args = parser.parse_args()

    if not os.path.exists(args.wav_dir):
        raise FileNotFoundError(f"Directory {args.wav_dir} does not exist.")
    
    # cicling through all the files in the directory
    for filename in os.listdir(args.wav_dir):
        if filename.endswith(".wav"):
            wav_path = os.path.join(args.wav_dir, filename)
            mel_path = os.path.join(args.out_dir)
            spectrogram, fbank = convert_wav_to_mel_spectrogram(wav_path, mel_path)

if __name__ == '__main__':
    main()