import torch
import torch.nn.functional as F
import argparse
import soundfile as sf
import os 
import numpy as np
import torchaudio
import torch
import torchaudio

def convert_wav_to_mel_spectrogram(wav_path, mel_path):
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


    
def main():
    parser = argparse.ArgumentParser(description='Convert .wav to mel spectrogram')
    parser.add_argument('--wav_dir', type=str, help='Path to the directory containing .wav files', default="D:/data/wavs/balanced_train_segments")
    parser.add_argument('--out_dir', type=str, help='Path to save the spectrograms', default="D:/data/spectrograms/balanced_train_segments")
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