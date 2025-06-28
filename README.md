# MAE-AST

This project implements **MAE-AST**, a Masked Autoencoder for Audio Spectrogram Transformers. The goal is to replicate the some results highlighted in the Table 1 of the paper https://arxiv.org/pdf/2203.16691. 

-----

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/EdoardoCappelli/MAE-AST.git
    cd MAE-AST
    ```

2.  **Install dependencies:**
   
    ```bash
    pip install -r requirements.txt
    ```

-----

## Dataset Download

Before running the experiments, download the audio datasets. Dedicated scripts are provided for this purpose:

  * **ESC-50:**

    ```bash
    python download/download_esc.py
    ```

  * **AudioSet:**
    To download AudioSet audios, which are based on YouTube, you need to have **FFmpeg** installed on your system. FFmpeg is used to download the videos and extract the 10-second audio segments.

    ```bash
    python download/download_audioset.py
    ```
  * **VoxCeleb1:**
    You can download the VoxCeleb1 dataset directly from Hugging Face.
    Download the audio files:
    
    ```bash
    wget https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip?download=true -O path/to/vox1_dev_wav.zip
    ```

    Download the metadata file:
    
    ```bash
    wget https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_meta.csv -O path/to/vox1_meta.csv
    ```

After downloading, your data/ directory should have a structure similar to this, containing both audio content and metadata for fine-tuning:

```bash
data/

├── AudioSet/
    ├── balanced_train/
    ├── balanced_train_segments.csv

├── VoxCeleb/
    ├── vox1_dev_wav
    ├── vox1_meta.csv

├── LibriSpeech/
    ├── train-clean-100/
    ├── SPEAKERS.TXT

└── ESC/
```
-----

## Configuration
Specific architectural parameters and other experiment settings can be easily modified in the `config.py` file. This allows for flexible customization without altering the main scripts.

-----

## Running Experiments

The `run_experiments.sh` file is the main script for executing pre-training and fine-tuning experiments.

The script is configured to:

  * Perform **MAE pre-training** on the **LibriSpeech** and **VoxCeleb** datasets.
  * Perform **fine-tuning** of the pre-trained model on the **ESC-50** and **VoxCeleb** datasets for specific classification tasks.

To launch the experiments, simply run:

```bash
bash run_experiments.sh
```
-----

## Results

Results are showed in `MAE.pdf`.

-----

## Limitations
A small-scale dataset (AudioSet + LibriSpeech, ~69 hours) is used to validate the proof-of-concept. This is significantly smaller than the datasets used in the original papers (~6,500 hours).

-----

