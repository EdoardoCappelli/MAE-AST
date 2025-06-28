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

  * **AudioSet (for VoxCeleb):**
    To download AudioSet audios, which are based on YouTube, you need to have **FFmpeg** installed on your system. FFmpeg is used to download the videos and extract the 10-second audio segments.

    ```bash
    python download/download_audioset.py
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

