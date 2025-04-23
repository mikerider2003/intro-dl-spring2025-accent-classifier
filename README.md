# Accent Classification Project

## Overview
This repository contains code and resources for training neural networks to classify different accents. The first model processes raw 1D audio, and a second spectrogram-based model will be added in the future.

## Model 1: Raw Audio (1D)
- Implemented in [`models.CNN1DRawAudio`](models/cnn_1d_raw_audio.py)
- Trainable with [`train_1d_raw.py`](train/train_1d_raw.py)
- Uses a simple 1D convolutional architecture on raw audio waveforms.

- Train, Valdation split: 80% train, 20% validation

### Usage
1. Install dependencies via `pip install -r requirements.txt`.
2. Place `.wav` files in the `data/Train` folder.
3. Run `python train.train_1d_raw.py` to start training.
4. The model training accuracy can be visualized using `python plot_1d_raw.py`, which will generate a plot of the training and validation accuracy.
5. The model can predict on test data using `python evaluate_1d_raw.py`, which will generate a CSV file with predictions.

## Model 2: 2D Spectrogram (WIP)
- Will be implemented in [`models.cnn_2d_spectrogram`](models/cnn_2d_spectrogram.py)
- Training script will be in [`train_2d_spec.py`](train/train_2d_spec.py)
- Intended for spectrogram inputs.

Stay tuned for updates on the second model.