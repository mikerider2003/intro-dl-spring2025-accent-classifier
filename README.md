# Accent Classification Project

## Overview
This repository contains code and resources for training neural networks to classify different accents. The project implements a 1D CNN model processing raw audio waveforms, with plans for a 2D CNN spectrogram-based model.

- [Accent Classification Project](#accent-classification-project)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Model 1: Raw Audio (1D)](#model-1-raw-audio-1d)
    - [Architecture:](#architecture)
    - [Training Details:](#training-details)
    - [Usage](#usage)
  - [Model 2: 2D Spectrogram (WIP)](#model-2-2d-spectrogram-wip)
  - [Project Structure](#project-structure)

## Dataset
- **Format**: Audio samples as mono-channel WAV files at 16kHz sample rate
- **Processing**: Files are normalized and padded/trimmed to exactly 16000 samples (1 second)
- **Classes**: 5 accent classes (numbered 1-5 in filenames)

## Model 1: Raw Audio (1D)
- Implemented in [`models/cnn_1d_raw_audio.py`](models/cnn_1d_raw_audio.py)
- Trainable with [`train_1d_raw.py`](train_1d_raw.py)
- Uses a 1D convolutional architecture directly on raw audio waveforms

### Architecture:
- 3 Convolutional blocks (Conv1D + BatchNorm + ReLU + MaxPool)
- Final adaptive pooling and fully connected layer
- Output layer with 5 neurons (one per accent class)

### Training Details:
- **Regularization**: L2 regularization via weight decay in Adam optimizer
- **Cross-Validation**: 5-Fold CV during hyperparameter search
- **Early stopping**: With patience of 7 epochs
- **Train/Val Split**: 80% train, 20% validation for final model training
- **Hyperparameter Optimization**: Using Optuna with pruning

### Usage
1. Install dependencies via `pip install -r requirements.txt`
2. Place `.wav` files in the `data/Train` folder
3. Run `python train_1d_raw.py` to start training
4. Visualize training metrics using `python plot_1d_raw.py`
5. Generate predictions on test data using `python evaluate_1d_raw.py`

## Model 2: 2D Spectrogram (WIP)
- Will be implemented in [`models/cnn_2d_spectrogram.py`](models/cnn_2d_spectrogram.py)
- Training script will be in [`train_2d_spec.py`](train_2d_spec.py)
- Intended for spectrogram-based inputs

## Project Structure