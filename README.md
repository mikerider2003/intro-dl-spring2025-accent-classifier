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
    - [Architecture:](#architecture-1)
    - [Training Details:](#training-details-1)
    - [Optional Enhancements:](#optional-enhancements)
  - [TODO:](#todo)
  - [Project Structure](#project-structure)

## Dataset
- **Format**: Audio samples as mono-channel WAV files at 16kHz sample rate
- **Processing**: Files are normalized and padded/trimmed to exactly 80 000 samples (5 second) the selection was made to ensure that the model can capture more complex features of the audio like accent. (Average length of the audio files is 5.2 seconds)
- **Classes**: 5 accent classes (numbered 1-5 in filenames)

- Dataset is not included in the repository. Users must download the dataset separately and place it in the `data/` folder. Resulting directory structure should be:
```
data/
├── Train
│   ├── 1f_1018.wav
│   ├── 1f_1026.wav
│   ├── ...
├── Test set
│   ├── 1035.wav
│   ├── 1074.wav
│   ├── ...
```

## Model 1: Raw Audio (1D)
- Implemented in [`models/cnn_1d_raw_audio.py`](models/cnn_1d_raw_audio.py)
- Trainable with [`train_1d_raw.py`](train_1d_raw.py)
- Uses a 1D convolutional architecture directly on raw audio waveforms

### Architecture:
- **Input**: Raw audio waveforms (80000 samples at 16kHz)
- **Convolutional Blocks**: 
  - **Block 1**: 16 filters → BatchNorm → ReLU → MaxPool(4) → Dropout
  - **Block 2**: 32 filters → BatchNorm → ReLU → MaxPool(4) → Dropout
  - **Block 3**: 64 filters → BatchNorm → ReLU → AdaptiveMaxPool → Dropout
- **Output**: Fully connected layer with 5 neurons (one per accent class)
- **Kernel Size**: 9 with padding of 4 for all convolutional layers
- **Dropout**: Configurable rate (optimized via hyperparameter search)

### Training Details:
- Regularization: L2 regularization via weight decay in Adam optimizer
- Cross-Validation: 5-Fold CV during hyperparameter search
- Train/Val Split: 80% train, 20% validation during hyperparameter search
- Pruning: Optuna pruning for efficient hyperparameter optimization
- **Hyperparameters (being optimized):**
  - **Batch Size**: Varied between 2-8  
  - **Learning Rate**: Searched in range 1e-4 to 1e-1 (log scale)  
  - **Weight Decay**: Searched in range 1e-6 to 1e-2 (log scale)  
  - **Dropout Rate**: Searched in range 0.2 to 0.5  
  - **Epochs**: Optimized between 15-50  
- **Learning Rate Schedule**: Step decay during final training  
- **Final Training**: Model trained on 100% of training data using best hyperparameters

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

### Architecture:
- **Input**: Log-Mel Spectrograms derived from raw audio
- **Preprocessing**:
  - Convert `.wav` files into spectrograms
  - Apply logarithmic scaling to highlight small differences
  - Use Mel scale to mimic human hearing
  - Normalize and scale input values
- **Spectrogram Dimensions**: Fixed input shape required (cropping or padding applied)

![Spectrogram Example](Figures/spectrogram_example.png)

- **Convolutional Blocks**: 
  - **Block 1**: 2D Conv → BatchNorm → ReLU → MaxPool → Dropout
  - **Block 2**: 2D Conv → BatchNorm → ReLU → MaxPool → Dropout
  - **Block 3**: 2D Conv → BatchNorm → ReLU → AdaptiveAvgPool → Dropout
- **Output**: Fully connected layer with 5 neurons (one per accent class)
- **Dropout**: Configurable rate (optimized via hyperparameter search)

### Training Details:
- **Regularization**: L2 regularization via weight decay in Adam optimizer  
- **Cross-Validation**: 5-Fold CV during hyperparameter search  
- **Train/Val Split**: 80% train, 20% validation during hyperparameter search  
- **Pruning**: Optuna pruning for efficient hyperparameter optimization  

- **Hyperparameters (being optimized):**
  - **Batch Size**: Varied between 2-8  
  - **Learning Rate**: Searched in range 1e-4 to 1e-1 (log scale)  
  - **Weight Decay**: Searched in range 1e-6 to 1e-2 (log scale)  
  - **Dropout Rate**: Searched in range 0.2 to 0.5  
  - **Epochs**: Optimized between 15-50  
- **Learning Rate Schedule**: Step decay during final training  
- **Final Training**: Model trained on 100% of training data using best hyperparameters  

### Optional Enhancements:
- **Data Augmentation**:
  - Add Gaussian noise
  - Time stretch (speed up or slow down)
  - Volume change (amplify or reduce)
  - Random cropping (e.g., random 2-second segments)
- **Denoising**: Apply low-pass filter before spectrogram generation  
- **Mel-Spectrogram Tuning**: Adjust Mel bin size or frequency ranges  

## TODO:
- Check train_2d_spec.py for any issues
- Check evaluate_2d_spec.py for any issues
- Check plot_2d_spec.py for any issues
- Add more data augmentation techniques
- Add more hyperparameter tuning options
- Check for overfitting
- Normalization is implemented but scaling is not

## Project Structure