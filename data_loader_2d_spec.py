import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import matplotlib.pyplot as plt

class AccentSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160, max_len=500):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_len = max_len
        
        # Create log-mel spectrogram transformation
        self.mel_spec_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        
        # Convert stereo to mono by averaging channels
        waveform = waveform.mean(dim=0).unsqueeze(0)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Generate mel spectrogram
        mel_spec = self.mel_spec_transform(waveform)
        
        # Convert to log scale (add small number to avoid log(0))
        log_mel_spec = torch.log(mel_spec + 1e-9)
        
        # Handle spectrogram length
        if log_mel_spec.shape[2] < self.max_len:
            # Pad if shorter
            pad_len = self.max_len - log_mel_spec.shape[2]
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_len))
        else:
            # Truncate if longer
            log_mel_spec = log_mel_spec[:, :, :self.max_len]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return log_mel_spec, label

def get_label_from_filename(filename):
    return int(filename[0]) - 1

def prepare_datasets(data_dir='./data/Train', test_size=0.2, sample_rate=16000, n_mels=80, max_len=500):
    files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    file_paths = [os.path.join(data_dir, f) for f in files]
    labels = [get_label_from_filename(f) for f in files]

    return AccentSpectrogramDataset(file_paths, labels, sample_rate=sample_rate, n_mels=n_mels, max_len=max_len)

class TestSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160, max_len=500):
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.max_len = max_len
        
        # Create log-mel spectrogram transformation
        self.mel_spec_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        waveform, sr = torchaudio.load(filepath)
        
        # Convert stereo to mono
        waveform = waveform.mean(dim=0).unsqueeze(0)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Generate mel spectrogram
        mel_spec = self.mel_spec_transform(waveform)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-9)
        
        # Handle spectrogram length
        if log_mel_spec.shape[2] < self.max_len:
            pad_len = self.max_len - log_mel_spec.shape[2]
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_len))
        else:
            log_mel_spec = log_mel_spec[:, :, :self.max_len]

        return filepath, log_mel_spec

if __name__ == "__main__":
    dataset = prepare_datasets()
    print(f"Dataset size: {len(dataset)}")

    # Display first 5 samples
    for i in range(min(5, len(dataset))):
        spec, label = dataset[i]
        print(f"Sample {i}:")
        print(f"File path: {dataset.file_paths[i]}")
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Label: {label.item()}")
        print()
    
    # Get the last sample
    spec, label = dataset[-1]
    
    # Convert to numpy for visualization
    spec_np = spec.squeeze().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-Mel Spectrogram (Label: {label.item()})')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.tight_layout()
    plt.savefig('Figures/spectrogram_example.png')
    plt.show()
    
    print(f"Spectrogram saved as 'first_spectrogram.png'")