import os
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class AccentAudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_len=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        waveform = waveform.mean(dim=0).unsqueeze(0)

        # Normalize the waveform for 1 second = 16000 samples (standard choice for audio classification) | if shorter pad with zeros, if longer truncate
        if waveform.size(1) < self.max_len:
            pad_len = self.max_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.max_len]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, label

def get_label_from_filename(filename):
    return int(filename[0]) - 1

def prepare_datasets(data_dir='./data/Train', test_size=0.2, max_len=16000):
    files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    file_paths = [os.path.join(data_dir, f) for f in files]
    labels = [get_label_from_filename(f) for f in files]

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_dataset = AccentAudioDataset(train_files, train_labels, max_len)
    val_dataset = AccentAudioDataset(val_files, val_labels, max_len)
    return train_dataset, val_dataset

if __name__ == "__main__":
    train_dataset, val_dataset = prepare_datasets()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # First 5 samples
    for i in range(5):
        waveform, label = train_dataset[i]
        print(f"Sample {i}:")
        print(f"File_path: {train_dataset.file_paths[i]}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Label: {label.item()}")
        print()