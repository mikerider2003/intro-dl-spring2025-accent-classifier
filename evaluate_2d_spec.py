import os
import torch

from torch.utils.data import DataLoader
from models.cnn_2d_spectrogram import CNN2DSpectrogram
from data_loader_2d_spec import TestSpectrogramDataset

def load_model(model_path, device, num_classes=5):
    model = CNN2DSpectrogram(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def predict(model, dataloader, output_file='predictions_cnn2d.csv'):
    device = next(model.parameters()).device
    results = []

    with torch.no_grad():
        for filenames, spectrograms in dataloader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            preds = outputs.argmax(dim=1).cpu().numpy() + 1  # convert to 1–5

            for fname, pred in zip(filenames, preds):
                file_id = extract_id(fname)
                results.append((file_id, pred))

    # Save to CSV
    with open(output_file, 'w') as f:
        f.write("Id,label\n")
        for file_id, pred in results:
            f.write(f"{file_id},{pred}\n")

    print(f"Saved predictions to {output_file}")

def main():
    model_path = "cnn2d_model.pth"
    test_folder = "./data/Test set"
    output_file = "predictions_cnn2d.csv"
    sample_rate = 16000
    n_mels = 128
    max_len = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {model_path}")
    model = load_model(model_path, device)

    print(f"Loading test data from {test_folder}")
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.wav')]
    test_dataset = TestSpectrogramDataset(test_files, sample_rate=sample_rate, n_mels=n_mels, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Running predictions...")
    predict(model, test_loader, output_file)

if __name__ == "__main__":
    main()