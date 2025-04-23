import os
import torch

from torch.utils.data import DataLoader
from models.cnn_1d_raw_audio import CNN1DRawAudio
from data_loader import TestAudioDataset

def load_model(model_path, device, num_classes=5):
    model = CNN1DRawAudio(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def predict(model, dataloader, output_file='predictions_cnn1d.csv'):
    device = next(model.parameters()).device
    results = []

    with torch.no_grad():
        for filenames, waveforms in dataloader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms)
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
    model_path = "cnn1d_model.pth"
    test_folder = "./data/Test set"
    output_file = "predictions_cnn1d.csv"
    max_len = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {model_path}")
    model = load_model(model_path, device)

    print(f"Loading test data from {test_folder}")
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.wav')]
    test_dataset = TestAudioDataset(test_files, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=1)

    print(f"Running predictions...")
    predict(model, test_loader, output_file)

if __name__ == "__main__":
    main()
