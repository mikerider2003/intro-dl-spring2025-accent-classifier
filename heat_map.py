import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


from torch.utils.data import DataLoader
from models.cnn_1d_raw_audio import CNN1DRawAudio
from models.cnn_2d_spectrogram import CNN2DSpectrogram
from data_loader_2d_spec import TestSpectrogramDataset
from data_loader_1d_raw import TestAudioDataset

def load_model_1d(model_path, device, num_classes=5):
    model = CNN1DRawAudio(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_model_2d(model_path, device, num_classes=5):
    model = CNN2DSpectrogram(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def extract_class(filename):
    # 1f_1018.wav -> 1
    basename = os.path.basename(filename)
    return int(basename[0])

def extract_gender(filename):
    # 1f_1018.wav -> f
    basename = os.path.basename(filename)
    return basename[1]

def predict(model, dataloader, output_csv='heatmap_prediction_2d.csv'):
    device = next(model.parameters()).device
    results = []
    true_labels = []
    predictions = []
    all_data = []

    with torch.no_grad():
        for filenames, spectrograms in dataloader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            preds = outputs.argmax(dim=1).cpu().numpy() + 1  # convert to 1–5

            for fname, pred in zip(filenames, preds):
                file_id = extract_id(fname)
                row = {'id': file_id, 'prediction': int(pred)}
                
                # Extract true class and gender if available in filename
                try:
                    true_class = extract_class(fname)
                    gender = extract_gender(fname)
                    true_labels.append(true_class)
                    predictions.append(pred)
                    row['true_class'] = true_class
                    row['gender'] = gender
                except:
                    row['true_class'] = None
                    row['gender'] = None
                
                results.append((file_id, pred))
                all_data.append(row)
        
    # Save results to CSV file
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    return true_labels, predictions

def plot_heatmap(df1, df2, heatmap_file=None):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Process data and create heatmap for the 1D model
    grouped1 = df1.groupby(['true_class', 'gender'])
    results1 = []
    
    for (true_class, gender), group in grouped1:
        # Calculate accuracy for this specific true_class and gender
        true_values = group['true_class']
        pred_values = group['prediction']
        # Calculate accuracy instead of F1 for single-class groups
        accuracy = (true_values == pred_values).mean()
        results1.append({'Accent': true_class, 'Gender': gender, 'Accuracy': accuracy})
    
    f1_df1 = pd.DataFrame(results1)
    heatmap_data1 = f1_df1.pivot(index='Accent', columns='Gender', values='Accuracy')
    heatmap_data1 = heatmap_data1.rename(columns={'f': 'Female', 'm': 'Male'})
    
    # Process data and create heatmap for the 2D model
    grouped2 = df2.groupby(['true_class', 'gender'])
    results2 = []
    
    for (true_class, gender), group in grouped2:
        # Calculate accuracy for this specific true_class and gender
        true_values = group['true_class']
        pred_values = group['prediction']
        # Calculate accuracy instead of F1 for single-class groups
        accuracy = (true_values == pred_values).mean()
        results2.append({'Accent': true_class, 'Gender': gender, 'Accuracy': accuracy})
    
    f1_df2 = pd.DataFrame(results2)
    heatmap_data2 = f1_df2.pivot(index='Accent', columns='Gender', values='Accuracy')
    heatmap_data2 = heatmap_data2.rename(columns={'f': 'Female', 'm': 'Male'})
    
    # Keep the existing colormap as requested
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Plot first heatmap (1D model)
    sns.heatmap(heatmap_data1, annot=True, fmt=".3f", cmap=cmap, 
                cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 12}, ax=ax1)
    ax1.set_title("1D Model:", fontsize=14)
    ax1.set_xlabel("Gender", fontsize=12)
    ax1.set_ylabel("Accent", fontsize=12)
    
    # Plot second heatmap (2D model)
    sns.heatmap(heatmap_data2, annot=True, fmt=".3f", cmap=cmap, 
                cbar_kws={'label': 'Accuracy'}, annot_kws={"size": 12}, ax=ax2)
    ax2.set_title("2D Model", fontsize=14)
    ax2.set_xlabel("Gender", fontsize=12)
    ax2.set_ylabel("Accent", fontsize=12)
    
    plt.tight_layout()
    
    if heatmap_file:
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {heatmap_file}")
    
    plt.show()

def main():
    train_folder = "./data/Train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heatmap_file = "Figures/confusion_heatmap.png"
    
    model_path_2d = "cnn2d_model.pth"
    heatmap_predictions_2d = "heatmap_prediction_2d.csv"
    sample_rate = 16000
    n_mels = 128
    max_len = 500

    model_path_1d = "cnn1d_model.pth"
    heatmap_predictions_1d = "heatmap_prediction_1d.csv"
    max_len = 80000

    if not os.path.exists(heatmap_predictions_1d):
        # if predictions do not exist, load the model and run predictions
        print(f"Loading model from {model_path_1d}")
        model = load_model_1d(model_path_1d, device)

        print(f"Loading test data from {train_folder}")
        test_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.wav')]
        test_dataset = TestAudioDataset(test_files, max_len=max_len)
        test_loader = DataLoader(test_dataset, batch_size=1)

        print(f"Running predictions...")
        true_labels, predictions = predict(model, test_loader, output_csv=heatmap_predictions_1d)
    

    if not os.path.exists(heatmap_predictions_2d):
        # if predictions do not exist, load the model and run predictions
        print(f"Loading model from {model_path_2d}")
        model = load_model_2d(model_path_2d, device)

        print(f"Loading test data from {train_folder}")
        test_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.wav')]
        test_dataset = TestSpectrogramDataset(test_files, sample_rate=sample_rate, n_mels=n_mels, max_len=max_len)
        test_loader = DataLoader(test_dataset, batch_size=1)

        print(f"Running predictions...")
        true_labels, predictions = predict(model, test_loader, output_csv=heatmap_predictions_2d)
    
    # Load existing predictions
    print(f"Loading existing predictions from {heatmap_predictions_1d} and {heatmap_predictions_2d}")
    df1 = pd.read_csv(heatmap_predictions_1d)
    df2 = pd.read_csv(heatmap_predictions_2d)
    plot_heatmap(df1, df2, heatmap_file=heatmap_file)


if __name__ == "__main__":
    main()