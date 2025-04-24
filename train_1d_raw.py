import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from models.cnn_1d_raw_audio import CNN1DRawAudio
from data_loader import prepare_datasets

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for waveforms, labels in dataloader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * waveforms.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * waveforms.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # ==== Config ====
    data_path = './data/Train'
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.001
    num_classes = 5
    max_len = 16000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==== Data ====
    train_ds, val_ds = prepare_datasets(data_dir=data_path, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    print("Data loaded successfully.")

    # ==== Model ====
    model = CNN1DRawAudio(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model initialized successfully.")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # ==== Training Loop ====
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }
    np.save("cnn1d_model_metrics.npy", metrics)

    # ==== Save Model ====
    torch.save(model.state_dict(), 'cnn1d_model.pth')
    print("Training complete. Model saved to cnn1d_model.pth")

if __name__ == '__main__':
    main()
