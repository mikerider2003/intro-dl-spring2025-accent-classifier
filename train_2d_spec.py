import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import joblib

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from models.cnn_2d_spectrogram import CNN2DSpectrogram
from data_loader_2d_spec import prepare_datasets

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []

    for spectrograms, labels in dataloader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * spectrograms.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in dataloader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * spectrograms.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions and labels for F1 calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1

def objective(trial):
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 15, 50)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    # Print trial information
    print(f"\n{'='*50}")
    print(f"Starting Trial {trial.number}")
    print(f"Parameters: batch_size={batch_size}, lr={learning_rate:.6f}, epochs={num_epochs}, weight_decay={weight_decay:.6f}, dropout={dropout_rate:.2f}")
    print(f"{'='*50}")
    
    # Fixed parameters
    data_path = './data/Train'
    num_classes = 5
    sample_rate = 16000
    n_mels = 128
    max_len = 500  # Maximum number of time frames for spectrograms
    k_folds = 5    # Number of folds for cross-validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the complete dataset
    full_dataset = prepare_datasets(data_dir=data_path, sample_rate=sample_rate, n_mels=n_mels, max_len=max_len)
    
    # Setup K-Fold cross validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # For storing validation F1 scores across folds
    fold_val_f1_scores = []
    
    # Start K-Fold cross-validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\nFold {fold+1}/{k_folds}")
        # Create data samplers
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        # Create data loaders
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler)
        
        # Initialize model
        model = CNN2DSpectrogram(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop for this fold
        best_val_f1 = 0
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, train_f1 = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            
            # Print epoch progress
            print(f"  Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Pruning based on the F1 score
            if epoch == num_epochs // 2:
                trial.report(val_f1, epoch)  # Use F1 score instead of accuracy
                if trial.should_prune():
                    print(f"  Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()
            
            # Track best F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
        
        # Store best F1 score for this fold
        fold_val_f1_scores.append(best_val_f1)
        print(f"  Fold {fold+1} best validation F1 score: {best_val_f1:.4f}")
    
    # Average validation F1 score across all folds
    mean_val_f1 = np.mean(fold_val_f1_scores)
    print(f"\nTrial {trial.number} completed with mean validation F1 score: {mean_val_f1:.4f}")
    return mean_val_f1

def main():
    # Fixed parameters
    data_path = './data/Train'
    num_classes = 5
    sample_rate = 16000
    n_mels = 128
    max_len = 500  # Maximum number of time frames for spectrograms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create an Optuna study for maximizing F1 score
    study = optuna.create_study(direction='maximize', 
                              pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    # TODO: Use more trials (e.g., 30) for production
    study.optimize(objective, n_trials=50)  
    
    joblib.dump(study, "cnn2d_model_optuna_study.pkl")
    print("Optuna study saved to cnn2d_model_optuna_study.pkl")

    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*50}")
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best Cross-validated F1 Score: {best_value:.4f}")
    print(f"{'='*50}")
    
    # Train final model with 100% of training data
    print("\nTraining final model with 100% of training data...")
    
    # Extract best hyperparameters
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    num_epochs = best_params['num_epochs']
    weight_decay = best_params.get('weight_decay', 0)
    dropout_rate = best_params.get('dropout_rate', 0.3)
    
    # Load the full dataset for final model training
    full_dataset = prepare_datasets(data_dir=data_path, sample_rate=sample_rate, n_mels=n_mels, max_len=max_len)
    
    # Use all training data without validation split
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    print("Data loaded successfully.")
    
    # Initialize model
    model = CNN2DSpectrogram(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print("Model initialized successfully.")
    
    # Track metrics
    train_losses = []
    train_accuracies = []
    train_f1_scores = []  # Add this line

    # Train final model
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train(model, train_loader, criterion, optimizer, device)
        
        # Update learning rate scheduler
        scheduler.step(train_loss)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)  # Add this line
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
    # Save results
    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'train_f1': train_f1_scores,  # Add this line
        'best_params': best_params,
        'best_cv_f1': best_value,
    }
    np.save("cnn2d_model_metrics.npy", metrics)
    
    # Save the final model
    torch.save(model.state_dict(), 'cnn2d_model.pth')
    print("Training complete. Model saved to cnn2d_model.pth")

if __name__ == '__main__':
    main()