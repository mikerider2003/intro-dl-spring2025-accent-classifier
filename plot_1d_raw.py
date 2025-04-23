import numpy as np
import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Plot training and validation metrics (loss and accuracy) over epochs on a single plot.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        save_path: Path to save the plot image (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with primary and secondary y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot losses on primary y-axis with reduced opacity
    train_loss_line, = ax1.plot(epochs, train_losses, 'b--', label='Training Loss', alpha=0.4)
    val_loss_line, = ax1.plot(epochs, val_losses, 'r--', label='Validation Loss', alpha=0.4)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    # Create secondary y-axis for accuracy
    ax2 = ax1.twinx()
    train_acc_line, = ax2.plot(epochs, train_accuracies, 'b-', marker='o', label='Training Accuracy')
    val_acc_line, = ax2.plot(epochs, val_accuracies, 'r-', marker='o', label='Validation Accuracy')
    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    plt.title('Training Metrics')
    
    # Combine legends from both axes
    lines = [train_loss_line, val_loss_line, train_acc_line, val_acc_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='center right')
    
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    file_name = "cnn1d_model_metrics.npy"
    
    # Load the metrics from the .npy file
    metrics = np.load(file_name, allow_pickle=True).item()
    
    train_losses = metrics['train_loss']
    train_accuracies = metrics['train_acc']
    val_losses = metrics['val_loss']
    val_accuracies = metrics['val_acc']

    # ==== Plot Metrics ====
    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)