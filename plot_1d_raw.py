import numpy as np
import optuna.visualization
import joblib

import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, train_accuracies, save_path=None):
    """
    Plot training metrics (loss and accuracy) over epochs on a single plot.
    
    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        save_path: Path to save the plot image (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with primary and secondary y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot losses on primary y-axis
    train_loss_line, = ax1.plot(epochs, train_losses, 'b--', label='Training Loss', alpha=0.6)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary y-axis for accuracy
    ax2 = ax1.twinx()
    train_acc_line, = ax2.plot(epochs, train_accuracies, 'r-', marker='o', label='Training Accuracy')
    ax2.set_ylabel('Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Final Model Training Metrics')
    
    # Add both lines to the legend
    lines = [train_loss_line, train_acc_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='center right')
    
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_hyperparameter_search_results(study, save_path=None):
    """
    Plot the results from the hyperparameter optimization study.
    
    Args:
        study: The Optuna study object
        save_path: Path to save the plot image (optional)
    """
    # Use Optuna visualization functions if available
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot optimization history
        fig1 = plot_optimization_history(study)
        if save_path:
            history_path = save_path.replace('.png', '_history.png')
            fig1.write_image(history_path)
            print(f"Optimization history saved to {history_path}")
        fig1.show()
        
        # Plot parameter importances
        fig2 = plot_param_importances(study)
        if save_path:
            importance_path = save_path.replace('.png', '_importance.png')
            fig2.write_image(importance_path)
            print(f"Parameter importance saved to {importance_path}")
        fig2.show()
        
    except ImportError:
        print("Optuna visualization package not available.")
    except Exception as e:
        print(f"Error generating Optuna visualizations: {e}")
        print("If this relates to missing packages, try: pip install plotly kaleido")

if __name__ == "__main__":
    file_name = "cnn1d_model_metrics.npy"
    
    # Load the metrics from the .npy file
    metrics = np.load(file_name, allow_pickle=True).item()
    
    # Extract available metrics
    train_losses = metrics['train_loss']
    train_accuracies = metrics['train_acc']
    
    # Print best hyperparameters
    print("Best hyperparameters from optimization:")
    for param, value in metrics['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best cross-validated accuracy: {metrics['best_cv_acc']:.4f}")
    
    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, save_path="Figures/1d_raw.png")
    
    # Load and plot the Optuna study results
    try:
        study = joblib.load("cnn1d_model_optuna_study.pkl")
        plot_hyperparameter_search_results(study, save_path="Figures/1d_raw_HP_search.png")
    except FileNotFoundError:
        print("Optuna study file not found. Run training first.")
    except ImportError:
        print("Joblib package not available.")