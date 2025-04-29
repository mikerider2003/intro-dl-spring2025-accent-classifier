import numpy as np
import optuna.visualization
import joblib

import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, train_accuracies, train_f1_scores=None, save_path=None):
    """
    Plot training metrics (loss, accuracy and F1-score) over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        train_f1_scores: List of training F1 scores per epoch (optional)
        save_path: Path to save the plot image (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure and primary axis for loss
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot losses on primary y-axis
    train_loss_line, = ax1.plot(epochs, train_losses, 'b--', label='Training Loss', alpha=0.6)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary y-axis for accuracy and F1 score
    ax2 = ax1.twinx()
    train_acc_line, = ax2.plot(epochs, train_accuracies, 'r-', marker='o', label='Training Accuracy', markersize=4)
    
    # Plot F1 score if available
    if train_f1_scores is not None:
        train_f1_line, = ax2.plot(epochs, train_f1_scores, 'g-', marker='s', label='Training F1 Score', markersize=4)
        ax2.set_ylabel('Accuracy / F1 Score', color='black')
        lines = [train_loss_line, train_acc_line, train_f1_line]
    else:
        ax2.set_ylabel('Accuracy', color='red')
        lines = [train_loss_line, train_acc_line]
    
    ax2.tick_params(axis='y', labelcolor='black')
    
    plt.title('CNN 2D Spectrogram - Training Metrics')
    
    # Add all lines to the legend
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
    Plot the results from the hyperparameter optimization study using only matplotlib.
    
    Args:
        study: The Optuna study object
        save_path: Path to save the plot image (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a figure with subplots - make it wider
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Extract data for history plot
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        trial_numbers = [t.number for t in trials if t.value is not None]
        
        # Plot optimization history on first subplot
        ax1.plot(trial_numbers, values, 'o-', color='blue', markersize=8)
        ax1.set_xlabel('Trial Number', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Optimization History', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=11)
        
        # Extract data for parameter importances
        importances = {}
        for trial in trials:
            if trial.value is None:
                continue
            for param_name, param_value in trial.params.items():
                if param_name not in importances:
                    importances[param_name] = []
                importances[param_name].append((param_value, trial.value))
        
        # Calculate simple correlations as importance metric
        param_importances = {}
        for param_name, values in importances.items():
            if len(values) <= 1:
                continue
            x = np.array([float(v[0]) if isinstance(v[0], (int, float)) else hash(str(v[0])) for v in values])
            y = np.array([v[1] for v in values])
            if len(np.unique(x)) > 1:  # Need at least two distinct values
                correlation = abs(np.corrcoef(x, y)[0, 1]) if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
                param_importances[param_name] = correlation
        
        # Sort importances
        sorted_importances = sorted(param_importances.items(), key=lambda x: x[1], reverse=True)
        
        # Plot parameter importances on second subplot
        if sorted_importances:
            names = [item[0] for item in sorted_importances]
            values = [item[1] for item in sorted_importances]
            # Adjust the height based on number of parameters
            bar_height = 0.6
            y_pos = np.arange(len(names))
            ax2.barh(y_pos, values, height=bar_height, color='green')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names, fontsize=11)
            ax2.set_xlabel('Importance (Correlation)', fontsize=12)
            ax2.set_title('Parameter Importances', fontsize=14)
            # Ensure there's enough vertical space
            ax2.set_ylim(-0.5, len(names) - 0.5 + 0.5)  # Add extra space at top
        else:
            ax2.text(0.5, 0.5, "Not enough data to calculate importances", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # Improve overall layout
        plt.subplots_adjust(wspace=0.3)  # Add more space between subplots
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hyperparameter search results saved to {save_path}")
        
        plt.show()
            
    except ImportError:
        print("Required packages not available.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

# Update the main section to properly display F1 scores
if __name__ == "__main__":
    file_name = "cnn2d_model_metrics.npy"
    
    # Load the metrics from the .npy file
    metrics = np.load(file_name, allow_pickle=True).item()
    
    # Extract available metrics
    train_losses = metrics['train_loss']
    train_accuracies = metrics['train_acc']
    train_f1_scores = metrics.get('train_f1', None)  # Get F1 scores if available
    
    # Print best hyperparameters
    print("Best hyperparameters from optimization:")
    for param, value in metrics['best_params'].items():
        print(f"  {param}: {value}")
    
    # Display best CV F1 score instead of accuracy
    if 'best_cv_f1' in metrics:
        print(f"Best cross-validated F1 score: {metrics['best_cv_f1']:.4f}")
    else:
        print(f"Best cross-validated accuracy: {metrics.get('best_cv_acc', 0):.4f}")
    
    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, train_f1_scores, save_path="Figures/2d_spec.png")
    
    # Load and plot the Optuna study results
    try:
        study = joblib.load("cnn2d_model_optuna_study.pkl")
        plot_hyperparameter_search_results(study, save_path="Figures/2d_spec_HP_search.png")
    except FileNotFoundError:
        print("Optuna study file not found. Run training first.")
    except ImportError:
        print("Joblib package not available.")