import re
import pandas as pd
import matplotlib.pyplot as plt


def plot_k_fold(file_path, save_path):
    df = extract_k_fold_scores(file_path)
    folds = df['Fold'].unique()

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
    axes = axes.flatten()

    for i, fold in enumerate(folds):
        ax = axes[i]
        fold_data = df[df['Fold'] == fold]
        ax.plot(fold_data['Epoch'], fold_data['Train_F1'], label='Train F1', linestyle='-')
        ax.plot(fold_data['Epoch'], fold_data['Val_F1'], label='Validation F1', linestyle='--')
        ax.set_title(f'Fold {fold}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(folds), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Train vs Validation F1 Scores per Fold', fontsize=16)
    fig.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=150)

def plot_k_fold_with_mean(file_path_1, file_path_2, save_path, model1_name="1D CNN", model2_name="2D CNN"):
    df_1 = extract_k_fold_scores(file_path_1)
    df_2 = extract_k_fold_scores(file_path_2)
    
    folds = df_1['Fold'].unique()
    
    # Create a single figure
    plt.figure(figsize=(6, 4))
    
    # Colors for each model
    model1_color = '#1f77b4'  # blue
    model2_color = '#ff7f0e'  # orange
    fold_alpha = 0.3  # transparency for individual folds
    
    # Plot individual folds for Model 1 with transparency
    for fold in folds:
        fold_data = df_1[df_1['Fold'] == fold]
        plt.plot(fold_data['Epoch'], fold_data['Train_F1'], 
                 color=model1_color, linestyle='-', alpha=fold_alpha)
        plt.plot(fold_data['Epoch'], fold_data['Val_F1'], 
                 color=model1_color, linestyle='--', alpha=fold_alpha)
    
    # Plot individual folds for Model 2 with transparency
    for fold in folds:
        fold_data = df_2[df_2['Fold'] == fold]
        plt.plot(fold_data['Epoch'], fold_data['Train_F1'], 
                 color=model2_color, linestyle='-', alpha=fold_alpha)
        plt.plot(fold_data['Epoch'], fold_data['Val_F1'], 
                 color=model2_color, linestyle='--', alpha=fold_alpha)
    
    # Calculate and plot mean for each model
    epochs = df_1['Epoch'].unique()
    
    # Model 1 means
    train_means_1 = [df_1[df_1['Epoch'] == e]['Train_F1'].mean() for e in epochs]
    val_means_1 = [df_1[df_1['Epoch'] == e]['Val_F1'].mean() for e in epochs]
    
    # Model 2 means
    train_means_2 = [df_2[df_2['Epoch'] == e]['Train_F1'].mean() for e in epochs]
    val_means_2 = [df_2[df_2['Epoch'] == e]['Val_F1'].mean() for e in epochs]
    
    # Plot means with thicker lines
    plt.plot(epochs, train_means_1, color=model1_color, linestyle='-', linewidth=2.5, 
             label=f'{model1_name} - Train (Mean)')
    plt.plot(epochs, val_means_1, color=model1_color, linestyle='--', linewidth=2.5, 
             label=f'{model1_name} - Val (Mean)')
    
    plt.plot(epochs, train_means_2, color=model2_color, linestyle='-', linewidth=2.5, 
             label=f'{model2_name} - Train (Mean)')
    plt.plot(epochs, val_means_2, color=model2_color, linestyle='--', linewidth=2.5, 
             label=f'{model2_name} - Val (Mean)')
    
    # plt.title('Training and Validation F1 Scores Comparison (Mean ± Individual Folds)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add a legend with only the mean lines
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_k_fold_scores(file_path, debug=False):
    with open(file_path, "r") as f:
        content = f.readlines()
    
    # Find the "Best Hyperparameters:"
    for i, line in enumerate(content):
        if "Best Hyperparameters:" in line:
            best_HP = (content[i:i+7])
    
    # Extract the best_F1 score
    best_CV_F1 = float(best_HP[-1].split(": ")[1].strip())
    if debug:
        print(f"Best CV F1: {best_CV_F1}")

    # Find the best_F1 index in content
    for i, line in enumerate(content):
        if f"mean validation F1 score: {best_CV_F1}" in line:
            best_trial = int(line.split("completed")[0].split(" ")[1])
    
    if debug:
        print(f"Best trial: {best_trial}")

    start = f"Starting Trial {best_trial}"
    end = f"Trial {best_trial} completed with mean validation F1 score:"

    for i, line in enumerate(content):
        if start in line:
            start_index = i
        if end in line:
            end_index = i
            extracted_content = ''.join(content[start_index+4:end_index])
            break

    # Regex pattern capturing all metrics
    pattern_fold = r"^Fold\s+(\d+)/5"
    pattern_epoch = (
        r"^Epoch\s+(\d+)/\d+\s+\|"
        r"\s+Train Loss:\s+([0-9.]+),\s+Acc:\s+([0-9.]+),\s+F1:\s+([0-9.]+)\s+\|"
        r"\s+Val Loss:\s+([0-9.]+),\s+Acc:\s+([0-9.]+),\s+F1:\s+([0-9.]+)"
    )

    folds_data = []
    current_fold = None

    for line in extracted_content.splitlines():
        line = line.strip()
        fold_match = re.match(pattern_fold, line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            continue
        epoch_match = re.match(pattern_epoch, line)
        if epoch_match and current_fold is not None:
            (
                epoch,
                train_loss, train_acc, train_f1,
                val_loss, val_acc, val_f1
            ) = epoch_match.groups()

            folds_data.append({
                "Fold": current_fold,
                "Epoch": int(epoch),
                "Train_Loss": float(train_loss),
                "Train_Acc": float(train_acc),
                "Train_F1": float(train_f1),
                "Val_Loss": float(val_loss),
                "Val_Acc": float(val_acc),
                "Val_F1": float(val_f1)
            })

    df = pd.DataFrame(folds_data)
    if debug:
        print(df)

    return df


if __name__ == "__main__":
    plot_k_fold("cnn_1d_slurm.cerulean.30550.out", "Figures/1d_k_fold.png")
    plot_k_fold("cnn_2d_slurm.cerulean.30252.out", "Figures/2d_k_fold.png")

    # Save the DataFrame to a CSV file
    df = extract_k_fold_scores("cnn_1d_slurm.cerulean.30550.out")
    df.to_csv("Figures/Best Trials/1d_folds.csv", index=False)

    df = extract_k_fold_scores("cnn_2d_slurm.cerulean.30252.out")
    df.to_csv("Figures/Best Trials/2d_folds.csv", index=False)

    plot_k_fold_with_mean("cnn_1d_slurm.cerulean.30550.out", 
                          "cnn_2d_slurm.cerulean.30252.out", 
                          "Figures/combined_models_comparison.png")