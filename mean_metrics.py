import pandas as pd

def calculate_mean_metrics(df):
    # Take best trail for each fold by maximum validation F1 score
    best_per_fold = df.loc[df.groupby("Fold")["Val_F1"].idxmax()]

    # Calculate mean metrics from those rows
    mean_metrics = best_per_fold[[
        "Train_Loss", "Train_Acc", "Train_F1",
        "Val_Loss", "Val_Acc", "Val_F1"
    ]].mean()

    return mean_metrics


if __name__ == "__main__":
    # 1D CNN
    print("1D CNN:")
    df = pd.read_csv("Figures/Best Trials/1d_folds.csv")

    # Calculate mean metrics
    mean_metrics = calculate_mean_metrics(df)
    print(mean_metrics)


    # 2D CNN
    print("2D CNN:")
    df = pd.read_csv("Figures/Best Trials/2d_folds.csv")
    
    # Calculate mean metrics
    mean_metrics = calculate_mean_metrics(df)
    print(mean_metrics)
