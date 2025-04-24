import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DRawAudio(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = CNN1DRawAudio(num_classes=5)
    print(model)

    example_input = torch.randn(8, 1, 16000)
    output = model(example_input)
    print(output.shape)