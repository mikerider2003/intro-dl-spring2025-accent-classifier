import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2DSpectrogram(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Input shape: [batch_size, 1, n_mels, time_frames]
        
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    model = CNN2DSpectrogram(num_classes=5)
    print(model)
    
    # Test with a sample input (batch_size, channels, n_mels, time_frames)
    example_input = torch.randn(8, 1, 128, 500)
    output = model(example_input)
    print(f"Input shape: {example_input.shape}")
    print(f"Output shape: {output.shape}")