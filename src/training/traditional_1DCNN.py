import torch
import torch.nn as nn
from training.attention import Attention


# === Traditional 1D CNN with Attention ===
class ECG_1DCNN_Attention(nn.Module):
    
    """
    Traditional 1D CNN for ECG classification with attention mechanism.
    
    Architecture inspired by:
    - Kiranyaz et al. (2015) "Real-Time Patient-Specific ECG Classification by 1D CNNs"
    - Yildirim et al. (2018) "A novel wavelet sequence based on deep bidirectional LSTM network model for ECG signal classification"
    
    Upgrades from baseline:
    1. Increased depth (5 conv blocks instead of 3)
    2. Progressive filter expansion (64->128->256->512->512)
    3. Batch Normalization after each conv layer for training stability
    4. Dropout layers (0.5) for regularization
    5. Attention mechanism for feature weighting
    6. Adaptive feature map sizes with varying kernel sizes
    """
    
    def __init__(self, input_channels=6, num_classes=15, dropout_rate=0.5):
        super(ECG_1DCNN_Attention, self).__init__()
        
        # Block 1: Initial feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 2: Feature learning
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 3: Deep feature extraction
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 4: High-level features
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 5: Abstract features
        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = Attention(512)
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input: (batch, channels, seq_len)
        # For 6 channels (V1-V6), f=500Hz
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout(x)
        
        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool5(x)
        x = self.dropout(x)
        
        # Attention mechanism
        x = x.permute(0, 2, 1)  # (batch, seq_len, feat_dim)
        x, attention_weights = self.attention(x)
        
        # Classification
        out = self.fc(x)
        
        return out