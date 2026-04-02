# training/model_resnet18_lstm.py
import torch
import torch.nn as nn
from training.attention import Attention

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    
    
    
# ========== VARIANT 1: ResNet18 Thuần ==========
class ResNet18_Pure(nn.Module):
    """
    ResNet18 thuần túy - chỉ có CNN + Global Average Pooling
    Không có LSTM, không có Attention
    """
    def __init__(self, num_classes=7, input_channels=12):
        super(ResNet18_Pure, self).__init__()
        self.in_channels = 64
        
        # CNN Backbone (ResNet18)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global Average Pooling thay vì LSTM/Attention
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, 12, Seq_Len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (Batch, 512, Reduced_Seq_Len)

        # Global Average Pooling
        x = self.global_pool(x)  # (Batch, 512, 1)
        x = x.squeeze(-1)  # (Batch, 512)
        
        out = self.fc(x)
        return out


# ========== VARIANT 2: ResNet18 + Attention ==========
class ResNet18_Attention(nn.Module):
    """
    ResNet18 + Attention
    Không có LSTM, chỉ dùng Attention để aggregate temporal features
    """
    def __init__(self, num_classes=7, input_channels=12):
        super(ResNet18_Attention, self).__init__()
        self.in_channels = 64
        
        # CNN Backbone (ResNet18)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Attention Layer (trực tiếp từ CNN features)
        self.attention = Attention(512)
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, 12, Seq_Len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (Batch, 512, Reduced_Seq_Len)

        # Prepare for Attention: (Batch, Seq, Features)
        x = x.permute(0, 2, 1)  # (Batch, Reduced_Seq_Len, 512)
        
        # Attention Pooling
        x, weights = self.attention(x)  # (Batch, 512)
        
        out = self.fc(x)
        return out

class ResNet18_LSTM_Attn(nn.Module):
    def __init__(self, num_classes=7, input_channels=12):
        super(ResNet18_LSTM_Attn, self).__init__()
        self.in_channels = 64
        
        # CNN Backbone (ResNet18)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # LSTM Layer (Temporal Context)
        # Input to LSTM: (Batch, Seq_len_features, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        
        # Attention Layer
        # Input to Attention: 256 * 2 (bidirectional) = 512
        self.attention = Attention(512)
        
        # Classifier
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, 12, Seq_Len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
        # CNN Output: (Batch, 512, Reduced_Seq_Len)

        # Prepare for LSTM
        x = x.permute(0, 2, 1) # (Batch, Reduced_Seq_Len, 512)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # (Batch, Reduced_Seq_Len, 512)
        
        # Attention Pooling
        x, weights = self.attention(x) # (Batch, 512)
        
        out = self.fc(x)
        return out