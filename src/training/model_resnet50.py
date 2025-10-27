import torch
import torch.nn as nn
from training.attention import Attention

# === Bottleneck Block cho ResNet50 1D ===
class Bottleneck1D(nn.Module):
    expansion = 4  # số nhân kênh (ResNet50 tiêu chuẩn)

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity
        out = self.relu(out)
        return out
    
# === ResNet50 1D + Attention ===
class ResNet1D50Attention(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet1D50Attention, self).__init__()
        self.in_channels = 64

        # Khởi tạo conv đầu vào
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Các block ResNet50
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Attention layer
        self.attention = Attention(512 * Bottleneck1D.expansion)

        # Fully connected
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * Bottleneck1D.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * Bottleneck1D.expansion),
            )

        layers = [Bottleneck1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Bottleneck1D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: (batch, 12, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Attention
        x = x.permute(0, 2, 1)  # (batch, seq_len, feat_dim)
        x, weights = self.attention(x)

        out = self.fc(x)
        return out