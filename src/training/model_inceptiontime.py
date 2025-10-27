import torch
import torch.nn as nn
from training.attention import Attention

# === Khối Inception cơ bản với InstanceNorm ===
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=16):  # giảm bottleneck
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )

        self.conv1 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=7, padding=3, bias=False)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.norm = nn.InstanceNorm1d(out_channels * 4, affine=True)
        self.relu = nn.ReLU(inplace=False)  # tránh inplace

    def forward(self, x):
        x_bottle = self.bottleneck(x)
        x1 = self.conv1(x_bottle)
        x2 = self.conv2(x_bottle)
        x3 = self.conv3(x_bottle)
        x4 = self.conv_pool(self.pool(x))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.relu(self.norm(out))


# === Khối Inception + Residual ===
class InceptionResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inception = InceptionBlock1D(in_channels, out_channels)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels * 4, affine=True)
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res = self.residual(x)
        x = self.inception(x)
        x = x + res
        return self.relu(x)


# === Mô hình InceptionTime nhẹ hơn (feature cuối 512) + Attention ===
class InceptionTime1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block1 = InceptionResidualBlock1D(in_channels, 16)    # 16*4 = 64
        self.block2 = InceptionResidualBlock1D(64, 16)             # 16*4 = 64
        self.block3 = InceptionResidualBlock1D(64, 32)             # 32*4 = 128
        self.block4 = InceptionResidualBlock1D(128, 32)            # 128
        self.block5 = InceptionResidualBlock1D(128, 64)            # 64*4=256
        self.block6 = InceptionResidualBlock1D(256, 64)            # 256
        self.block7 = InceptionResidualBlock1D(256, 128)           # 128*4=512
        self.block8 = InceptionResidualBlock1D(512, 128)           # 512

        self.attention = Attention(input_dim=512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, feat_dim)
        x, _ = self.attention(x)
        out = self.fc(x)
        return out
