import torch
import torch.nn as nn

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1d(nn.Module):
    def __init__(self, output_dim=1, use_duration_fusion=False):
        super(ResNet1d, self).__init__()
        # 将 S 和 P 融合，作为超长的一维信号处理，或者保持 Cycle-wise
        # 这里推荐保持 Cycle-wise 提取后，对 S 维度再做一次 ResNet
        self.cycle_feature = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.use_duration_fusion = bool(use_duration_fusion)
        if self.use_duration_fusion:
            self.duration_encoder = nn.Linear(1, 32)
            self.duration_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.duration_encoder = None
            self.register_parameter("duration_gate", None)
        
        self.layer1 = ResBlock1d(32, 64, stride=1)
        self.layer2 = ResBlock1d(64, 128, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x1, x2, duration=None):
        B, S, P = x1.shape
        x = torch.stack([x1, x2], dim=2).view(B * S, 2, P)
        
        # 1. 提取每个循环的特征 -> [B*S, 32, 1]
        x = self.cycle_feature(x).view(B, S, 32)
        if self.use_duration_fusion:
            if duration is None:
                raise ValueError("duration is required when use_duration_fusion=True")
            if duration.ndim == 2:
                duration = duration.unsqueeze(-1)
            if duration.shape != (B, S, 1):
                raise ValueError(f"duration must have shape {(B, S, 1)}, got {tuple(duration.shape)}")
            x = x + self.duration_gate * self.duration_encoder(
                duration.to(device=x.device, dtype=x.dtype)
            )
        
        # 2. 将 S 维度视为序列维度，进行残差卷积 -> [B, 32, S]
        x = x.transpose(1, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 3. 池化并输出
        x = self.avgpool(x).view(B, -1)
        out = self.fc(x)
        return out
