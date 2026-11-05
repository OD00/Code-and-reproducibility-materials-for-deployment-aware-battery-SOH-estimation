import torch.nn as nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# --- 多尺度特征提取模块 ---
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1d, self).__init__()
        # 不同的感受野捕捉曲线的缓变和突变
        self.conv1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        # 拼接三个尺度的特征
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out) # 赋予注意力权重
        return out




class CnnLSTMControl(nn.Module):
    def __init__(
        self,
        d_model=64,
        hidden_size=128,
        output_dim=1,
        use_duration_fusion=False,
        use_simple_cnn_frontend=False,
    ):
        super(CnnLSTMControl, self).__init__()
        self.use_simple_cnn_frontend = bool(use_simple_cnn_frontend)
        if self.use_simple_cnn_frontend:
            # Deployment ablation: use exactly the same lightweight cycle
            # frontend as CnnTransformer and CnnMambaSimpleCNN.  The default
            # remains the historical multi-scale CNN+SE path so existing
            # checkpoints retain their original architecture.
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            frontend_channels = 64
        else:
            self.feature_extractor = nn.Sequential(
                MultiScaleConv1d(in_channels=2, out_channels=32),
                nn.MaxPool1d(2),
                MultiScaleConv1d(in_channels=32, out_channels=64),
                nn.MaxPool1d(2),
                MultiScaleConv1d(in_channels=64, out_channels=128),
                # 自适应池化：不管输入点数，最后都池化为 1 个点
                nn.AdaptiveAvgPool1d(1),
            )
            frontend_channels = 128
        self.proj_to_seq = nn.Linear(frontend_channels, d_model)
        self.use_duration_fusion = bool(use_duration_fusion)
        if self.use_duration_fusion:
            self.duration_encoder = nn.Linear(1, d_model)
            self.duration_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.duration_encoder = None
            self.register_parameter("duration_gate", None)
        
        # 将 Mamba 替换为 LSTM
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_size, 
                            num_layers=2, batch_first=True)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x1, x2, duration=None):
        B, S, P = x1.shape
        x = torch.stack([x1, x2], dim=2).view(B * S, 2, P)
        
        # CNN 提取特征
        x = self.feature_extractor(x).view(B * S, -1)
        x = self.proj_to_seq(x).view(B, S, -1)
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
        
        # LSTM 处理序列
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步进行预测 (方案A)
        out = self.fc_out(lstm_out[:, -1, :])
        return out
