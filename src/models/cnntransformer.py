import torch
import torch.nn as nn


class CnnTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_dim=1,
        use_duration_fusion=False,
    ):
        super(CnnTransformer, self).__init__()
        
        # 复用你的 MultiScaleConv1d 逻辑（为了公平对比，可以保持 CNN 部分一致）
        # 这里为了演示简洁，简化了 CNN 部分，建议实验时保持与原模型 CNN 一致
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.proj = nn.Linear(64, d_model)
        self.use_duration_fusion = bool(use_duration_fusion)
        if self.use_duration_fusion:
            self.duration_encoder = nn.Linear(1, d_model)
            self.duration_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.duration_encoder = None
            self.register_parameter("duration_gate", None)
        
        # Transformer 需要位置编码 (Positional Encoding)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 500, d_model)) # 假设最大序列长度500
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x1, x2, duration=None):
        B, S, P = x1.shape
        x = torch.stack([x1, x2], dim=2).view(B * S, 2, P)
        
        x = self.feature_extractor(x).view(B * S, -1)
        x = self.proj(x).view(B, S, -1)
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
        
        # 加入位置信息
        x = x + self.pos_embedding[:, :S, :]
        
        # Transformer 建模
        x = self.transformer(x)
        out = self.fc_out(x[:, -1, :]) # 取最后一个时间步
        return out
