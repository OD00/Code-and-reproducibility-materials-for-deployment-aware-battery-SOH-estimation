import torch
import torch.nn as nn
from mamba_ssm import Mamba


# CnnMambaImproved 的全局实验开关。
# False: 原版双输入模型（电压、电流），保持旧检查点参数兼容。
# True:  修改版三输入模型（电压、电流、已归一化时长）。
# 也可在构造模型时通过 use_duration_fusion=True/False 显式覆盖。
USE_DURATION_FUSION = False


# --- 引入 SE 注意力机制 ---
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

# 解码器
class AttentionDecoder(nn.Module):
    def __init__(self, mamba_dim, output_dim):
        super().__init__()
        # 学习一个 Query 向量（代表"我需要什么信息来预测目标"）
        self.query = nn.Parameter(torch.randn(1, 1, mamba_dim))
        
        # 注意力计算层
        self.attn = nn.MultiheadAttention(embed_dim=mamba_dim, num_heads=4, batch_first=True)
        
        # 现代版的 MLP
        self.swiglu_linear1 = nn.Linear(mamba_dim, mamba_dim * 2)
        self.swiglu_linear2 = nn.Linear(mamba_dim, mamba_dim * 2)
        self.silu = nn.SiLU()
        self.output_proj = nn.Linear(mamba_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, mamba_memory):
        # mamba_memory 形状: [B, S, mamba_dim] (这就是你的记忆库)
        B = mamba_memory.size(0)
        
        # 扩展 Query 以匹配 Batch Size: [B, 1, mamba_dim]
        q = self.query.expand(B, -1, -1)
        
        # Cross-Attention: Query 去查询 Mamba Memory
        # context 形状: [B, 1, mamba_dim]
        context, attn_weights = self.attn(query=q, key=mamba_memory, value=mamba_memory)
        context = context.squeeze(1) # [B, mamba_dim]
        
        # 经过 SwiGLU 解码输出
        # SwiGLU = (xW1 * SiLU(xW2))
        gate = self.silu(self.swiglu_linear1(context))
        up = self.swiglu_linear2(context)
        out = gate * up
        
        out = self.dropout(out)
        out = self.output_proj(out) # [B, output_dim]
        
        return out, attn_weights # 返回注意力权重还可以用于可视化，解释模型为什么这么预测



# ============================================================
# 消融实验对照组
# ============================================================

# --- V1: CNN + SE（无 Mamba，无 Attention Decoder）---
class CnnSE(nn.Module):
    """基线模型：仅使用多尺度 CNN + SE 注意力，无时序建模"""
    def __init__(self, output_dim=1):
        super(CnnSE, self).__init__()

        self.feature_extractor = nn.Sequential(
            MultiScaleConv1d(in_channels=2, out_channels=32),
            nn.MaxPool1d(2),
            MultiScaleConv1d(in_channels=32, out_channels=64),
            nn.MaxPool1d(2),
            MultiScaleConv1d(in_channels=64, out_channels=128),
            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x1, x2):
        B, S, P = x1.shape
        x = torch.stack([x1, x2], dim=2)       # [B, S, 2, P]
        x = x.view(B * S, 2, P)                # [B*S, 2, P]
        x = self.feature_extractor(x)           # [B*S, 128, 1]
        x = x.view(B * S, -1)                  # [B*S, 128]
        out = self.regressor(x)                 # [B*S, output_dim]
        out = out.view(B, S, -1)                # [B, S, output_dim]
        return out, None


# --- V2: CNN + SE + Mamba（无 Attention Decoder，简单线性输出）---
class CnnMambaSimple(nn.Module):
    """消融：CNN + SE + Mamba，但用简单线性层替代 Attention Decoder"""
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, output_dim=1):
        super(CnnMambaSimple, self).__init__()

        self.mamba_dim = d_model

        self.feature_extractor = nn.Sequential(
            MultiScaleConv1d(in_channels=2, out_channels=32),
            nn.MaxPool1d(2),
            MultiScaleConv1d(in_channels=32, out_channels=64),
            nn.MaxPool1d(2),
            MultiScaleConv1d(in_channels=64, out_channels=128),
            nn.AdaptiveAvgPool1d(4)
        )

        self.proj_to_mamba = nn.Sequential(
            nn.Linear(128 * 4, self.mamba_dim),
            nn.LayerNorm(self.mamba_dim),
            nn.SiLU()
        )

        self.mamba = Mamba(
            d_model=self.mamba_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.norm_out = nn.LayerNorm(self.mamba_dim)

        # 简单线性输出，替代 Attention Decoder
        self.regressor = nn.Sequential(
            nn.Linear(self.mamba_dim, 32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x1, x2):
        B, S, P = x1.shape
        x = torch.stack([x1, x2], dim=2)
        x = x.view(B * S, 2, P)
        x = self.feature_extractor(x)
        x = x.view(B * S, -1)
        x = self.proj_to_mamba(x)
        x = x.view(B, S, self.mamba_dim)
        x = x + self.mamba(x)           # 残差连接
        x = self.norm_out(x)
        out = self.regressor(x)                 # [B, S, output_dim]
        return out, None


# --- V3: CNN + SE + LSTM + Attention（用 LSTM 替代 Mamba）---
class CnnLstmAttention(nn.Module):
    """消融：用 LSTM 替代 Mamba，保留 Attention Decoder 以公平比较"""
    def __init__(self, d_model=64, hidden_size=64, num_layers=1, output_dim=1):
        super(CnnLstmAttention, self).__init__()

        self.lstm_dim = d_model

        self.feature_extractor = nn.Sequential(
            MultiScaleConv1d(in_channels=2, out_channels=32),
            nn.MaxPool1d(2),
            MultiScaleConv1d(in_channels=32, out_channels=64),
            nn.MaxPool1d(2),
            MultiScaleConv1d(in_channels=64, out_channels=128),
            nn.AdaptiveAvgPool1d(4)
        )

        self.proj_to_lstm = nn.Sequential(
            nn.Linear(128 * 4, self.lstm_dim),
            nn.LayerNorm(self.lstm_dim),
            nn.SiLU()
        )

        # LSTM 替代 Mamba
        self.lstm = nn.LSTM(
            input_size=self.lstm_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.norm_out = nn.LayerNorm(hidden_size)

        # 保留 Attention Decoder 以公平比较
        self.decoder = AttentionDecoder(mamba_dim=hidden_size, output_dim=output_dim)

    def forward(self, x1, x2):
        B, S, P = x1.shape
        x = torch.stack([x1, x2], dim=2)
        x = x.view(B * S, 2, P)
        x = self.feature_extractor(x)
        x = x.view(B * S, -1)
        x = self.proj_to_lstm(x)
        x = x.view(B, S, self.lstm_dim)
        x, _ = self.lstm(x)
        x = self.norm_out(x)
        out, attn_weights = self.decoder(x)
        return out, attn_weights


# --- V4: Full Model（CNN + SE + Mamba + Attention Decoder）---
class CnnMambaImproved(nn.Module):
    # 移除了 seq_len 和 sample_points 参数，因为现在的网络完全动态适应
    def __init__(
        self,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        output_dim=1,
        num_mamba_layers=1,
        use_duration_fusion=None,
        use_simple_cnn_frontend=False,
    ):
        super(CnnMambaImproved, self).__init__()

        if num_mamba_layers < 1:
            raise ValueError("num_mamba_layers must be at least 1.")

        self.mamba_dim = d_model
        self.num_mamba_layers = int(num_mamba_layers)
        self.use_simple_cnn_frontend = bool(use_simple_cnn_frontend)
        self.use_duration_fusion = (
            USE_DURATION_FUSION
            if use_duration_fusion is None
            else bool(use_duration_fusion)
        )

        # --- [阶段 1 & 2] 特征提取层 (针对单循环曲线) ---
        if self.use_simple_cnn_frontend:
            # Deployment ablation: match the CnnTransformer cycle frontend.
            # The default remains the historical multi-scale CNN+SE path so
            # existing checkpoints and formal experiment definitions do not
            # change silently.
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
                # 自适应池化：压缩为 1 个点，与 CnnLSTMControl 保持一致
                nn.AdaptiveAvgPool1d(1),
            )
            frontend_channels = 128

        # 将 CNN 提取的特征投影到 Mamba 需要的维度
        self.proj_to_mamba = nn.Sequential(
            nn.Linear(frontend_channels, self.mamba_dim),
            nn.LayerNorm(self.mamba_dim),
            nn.SiLU()
        )

        # 时长是“单循环级”特征，不复制到原始采样维度，也不经过 CNN 池化。
        # 修改版将 [B, S, 1] 编码为 [B, S, d_model]，再以可学习门控
        # 残差融合到每个循环 token，之后交给 Mamba 建模跨循环依赖。
        # 关闭开关时不创建新参数，使原版 state_dict 键保持不变。
        if self.use_duration_fusion:
            self.duration_encoder = nn.Linear(1, self.mamba_dim)
            self.duration_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.duration_encoder = None
            self.register_parameter("duration_gate", None)

        # --- [阶段 3] MAMBA 时序预测 ---
        self.mamba = Mamba(
            d_model=self.mamba_dim,  
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        # Keep the first block under the historical ``mamba.*`` key so the
        # one-layer default remains checkpoint-compatible with earlier runs.
        self.extra_mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.mamba_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(self.num_mamba_layers - 1)
            ]
        )
        
        # --- [阶段 4] 输出层 ---
        self.norm_out = nn.LayerNorm(self.mamba_dim)
        
        self.decoder = AttentionDecoder(mamba_dim=self.mamba_dim, output_dim=output_dim)

    def forward(self, x1, x2, duration=None):
        """Estimate capacity/SOH from cycle windows.

        Args:
            x1: Voltage tensor shaped [B, S, P].
            x2: Current tensor shaped [B, S, P].
            duration: Training-split-normalized duration shaped [B, S, 1]
                (or [B, S]). Required only when duration fusion is enabled.
                It is ignored in original-model mode so the same training loop
                can be used for a controlled ablation.
        """
        # 此时 B: Batch, S: 任意序列长度, P: 任意采样点数
        B, S, P = x1.shape 
        
        # x: [B, S, 2, P]
        x = torch.stack([x1, x2], dim=2) 
        
        # 折叠 Batch 和 Seq_len: [B * S, 2, P]
        x = x.view(B * S, 2, P)
        
        # 输出形状: [B * S, frontend_channels, 1]
        x = self.feature_extractor(x)
        x = x.view(B * S, -1)
        
        # 投影到 Mamba 维度并恢复序列形状 -> [B, S, 64]
        x = self.proj_to_mamba(x) 
        x = x.view(B, S, self.mamba_dim)

        if self.use_duration_fusion:
            if duration is None:
                raise ValueError(
                    "duration is required when use_duration_fusion=True. "
                    "Expected shape [B, S, 1] or [B, S]."
                )
            if duration.ndim == 2:
                duration = duration.unsqueeze(-1)
            if duration.ndim != 3 or duration.shape != (B, S, 1):
                raise ValueError(
                    "duration must have shape [B, S, 1] or [B, S]; "
                    f"got {tuple(duration.shape)} for voltage shape {tuple(x1.shape)}."
                )
            duration = duration.to(device=x.device, dtype=x.dtype)
            duration_features = self.duration_encoder(duration)
            x = x + self.duration_gate * duration_features
        
        # MAMBA 处理跨 Cycle 依赖 -> 输出: [B, S, 64]
        x = self.mamba(x)
        for mamba_layer in self.extra_mamba_layers:
            x = mamba_layer(x)
        x = self.norm_out(x)
        
        # 【修改点3】：处理动态的序列维度 S
        out, attn_weights = self.decoder(x)
        
        return out , attn_weights


