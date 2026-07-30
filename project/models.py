import torch
import torch.nn as nn

class CrossAttentionDQN(nn.Module):
    def __init__(self, d_model=64, nhead=4, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # 分支投影
        self.ag_proj = nn.Linear(3, d_model)
        self.obs_proj = nn.Linear(8, d_model)

        # 位置编码（可学习）
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # Self-Attention 编码层
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=d_model*2, batch_first=True)
        self.ag_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.obs_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Cross-Attention: AG 查 OBS
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x):
        # x: (batch, seq_len, 11)
        batch, seq_len, _ = x.shape

        ag = x[:, :, :3]          # (batch, seq_len, 3)
        obs = x[:, :, 3:]         # (batch, seq_len, 8)

        # 投影
        ag = self.ag_proj(ag)     # (batch, seq_len, d_model)
        obs = self.obs_proj(obs)  # (batch, seq_len, d_model)

        # 加位置编码
        pos = self.pos_embed(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        ag = ag + pos
        obs = obs + pos

        # 各自的 Self-Attention
        ag = self.ag_encoder(ag)    # (batch, seq_len, d_model)
        obs = self.obs_encoder(obs) # (batch, seq_len, d_model)

        # Cross-Attention: AG 作为 Query，OBS 作为 Key/Value
        fused, _ = self.cross_attn(query=ag, key=obs, value=obs)

        # 残差连接
        fused = fused + ag

        # 取最后时间步
        h = fused[:, -1, :]         # (batch, d_model)

        return self.fc(h)           # (batch, 8)
