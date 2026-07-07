# models.py
import torch
import torch.nn as nn

class TransformerDQN(nn.Module):
    def __init__(self, input_dim=1875, seq_len=5, d_model=128, nhead=4, num_layers=2, output_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc_out = nn.Linear(64, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.transformer(x)         # (batch, seq_len, d_model)
        x = x[:, -1, :]                 # 取最后一个时间步 (batch, d_model)
        x = torch.relu(self.fc1(x))
        return self.fc_out(x)           # (batch, output_dim)