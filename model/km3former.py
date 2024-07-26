import math

import torch
import torch.nn as nn
from tqdm.notebook import tqdm


class KM3Former(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
    ):
        super(KM3Former, self).__init__()
        self.model_dim = model_dim

        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(model_dim, 3)

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(
            torch.tensor(self.model_dim, dtype=torch.float32)
        )
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src)
        output = self.fc_out(memory.mean(dim=1))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
