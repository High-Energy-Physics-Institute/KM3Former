import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KM3Former(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
        max_hits=512,
        pairwise_neighbors=32,
        pairwise_hidden_dim=None,
    ):
        super().__init__()
        self.model_dim = model_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.positional_encoding = PositionalEncoding(
            d_model=model_dim,
            dropout=dropout,
            max_len=max_hits,
        )
        self.pairwise_bias = PairwiseAttentionBias(
            num_heads=num_heads,
            hidden_dim=pairwise_hidden_dim or model_dim,
            time_neighbors=pairwise_neighbors,
            spatial_neighbors=pairwise_neighbors,
        )
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.pooler = MaskedAttentionPooling(model_dim=model_dim)
        self.fc_out = nn.Linear(model_dim, 3)
        self.quality_head = nn.Linear(model_dim, 1)

    def forward(self, src, raw_hits=None, padding_mask=None, return_quality=False):
        if padding_mask is None:
            padding_mask = torch.zeros(
                src.shape[:2],
                dtype=torch.bool,
                device=src.device,
            )

        if raw_hits is None:
            raw_hits = src

        valid_mask = (~padding_mask).unsqueeze(-1)
        src = self.embedding(src) * math.sqrt(self.model_dim)
        src = self.positional_encoding(src)
        src = src * valid_mask

        attention_bias = self.pairwise_bias(
            raw_hits=raw_hits,
            padding_mask=padding_mask,
        )

        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(
                memory,
                src_mask=attention_bias,
                src_key_padding_mask=padding_mask,
            )

        memory = self.final_norm(memory) * valid_mask
        pooled = self.pooler(memory, padding_mask=padding_mask)
        direction = F.normalize(self.fc_out(pooled), dim=-1)

        if return_quality:
            quality = torch.sigmoid(self.quality_head(pooled)).squeeze(-1)
            return direction, quality

        return direction


class MaskedAttentionPooling(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, memory, padding_mask=None):
        scores = self.score(memory).squeeze(-1)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, 0.0)

        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return torch.sum(memory * weights.unsqueeze(-1), dim=1)


class PairwiseAttentionBias(nn.Module):
    def __init__(self, num_heads, hidden_dim, time_neighbors=32, spatial_neighbors=32):
        super().__init__()
        self.num_heads = num_heads
        self.time_neighbors = time_neighbors
        self.spatial_neighbors = spatial_neighbors
        self.bias_mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, raw_hits, padding_mask=None):
        times = raw_hits[..., 0]
        positions = raw_hits[..., 1:4]
        directions = raw_hits[..., 4:7]

        delta_t = times.unsqueeze(2) - times.unsqueeze(1)
        delta_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        distance = torch.linalg.norm(delta_pos, dim=-1)

        safe_distance = distance.unsqueeze(-1).clamp(min=1e-6)
        direction_to_neighbor = delta_pos / safe_distance
        orientation_source = (
            directions.unsqueeze(2) * direction_to_neighbor
        ).sum(dim=-1)
        orientation_target = (
            directions.unsqueeze(1) * -direction_to_neighbor
        ).sum(dim=-1)

        pair_features = torch.stack(
            [
                delta_t,
                delta_t.abs(),
                delta_pos[..., 0],
                delta_pos[..., 1],
                delta_pos[..., 2],
                distance,
                orientation_source,
                orientation_target,
            ],
            dim=-1,
        )
        attention_bias = self.bias_mlp(pair_features).permute(0, 3, 1, 2)

        allowed_pairs = self.build_sparse_attention_mask(
            delta_t=delta_t,
            distance=distance,
            padding_mask=padding_mask,
        )
        attention_bias = attention_bias.masked_fill(
            ~allowed_pairs.unsqueeze(1),
            torch.finfo(attention_bias.dtype).min,
        )

        batch_size, _, num_hits, _ = attention_bias.shape
        return attention_bias.reshape(batch_size * self.num_heads, num_hits, num_hits)

    def build_sparse_attention_mask(self, delta_t, distance, padding_mask=None):
        batch_size, num_hits, _ = delta_t.shape
        device = delta_t.device

        time_rank = delta_t.abs()
        distance_rank = distance

        time_k = min(self.time_neighbors, num_hits)
        spatial_k = min(self.spatial_neighbors, num_hits)

        time_indices = torch.topk(
            time_rank,
            k=time_k,
            largest=False,
            dim=-1,
        ).indices
        spatial_indices = torch.topk(
            distance_rank,
            k=spatial_k,
            largest=False,
            dim=-1,
        ).indices

        allowed_time = torch.zeros(
            batch_size,
            num_hits,
            num_hits,
            dtype=torch.bool,
            device=device,
        )
        allowed_space = torch.zeros_like(allowed_time)
        allowed_time.scatter_(-1, time_indices, True)
        allowed_space.scatter_(-1, spatial_indices, True)

        allowed_pairs = allowed_time | allowed_space
        if padding_mask is not None:
            valid_keys = (~padding_mask).unsqueeze(1)
            allowed_pairs = allowed_pairs & valid_keys

        diagonal = torch.eye(num_hits, dtype=torch.bool, device=device).unsqueeze(0)
        return allowed_pairs | diagonal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
