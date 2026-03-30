import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TOT_FEATURE_INDEX = 7
DEFAULT_PROPAGATION_SPEED = 0.225


def resolve_neighbor_counts(
    pairwise_neighbors=None,
    time_neighbors=None,
    spatial_neighbors=None,
):
    shared_neighbors = 32 if pairwise_neighbors is None else pairwise_neighbors
    resolved_time_neighbors = (
        shared_neighbors if time_neighbors is None else time_neighbors
    )
    resolved_spatial_neighbors = (
        shared_neighbors if spatial_neighbors is None else spatial_neighbors
    )
    return resolved_time_neighbors, resolved_spatial_neighbors


def gather_token_neighbors(token_tensor, neighbor_indices):
    batch_size, num_hits, _ = neighbor_indices.shape
    head_count = token_tensor.size(1)
    head_dim = token_tensor.size(-1)

    flattened_tokens = token_tensor.permute(0, 2, 1, 3).reshape(
        batch_size * token_tensor.size(2),
        head_count,
        head_dim,
    )
    batch_offsets = (
        torch.arange(batch_size, device=neighbor_indices.device).view(batch_size, 1, 1)
        * token_tensor.size(2)
    )
    flattened_indices = (neighbor_indices + batch_offsets).reshape(-1)
    gathered = flattened_tokens[flattened_indices].reshape(
        batch_size,
        num_hits,
        neighbor_indices.size(-1),
        head_count,
        head_dim,
    )
    return gathered.permute(0, 3, 1, 2, 4)


def gather_hit_neighbors(hit_tensor, neighbor_indices):
    batch_size, num_hits, _ = neighbor_indices.shape
    feature_dim = hit_tensor.size(-1)

    flattened_hits = hit_tensor.reshape(batch_size * hit_tensor.size(1), feature_dim)
    batch_offsets = (
        torch.arange(batch_size, device=neighbor_indices.device).view(batch_size, 1, 1)
        * hit_tensor.size(1)
    )
    flattened_indices = (neighbor_indices + batch_offsets).reshape(-1)
    return flattened_hits[flattened_indices].reshape(
        batch_size,
        num_hits,
        neighbor_indices.size(-1),
        feature_dim,
    )


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
        time_neighbors=None,
        spatial_neighbors=None,
        pairwise_hidden_dim=None,
        target_dim=3,
        target_kind="vector_regression",
        position_encoding="sinusoidal",
        time_neighborhood_mode="index_window",
        time_radius=None,
        pairwise_time_transform="raw",
        pairwise_distance_transform="raw",
        pairwise_feature_version="v1",
        exclude_self_from_spatial_knn=False,
        deduplicate_neighbors=False,
        pooling="attention",
        count_head="multiclass",
        pairwise_position_scale=1.0,
        propagation_speed=DEFAULT_PROPAGATION_SPEED,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.target_dim = target_dim
        self.target_kind = target_kind
        self.count_head = count_head
        self.time_neighbors, self.spatial_neighbors = resolve_neighbor_counts(
            pairwise_neighbors=pairwise_neighbors,
            time_neighbors=time_neighbors,
            spatial_neighbors=spatial_neighbors,
        )
        if count_head not in {"multiclass", "ordinal_coral"}:
            raise ValueError(f"Unsupported count_head: {count_head!r}")
        if target_kind != "multiclass" and count_head != "multiclass":
            raise ValueError("count_head is only supported for multiclass targets.")

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        if position_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(
                d_model=model_dim,
                dropout=dropout,
                max_len=max_hits,
            )
        elif position_encoding == "none":
            self.positional_encoding = NoOpPositionalEncoding()
        else:
            raise ValueError(f"Unsupported position_encoding: {position_encoding!r}")
        self.encoder_layers = nn.ModuleList(
            [
                SparseNeighborhoodEncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    pairwise_hidden_dim=pairwise_hidden_dim or model_dim,
                    time_neighbors=self.time_neighbors,
                    spatial_neighbors=self.spatial_neighbors,
                    time_neighborhood_mode=time_neighborhood_mode,
                    time_radius=time_radius,
                    pairwise_time_transform=pairwise_time_transform,
                    pairwise_distance_transform=pairwise_distance_transform,
                    pairwise_feature_version=pairwise_feature_version,
                    exclude_self_from_spatial_knn=exclude_self_from_spatial_knn,
                    deduplicate_neighbors=deduplicate_neighbors,
                    pairwise_position_scale=pairwise_position_scale,
                    propagation_speed=propagation_speed,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        if pooling == "attention":
            self.pooler = MaskedAttentionPooling(model_dim=model_dim)
            pooled_dim = model_dim
        elif pooling == "attention_mean_concat":
            self.pooler = AttentionMeanConcatPooling(model_dim=model_dim)
            pooled_dim = model_dim * 2
        elif pooling == "attention_mean_sum_count_concat":
            self.pooler = AttentionMeanSumCountConcatPooling(model_dim=model_dim)
            pooled_dim = model_dim * 3 + 1
        else:
            raise ValueError(f"Unsupported pooling: {pooling!r}")

        if pooling in {
            "attention_mean_concat",
            "attention_mean_sum_count_concat",
        }:
            self.fc_out = self._build_projected_output_head(
                pooled_dim=pooled_dim,
                model_dim=model_dim,
            )
            self.quality_head = nn.Sequential(
                nn.Linear(pooled_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, 1),
            )
        else:
            if self.target_kind == "multiclass" and self.count_head == "ordinal_coral":
                self.fc_out = CoralOrdinalHead(pooled_dim, target_dim)
            else:
                self.fc_out = nn.Linear(pooled_dim, target_dim)
            self.quality_head = nn.Linear(pooled_dim, 1)

    def _build_projected_output_head(self, pooled_dim, model_dim):
        if self.target_kind == "multiclass" and self.count_head == "ordinal_coral":
            return nn.Sequential(
                nn.Linear(pooled_dim, model_dim),
                nn.GELU(),
                CoralOrdinalHead(model_dim, self.target_dim),
            )
        return nn.Sequential(
            nn.Linear(pooled_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, self.target_dim),
        )

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

        memory = src
        attention_context = None
        if self.encoder_layers:
            attention_context = self.encoder_layers[0].build_attention_context(
                raw_hits=raw_hits,
                padding_mask=padding_mask,
            )
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(
                memory,
                raw_hits=raw_hits,
                padding_mask=padding_mask,
                attention_context=attention_context,
            )
            memory = memory * valid_mask

        memory = self.final_norm(memory) * valid_mask
        pooled = self.pooler(memory, padding_mask=padding_mask)
        prediction = self.fc_out(pooled)
        if self.target_kind == "vector_regression":
            prediction = F.normalize(prediction, dim=-1)

        if return_quality and self.target_kind == "vector_regression":
            quality = torch.sigmoid(self.quality_head(pooled)).squeeze(-1)
            return prediction, quality

        return prediction


class SparseNeighborhoodEncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        dim_feedforward,
        dropout,
        pairwise_hidden_dim,
        time_neighbors,
        spatial_neighbors,
        time_neighborhood_mode,
        time_radius,
        pairwise_time_transform,
        pairwise_distance_transform,
        pairwise_feature_version,
        exclude_self_from_spatial_knn,
        deduplicate_neighbors,
        pairwise_position_scale,
        propagation_speed,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attention = SparseNeighborhoodSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            pairwise_hidden_dim=pairwise_hidden_dim,
            time_neighbors=time_neighbors,
            spatial_neighbors=spatial_neighbors,
            time_neighborhood_mode=time_neighborhood_mode,
            time_radius=time_radius,
            pairwise_time_transform=pairwise_time_transform,
            pairwise_distance_transform=pairwise_distance_transform,
            pairwise_feature_version=pairwise_feature_version,
            exclude_self_from_spatial_knn=exclude_self_from_spatial_knn,
            deduplicate_neighbors=deduplicate_neighbors,
            pairwise_position_scale=pairwise_position_scale,
            propagation_speed=propagation_speed,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.linear1 = nn.Linear(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim)
        self.dropout2 = nn.Dropout(dropout)

    def build_attention_context(self, raw_hits, padding_mask):
        return self.self_attention.build_attention_context(
            raw_hits=raw_hits,
            padding_mask=padding_mask,
        )

    def forward(self, src, raw_hits, padding_mask, attention_context=None):
        src = src + self.dropout1(
            self.self_attention(
                self.norm1(src),
                raw_hits=raw_hits,
                padding_mask=padding_mask,
                attention_context=attention_context,
            )
        )
        src = src + self._feed_forward_block(self.norm2(src))
        return src

    def _feed_forward_block(self, src):
        src = self.linear1(src)
        src = F.gelu(src)
        src = self.dropout(src)
        src = self.linear2(src)
        return self.dropout2(src)


class SparseNeighborhoodSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        dropout,
        pairwise_hidden_dim,
        time_neighbors=32,
        spatial_neighbors=32,
        spatial_chunk_size=64,
        time_neighborhood_mode="index_window",
        time_radius=None,
        pairwise_time_transform="raw",
        pairwise_distance_transform="raw",
        pairwise_feature_version="v1",
        exclude_self_from_spatial_knn=False,
        deduplicate_neighbors=False,
        pairwise_position_scale=1.0,
        propagation_speed=DEFAULT_PROPAGATION_SPEED,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if time_neighborhood_mode not in {"index_window", "delta_t_radius"}:
            raise ValueError(
                f"Unsupported time_neighborhood_mode: {time_neighborhood_mode!r}"
            )
        if time_neighborhood_mode == "delta_t_radius" and time_radius is None:
            raise ValueError(
                "time_radius must be provided when using delta_t_radius neighborhoods."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.time_neighbors = time_neighbors
        self.spatial_neighbors = spatial_neighbors
        self.spatial_chunk_size = spatial_chunk_size
        self.time_neighborhood_mode = time_neighborhood_mode
        self.time_radius = time_radius
        self.exclude_self_from_spatial_knn = exclude_self_from_spatial_knn
        self.deduplicate_neighbors = deduplicate_neighbors

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.pairwise_bias = PairwiseAttentionBias(
            num_heads=num_heads,
            hidden_dim=pairwise_hidden_dim,
            time_transform=pairwise_time_transform,
            distance_transform=pairwise_distance_transform,
            feature_version=pairwise_feature_version,
            position_scale=pairwise_position_scale,
            propagation_speed=propagation_speed,
        )

    def build_attention_context(self, raw_hits, padding_mask=None):
        neighbor_indices, neighbor_mask = self.build_neighborhood_indices(
            raw_hits=raw_hits,
            padding_mask=padding_mask,
        )
        return {
            "neighbor_indices": neighbor_indices,
            "neighbor_mask": neighbor_mask,
            "pair_features": self.pairwise_bias.build_pair_features(
                raw_hits=raw_hits,
                neighbor_indices=neighbor_indices,
            ),
        }

    def forward(
        self,
        hidden_states,
        raw_hits,
        padding_mask=None,
        attention_context=None,
    ):
        batch_size, num_hits, _ = hidden_states.shape
        if attention_context is None:
            attention_context = self.build_attention_context(
                raw_hits=raw_hits,
                padding_mask=padding_mask,
            )
        neighbor_indices = attention_context["neighbor_indices"]
        neighbor_mask = attention_context["neighbor_mask"]
        query_valid = self._get_query_valid_mask(
            batch_size=batch_size,
            num_hits=num_hits,
            padding_mask=padding_mask,
            device=hidden_states.device,
        )

        query = self._project_heads(self.q_proj(hidden_states))
        key = self._project_heads(self.k_proj(hidden_states))
        value = self._project_heads(self.v_proj(hidden_states))

        gathered_key = gather_token_neighbors(key, neighbor_indices)
        gathered_value = gather_token_neighbors(value, neighbor_indices)

        logits = (
            torch.sum(query.unsqueeze(-2) * gathered_key, dim=-1)
            / math.sqrt(self.head_dim)
        )
        logits = logits + self.pairwise_bias(
            pair_features=attention_context["pair_features"]
        )

        attention_mask = neighbor_mask.unsqueeze(1)
        logits = logits.masked_fill(~attention_mask, -1e9)
        invalid_queries = ~neighbor_mask.any(dim=-1)
        logits = logits.masked_fill(invalid_queries.unsqueeze(1).unsqueeze(-1), 0.0)

        attention_weights = torch.softmax(logits, dim=-1)
        attention_weights = attention_weights * attention_mask
        attention_weights = self.attn_dropout(attention_weights)
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-6)

        context = torch.sum(attention_weights.unsqueeze(-1) * gathered_value, dim=-2)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, num_hits, -1)
        context = self.out_proj(context)
        return context * query_valid.unsqueeze(-1)

    def _project_heads(self, tensor):
        batch_size, num_hits, _ = tensor.shape
        return tensor.view(
            batch_size,
            num_hits,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def build_neighborhood_indices(self, raw_hits, padding_mask=None):
        batch_size, num_hits, _ = raw_hits.shape
        device = raw_hits.device

        if self.time_neighborhood_mode == "delta_t_radius":
            time_indices, time_mask = self._build_delta_t_radius_indices(raw_hits)
        else:
            time_indices, time_mask = self._build_time_window_indices(
                batch_size=batch_size,
                num_hits=num_hits,
                device=device,
            )
        spatial_indices, spatial_mask = self._build_spatial_knn_indices(
            raw_hits=raw_hits,
            padding_mask=padding_mask,
        )

        neighbor_indices = torch.cat([time_indices, spatial_indices], dim=-1)
        neighbor_mask = torch.cat([time_mask, spatial_mask], dim=-1)
        if self.deduplicate_neighbors:
            neighbor_mask = self._deduplicate_neighbor_mask(
                neighbor_indices=neighbor_indices,
                neighbor_mask=neighbor_mask,
            )

        if padding_mask is None:
            return neighbor_indices, neighbor_mask

        query_valid = (~padding_mask).unsqueeze(-1)
        key_valid = (~padding_mask).gather(
            1,
            neighbor_indices.reshape(batch_size, -1),
        ).reshape_as(neighbor_indices)
        neighbor_mask = neighbor_mask & query_valid & key_valid
        return neighbor_indices, neighbor_mask

    def _build_time_window_indices(self, batch_size, num_hits, device):
        offsets = torch.arange(
            -self.time_neighbors,
            self.time_neighbors + 1,
            device=device,
        )
        centers = torch.arange(num_hits, device=device).view(1, num_hits, 1)
        window_indices = centers + offsets.view(1, 1, -1)
        window_mask = (window_indices >= 0) & (window_indices < num_hits)
        window_indices = window_indices.clamp(0, num_hits - 1).expand(
            batch_size,
            -1,
            -1,
        )
        window_mask = window_mask.expand(batch_size, -1, -1)
        return window_indices, window_mask

    def _build_delta_t_radius_indices(self, raw_hits):
        batch_size, num_hits, _ = raw_hits.shape
        device = raw_hits.device
        hit_indices = torch.arange(num_hits, device=device).view(1, 1, num_hits).expand(
            batch_size,
            num_hits,
            -1,
        )
        hit_times = raw_hits[..., 0]
        delta_t = hit_times.unsqueeze(-1) - hit_times.unsqueeze(-2)
        return hit_indices, delta_t.abs() <= self.time_radius

    def _build_spatial_knn_indices(self, raw_hits, padding_mask=None):
        batch_size, num_hits, _ = raw_hits.shape
        device = raw_hits.device
        available_neighbors = (
            max(num_hits - 1, 0) if self.exclude_self_from_spatial_knn else num_hits
        )
        k_neighbors = min(self.spatial_neighbors, available_neighbors)

        if k_neighbors == 0:
            empty_indices = torch.empty(
                batch_size,
                num_hits,
                0,
                dtype=torch.long,
                device=device,
            )
            empty_mask = torch.empty(
                batch_size,
                num_hits,
                0,
                dtype=torch.bool,
                device=device,
            )
            return empty_indices, empty_mask

        positions = raw_hits[..., 1:4]
        all_indices = []
        all_masks = []

        for start in range(0, num_hits, self.spatial_chunk_size):
            end = min(start + self.spatial_chunk_size, num_hits)
            query_positions = positions[:, start:end]
            distances = torch.cdist(query_positions, positions)

            # Keep spatial neighborhoods distinct from the mandatory time-window self edge.
            if self.exclude_self_from_spatial_knn:
                query_axis = torch.arange(end - start, device=device)
                global_query_indices = torch.arange(start, end, device=device)
                distances[:, query_axis, global_query_indices] = float("inf")

            if padding_mask is not None:
                invalid_keys = padding_mask.unsqueeze(1).expand(-1, end - start, -1)
                distances = distances.masked_fill(invalid_keys, float("inf"))

            topk = torch.topk(
                distances,
                k=k_neighbors,
                largest=False,
                dim=-1,
            )
            all_indices.append(topk.indices)
            all_masks.append(torch.isfinite(topk.values))

        spatial_indices = torch.cat(all_indices, dim=1)
        spatial_mask = torch.cat(all_masks, dim=1)
        return spatial_indices, spatial_mask

    def _get_query_valid_mask(self, batch_size, num_hits, padding_mask, device):
        if padding_mask is None:
            return torch.ones(batch_size, num_hits, dtype=torch.bool, device=device)
        return ~padding_mask

    def _deduplicate_neighbor_mask(self, neighbor_indices, neighbor_mask):
        deduplicated = neighbor_mask.clone()
        for slot in range(neighbor_indices.size(-1)):
            if slot == 0:
                continue
            duplicate = (
                neighbor_indices[..., :slot] == neighbor_indices[..., slot : slot + 1]
            ).any(dim=-1)
            deduplicated[..., slot] = deduplicated[..., slot] & ~duplicate
        return deduplicated


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


class MaskedMeanPooling(nn.Module):
    def forward(self, memory, padding_mask=None):
        if padding_mask is None:
            return memory.mean(dim=1)

        valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=memory.dtype)
        return (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)


class AttentionMeanConcatPooling(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.attention_pool = MaskedAttentionPooling(model_dim=model_dim)
        self.mean_pool = MaskedMeanPooling()

    def forward(self, memory, padding_mask=None):
        attention_summary = self.attention_pool(memory, padding_mask=padding_mask)
        mean_summary = self.mean_pool(memory, padding_mask=padding_mask)
        return torch.cat([attention_summary, mean_summary], dim=-1)


class AttentionMeanSumCountConcatPooling(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.attention_pool = MaskedAttentionPooling(model_dim=model_dim)
        self.mean_pool = MaskedMeanPooling()

    def forward(self, memory, padding_mask=None):
        attention_summary = self.attention_pool(memory, padding_mask=padding_mask)
        mean_summary = self.mean_pool(memory, padding_mask=padding_mask)
        if padding_mask is None:
            sum_summary = memory.sum(dim=1)
            valid_counts = torch.full(
                (memory.size(0), 1),
                memory.size(1),
                dtype=memory.dtype,
                device=memory.device,
            )
        else:
            valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=memory.dtype)
            sum_summary = (memory * valid_mask).sum(dim=1)
            valid_counts = valid_mask.sum(dim=1)
        return torch.cat(
            [
                attention_summary,
                mean_summary,
                sum_summary,
                torch.log1p(valid_counts.clamp(min=0.0)),
            ],
            dim=-1,
        )


class PairwiseAttentionBias(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        time_transform="raw",
        distance_transform="raw",
        feature_version="v1",
        position_scale=1.0,
        propagation_speed=DEFAULT_PROPAGATION_SPEED,
    ):
        super().__init__()
        if time_transform not in {"raw", "signed_log1p"}:
            raise ValueError(f"Unsupported time_transform: {time_transform!r}")
        if distance_transform not in {"raw", "log1p"}:
            raise ValueError(
                f"Unsupported distance_transform: {distance_transform!r}"
            )
        if feature_version not in {"v1", "v2_physics"}:
            raise ValueError(f"Unsupported feature_version: {feature_version!r}")
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive.")
        if propagation_speed <= 0.0:
            raise ValueError("propagation_speed must be positive.")
        self.time_transform = time_transform
        self.distance_transform = distance_transform
        self.feature_version = feature_version
        self.position_scale = float(position_scale)
        self.propagation_speed = float(propagation_speed)
        self.bias_mlp = nn.Sequential(
            nn.Linear(8 if feature_version == "v1" else 13, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def build_pair_features(self, raw_hits, neighbor_indices):
        neighbor_hits = gather_hit_neighbors(raw_hits, neighbor_indices)

        source_times = raw_hits[..., 0].unsqueeze(-1)
        target_times = neighbor_hits[..., 0]
        source_positions = raw_hits[..., 1:4].unsqueeze(-2)
        target_positions = neighbor_hits[..., 1:4]
        source_directions = raw_hits[..., 4:7].unsqueeze(-2)
        target_directions = neighbor_hits[..., 4:7]

        delta_t = source_times - target_times
        delta_pos = source_positions - target_positions
        distance = torch.linalg.norm(delta_pos, dim=-1)
        delta_t_abs = delta_t.abs()

        if self.time_transform == "signed_log1p":
            signed_delta_t = torch.sign(delta_t) * torch.log1p(delta_t_abs)
            delta_t_magnitude = torch.log1p(delta_t_abs)
        else:
            signed_delta_t = delta_t
            delta_t_magnitude = delta_t_abs

        if self.distance_transform == "log1p":
            distance_feature = torch.log1p(distance)
        else:
            distance_feature = distance

        safe_distance = distance.unsqueeze(-1).clamp(min=1e-6)
        direction_to_neighbor = delta_pos / safe_distance
        orientation_source = (source_directions * direction_to_neighbor).sum(dim=-1)
        orientation_target = (target_directions * -direction_to_neighbor).sum(dim=-1)

        pair_features = [
            signed_delta_t,
            delta_t_magnitude,
            delta_pos[..., 0],
            delta_pos[..., 1],
            delta_pos[..., 2],
            distance_feature,
            orientation_source,
            orientation_target,
        ]
        if self.feature_version == "v2_physics":
            physical_delta_pos = delta_pos * self.position_scale
            physical_distance = torch.linalg.norm(physical_delta_pos, dim=-1)
            causal_residual = delta_t - (
                physical_distance / self.propagation_speed
            )
            source_log_tot = raw_hits[..., TOT_FEATURE_INDEX].unsqueeze(-1).expand_as(
                delta_t
            )
            target_log_tot = neighbor_hits[..., TOT_FEATURE_INDEX]
            pair_features.extend(
                [
                    causal_residual,
                    causal_residual.abs(),
                    source_log_tot,
                    target_log_tot,
                    source_log_tot - target_log_tot,
                ]
            )
        return torch.stack(pair_features, dim=-1)

    def forward(self, raw_hits=None, neighbor_indices=None, pair_features=None):
        if pair_features is None:
            if raw_hits is None or neighbor_indices is None:
                raise ValueError(
                    "PairwiseAttentionBias.forward requires pair_features or both raw_hits and neighbor_indices."
                )
            pair_features = self.build_pair_features(
                raw_hits=raw_hits,
                neighbor_indices=neighbor_indices,
            )
        return self.bias_mlp(pair_features).permute(0, 3, 1, 2)


class CoralOrdinalHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Ordinal classification requires at least two classes.")
        self.score = nn.Linear(input_dim, 1)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, inputs):
        return self.score(inputs) + self.bias.view(1, -1)


class NoOpPositionalEncoding(nn.Module):
    def forward(self, x):
        return x


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
