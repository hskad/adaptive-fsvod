"""Small trainable heads over frozen OWLv2 class features.

Fixed and adaptive runs differ in select_memory(). The optional confidence
gate controls the complete temporal residual and adds no learned parameters.
This is a reduced adaptation, not the authors' original model.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    input_dim: int = 512
    hidden_dim: int = 512
    attention_heads: int = 4
    dropout: float = 0.1
    memory_mode: str = "fixed"
    fixed_threshold: float = 0.7
    alpha: float = 1.0
    threshold_min: float = 0.5
    threshold_max: float = 0.95
    max_memory_per_class: int = 16
    fusion_gate: str = "none"
    gate_threshold: float = 0.7
    memory_weighting: str = "none"
    feature_centering: bool = False

    def __post_init__(self):
        if self.memory_mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("memory_mode must be none, fixed, or adaptive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0 <= self.threshold_min <= self.threshold_max <= 1:
            raise ValueError("Invalid adaptive threshold bounds")
        if not 0 <= self.fixed_threshold <= 1 or self.max_memory_per_class < 1:
            raise ValueError("Invalid memory parameters")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if self.fusion_gate not in {"none", "confidence"}:
            raise ValueError("fusion_gate must be none or confidence")
        if not 0 <= self.gate_threshold < 1:
            raise ValueError("gate_threshold must be in [0, 1)")
        if self.memory_weighting not in {"none", "objectness"}:
            raise ValueError("memory_weighting must be none or objectness")
        if not isinstance(self.feature_centering, bool):
            raise ValueError("feature_centering must be a boolean")


def confidence_gate(probabilities, threshold):
    """Detached foreground confidence margin; background winners get no update."""
    scores = probabilities[:, :-1].max(-1).values
    foreground_wins = probabilities.argmax(-1) < probabilities.shape[-1] - 1
    return ((scores - threshold) / (1 - threshold)).clamp(0, 1) * foreground_wins


def select_memory(probabilities, config):
    """Background is the final column. Return unique, capped token indices."""
    foreground = probabilities[:, :-1]
    if config.memory_mode == "adaptive":
        thresholds = (foreground.mean(0) + config.alpha * foreground.std(0, unbiased=False))
        thresholds = thresholds.clamp(config.threshold_min, config.threshold_max)
    else:
        thresholds = foreground.new_full((foreground.shape[1],), config.fixed_threshold)
    if config.memory_mode == "none":
        return torch.empty(0, dtype=torch.long, device=probabilities.device), thresholds
    winners = probabilities.argmax(-1)
    selected = []
    for category in range(foreground.shape[1]):
        indices = torch.where((winners == category) & (foreground[:, category] > thresholds[category]))[0]
        if indices.numel():
            order = torch.argsort(foreground[indices, category], descending=True, stable=True)
            selected.append(indices[order[:config.max_memory_per_class]])
    if selected:
        return torch.cat(selected), thresholds
    return torch.empty(0, dtype=torch.long, device=probabilities.device), thresholds


class TemporalDetector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.feature_centering:
            self.register_buffer("feature_center", torch.zeros(cfg.input_dim))
        self.project = nn.Identity() if cfg.input_dim == cfg.hidden_dim else nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.attention = nn.MultiheadAttention(cfg.hidden_dim, cfg.attention_heads,
                                             dropout=cfg.dropout, batch_first=True)
        self.memory_scale = nn.Parameter(torch.tensor(0.1))
        self.ffn = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.class_head = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.box_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(),
                                     nn.Linear(cfg.hidden_dim, 4))
        self.background_head = nn.Linear(cfg.hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Preserve the public detector's semantic features and boxes at initialization.
        for layer in [self.ffn[-1], self.class_head, self.box_head[-1], self.background_head]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def class_features(self, features):
        return F.normalize(features + self.class_head(features), dim=-1)

    def input_features(self, features):
        if self.config.feature_centering:
            features = F.normalize(features, dim=-1) - self.feature_center
        return F.normalize(self.project(features), dim=-1)

    def prototypes(self, support):
        return torch.stack([
            F.normalize(self.class_features(self.input_features(features)).mean(0), dim=-1)
            for features in support
        ])

    def forward_frame(self, features, boxes, prototypes, memory=None, objectness=None, memory_quality=None):
        current = self.input_features(features)
        gate = current.new_zeros(current.shape[0])
        if self.config.memory_weighting == "objectness":
            if objectness is None or objectness.shape != (features.shape[0],):
                raise ValueError("Objectness weighting requires one encoder objectness score per candidate")
            if not torch.isfinite(objectness).all() or (objectness < 0).any() or (objectness > 1).any():
                raise ValueError("Objectness scores must be finite probabilities")
        if memory is not None and memory.shape[0] and self.config.memory_mode != "none":
            attention_bias = None
            if self.config.memory_weighting == "objectness":
                if memory_quality is None or memory_quality.shape != (memory.shape[0],):
                    raise ValueError("Objectness weighting requires quality scores for stored memory")
                if not torch.isfinite(memory_quality).all() or (memory_quality < 0).any() or (memory_quality > 1).any():
                    raise ValueError("Stored memory quality must be finite probabilities")
                # Adding log(q_k) makes attention proportional to q_k*exp(a_jk).
                attention_bias = memory_quality.clamp(1e-6, 1).log()[None].expand(current.shape[0], -1)
            context, _ = self.attention(current[None], memory[None], memory[None],
                                         attn_mask=attention_bias, need_weights=False)
            updated = current + self.memory_scale * context[0]
            updated = updated + self.ffn(updated)
            if self.config.fusion_gate == "confidence":
                # Gate the full residual, including the FFN. Gating attention
                # alone would still let an unreliable temporal FFN change x.
                with torch.no_grad():
                    base_features = self.class_features(current)
                    base_foreground = self.logit_scale.exp().clamp(1, 100) * (base_features @ prototypes.T)
                    base_logits = torch.cat((base_foreground, self.background_head(current)), dim=-1)
                    gate = confidence_gate(base_logits.softmax(-1), self.config.gate_threshold)
                current = current + gate[:, None] * (updated - current)
            elif self.config.memory_weighting == "objectness":
                # Normalized attention alone is invariant to a common reduction
                # of every q. Also limit the full residual when all stored
                # candidates have weak object evidence. max avoids counting
                # duplicate candidates as independent evidence.
                gate = memory_quality.max().expand(current.shape[0])
                current = current + gate[:, None] * (updated - current)
            else:
                current = updated
                gate = current.new_ones(current.shape[0])
        classified = self.class_features(current)
        temperature = self.logit_scale.exp().clamp(1, 100)
        foreground_logits = temperature * (classified @ prototypes.T)
        logits = torch.cat((foreground_logits, self.background_head(current)), dim=-1)
        # Public OWLv2 boxes include the patch grid prior. Refine them in logit space.
        predicted_boxes = torch.sigmoid(torch.logit(boxes.clamp(1e-5, 1 - 1e-5)) + self.box_head(current))
        selected, thresholds = select_memory(logits.detach().softmax(-1), self.config)
        quality = objectness.detach()[selected] if objectness is not None else current.new_ones(len(selected))
        return {"logits": logits, "boxes": predicted_boxes, "memory": classified[selected],
                "memory_indices": selected, "thresholds": thresholds, "fusion_gate": gate,
                "memory_quality": quality}

    def forward(self, frames, support):
        if not frames or not support:
            raise ValueError("An episode needs frames and support classes")
        prototypes = self.prototypes(support)
        memory = None
        memory_quality = None
        outputs = []
        for frame in frames:
            output = self.forward_frame(frame["features"], frame["boxes"], prototypes, memory,
                                        frame.get("objectness"), memory_quality)
            outputs.append(output)
            memory = output["memory"]
            memory_quality = output["memory_quality"]
        return outputs

    def configuration(self):
        return asdict(self.config)
