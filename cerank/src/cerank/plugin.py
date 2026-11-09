"""Active CERank plugin without historical experimental branches."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CERankAdapter(nn.Module):
    """Bounded candidate representation adapter.

    This module returns adapted candidate representations. Scoring remains the
    responsibility of the native backbone bridge so that official biases and
    any native scorer semantics cannot be silently dropped.
    """

    def __init__(
        self,
        embedding_dim: int,
        evidence_channels: int,
        hidden_size: int = 32,
        dropout: float = 0.1,
        residual_cap: float = 0.2,
        evidence_mean: Sequence[float] = (),
        evidence_std: Sequence[float] = (),
        evidence_positive_mean: Sequence[float] = (),
        evidence_clip: float = 5.0,
        trust_temperature: float = 1.0,
        trust_floor: float = 0.0,
        zero_init_delta: bool = True,
        use_neural_features: bool = True,
        use_structural_features: bool = True,
        use_need_gate: bool = True,
        use_history_features: bool = True,
        use_trust_gate: bool = True,
        adaptation_space: str = "candidate",
    ) -> None:
        super().__init__()
        channels = int(evidence_channels)
        self.residual_cap = float(residual_cap)
        self.evidence_clip = float(evidence_clip)
        self.trust_temperature = float(trust_temperature)
        self.trust_floor = float(trust_floor)
        self.use_neural_features = bool(use_neural_features)
        self.use_structural_features = bool(use_structural_features)
        self.use_need_gate = bool(use_need_gate)
        self.use_history_features = bool(use_history_features)
        self.use_trust_gate = bool(use_trust_gate)
        self.adaptation_space = str(adaptation_space)
        if not 0.0 <= self.trust_floor <= 1.0:
            raise ValueError("trust_floor must lie in [0, 1]")
        if self.adaptation_space not in {"candidate", "score"}:
            raise ValueError("adaptation_space must be candidate or score")

        def prepared(values: Sequence[float], default: float) -> np.ndarray:
            if len(values) == 0:
                return np.full(channels, default, dtype=np.float32)
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (channels,):
                raise ValueError(
                    "Evidence statistic shape %s != (%d,)" % (array.shape, channels)
                )
            return array

        self.register_buffer(
            "evidence_mean", torch.from_numpy(prepared(evidence_mean, 0.0))
        )
        self.register_buffer(
            "evidence_std",
            torch.from_numpy(np.maximum(prepared(evidence_std, 1.0), 1e-6)),
        )
        self.register_buffer(
            "evidence_positive_mean",
            torch.from_numpy(
                np.maximum(prepared(evidence_positive_mean, 1.0), 1e-6)
            ),
        )
        self.hop_logits = nn.Parameter(torch.zeros(channels))
        feature_dim = int(embedding_dim) * 4 + 8
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.GELU(),
        )
        self.gate = nn.Linear(int(hidden_size), 1)
        delta_size = int(embedding_dim) if self.adaptation_space == "candidate" else 1
        self.delta = nn.Linear(int(hidden_size), delta_size)
        if zero_init_delta:
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        user_state: torch.Tensor,
        candidate_representations: torch.Tensor,
        base_scores: torch.Tensor,
        raw_evidence: torch.Tensor,
        target_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del base_scores  # Kept in the interface for future scorer-aware bridges.
        user_norm = F.normalize(user_state, dim=-1)
        item_norm = F.normalize(candidate_representations, dim=-1)
        expanded_user = user_norm.unsqueeze(1).expand_as(item_norm)
        neural_compatibility = (expanded_user * item_norm).sum(dim=-1)

        if not self.use_neural_features:
            expanded_user = torch.zeros_like(expanded_user)
            item_norm = torch.zeros_like(item_norm)
            neural_compatibility = torch.zeros_like(neural_compatibility)

        evidence_log = torch.log1p(raw_evidence.float().clamp_min(0.0))
        normalized = (evidence_log - self.evidence_mean) / self.evidence_std
        normalized = normalized.clamp(-self.evidence_clip, self.evidence_clip)
        hop_weights = torch.softmax(self.hop_logits, dim=0)
        structural = torch.einsum("bmc,c->bm", normalized, hop_weights)
        hop_trust = 1.0 - torch.exp(
            -evidence_log
            / (
                self.evidence_positive_mean
                * max(self.trust_temperature, 1e-6)
            )
        )
        trust = torch.einsum("bmc,c->bm", hop_trust.clamp(0.0, 1.0), hop_weights)
        coverage = raw_evidence.gt(0).float().mean(dim=-1)
        if not self.use_structural_features:
            structural = torch.zeros_like(structural)
            coverage = torch.zeros_like(coverage)
        if not self.use_trust_gate:
            trust = torch.ones_like(trust)
        length = target_sequence.ne(0).float().sum(dim=1) / float(
            max(target_sequence.size(1), 1)
        )
        need = 1.0 - length.clamp(0.0, 1.0)
        if not self.use_need_gate:
            need = torch.ones_like(need)
        if self.use_history_features:
            length_feature = length
            need_feature = need
        else:
            # Preserve the scalar dimensions and parameter count while
            # removing all record-specific target-history-length information.
            length_feature = torch.zeros_like(length)
            need_feature = torch.ones_like(need)
        scalars = torch.stack(
            [
                neural_compatibility,
                structural,
                neural_compatibility * structural,
                (neural_compatibility - structural).abs(),
                trust,
                length_feature[:, None].expand_as(structural),
                need_feature[:, None].expand_as(structural),
                coverage,
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                expanded_user,
                item_norm,
                expanded_user * item_norm,
                (expanded_user - item_norm).abs(),
                scalars,
            ],
            dim=-1,
        )
        hidden = self.trunk(features)
        learned_gate = torch.sigmoid(self.gate(hidden)).squeeze(-1)
        bounded_trust = self.trust_floor + (1.0 - self.trust_floor) * trust
        gate = learned_gate * need[:, None] * bounded_trust
        raw_shift = (
            self.residual_cap
            * gate.unsqueeze(-1)
            * torch.tanh(self.delta(hidden))
        )
        if self.adaptation_space == "candidate":
            shift = raw_shift
            adapted = candidate_representations + shift
            extra = {}
        else:
            shift = raw_shift
            adapted = candidate_representations
            extra = {"score_shift": raw_shift.squeeze(-1)}
        details = {
            "gate": gate,
            "shift": shift,
            "need": need,
            "history_length_feature": length_feature,
            "trust": trust,
            "hop_weights": hop_weights,
            "structural": structural,
        }
        details.update(extra)
        return adapted, details
