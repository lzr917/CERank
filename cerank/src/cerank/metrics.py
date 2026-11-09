"""Small ranking-metric implementation shared by all native bridges."""

from __future__ import annotations

from typing import Dict, Iterable

import torch


def ranking_metrics(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    cutoffs: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError(
            "scores and positive_mask must have the same [batch, candidates] shape"
        )
    positive_score = (scores * positive_mask).sum(dim=1, keepdim=True)
    # Candidate zero is positive in the protocol, but using the mask keeps this
    # evaluator reusable and gives deterministic tie handling.
    rank = 1 + ((scores > positive_score) & positive_mask.eq(0)).sum(dim=1)
    result = {}
    rank_float = rank.float()
    for cutoff in cutoffs:
        hit = rank.le(int(cutoff)).float()
        result["hr@%d" % cutoff] = float(hit.mean().cpu())
        result["recall@%d" % cutoff] = float(hit.mean().cpu())
        result["ndcg@%d" % cutoff] = float(
            (hit / torch.log2(rank_float + 1.0)).mean().cpu()
        )
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.total = {}
        self.count = 0

    def update(self, metrics: Dict[str, float], batch_size: int) -> None:
        self.count += int(batch_size)
        for key, value in metrics.items():
            self.total[key] = self.total.get(key, 0.0) + float(value) * batch_size

    def compute(self) -> Dict[str, float]:
        result = {
            key: value / max(self.count, 1) for key, value in self.total.items()
        }
        result["count"] = self.count
        return result

