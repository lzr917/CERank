"""Shared training/evaluation utilities for native backbone integrations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import MetricAccumulator, ranking_metrics
from .protocol import DirectionalDataset


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    non_blocking = str(device).startswith("cuda")
    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def data_loader_runtime_kwargs(config: dict, device: str) -> dict:
    """Runtime-only loader acceleration that preserves samples and ordering."""
    num_workers = int(config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    result = {
        "num_workers": num_workers,
        "pin_memory": bool(
            config.get("pin_memory", True)
            and str(device).startswith("cuda")
        ),
        # Workers must be recreated after train_set.set_epoch(epoch), otherwise
        # their dataset copies would retain stale epoch-resampling state.
        "persistent_workers": False,
    }
    if num_workers > 0:
        prefetch_factor = int(config.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        result["prefetch_factor"] = prefetch_factor
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class EvidenceDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict],
        maxlen: int,
        evidence: np.ndarray,
        control_seed: Optional[int] = None,
        evidence_control: str = "aligned",
        corruption_rate: float = 0.0,
    ) -> None:
        self.base = DirectionalDataset(records, maxlen)
        self.evidence = np.asarray(evidence, dtype=np.float32)
        if self.evidence.shape[0] != len(self.base):
            raise ValueError(
                "Evidence rows %d != records %d"
                % (self.evidence.shape[0], len(self.base))
            )
        self.evidence_control = str(evidence_control)
        if self.evidence_control not in {
            "aligned",
            "shuffled",
            "partial_shuffle",
            "dropout",
        }:
            raise ValueError(
                "evidence_control must be aligned, shuffled, "
                "partial_shuffle, or dropout"
            )
        self.corruption_rate = float(corruption_rate)
        if not 0.0 <= self.corruption_rate <= 1.0:
            raise ValueError("corruption_rate must lie in [0, 1]")
        if self.evidence_control == "aligned":
            self.corruption_rate = 0.0
        elif self.evidence_control == "shuffled":
            # Preserve the historical shuffled_structure control exactly.
            self.corruption_rate = 1.0
        if self.evidence_control != "aligned" and control_seed is None:
            raise ValueError("corrupted evidence requires control_seed")

        self.control_seed = None if control_seed is None else int(control_seed)
        self.evidence_rows = None
        self.evidence_candidates = None
        self.corrupt_rows = None
        if self.evidence_control in {"shuffled", "partial_shuffle"}:
            rng = np.random.RandomState(self.control_seed)
            self.evidence_rows = rng.permutation(len(self.base))
            self.evidence_candidates = rng.permutation(self.evidence.shape[1])
            if self.evidence_control == "partial_shuffle":
                self.corrupt_rows = rng.rand(len(self.base)) < self.corruption_rate

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.base[index]
        evidence_index = (
            index if self.evidence_rows is None else self.evidence_rows[index]
        )
        if self.corrupt_rows is not None and not self.corrupt_rows[index]:
            evidence_index = index
        evidence = self.evidence[evidence_index]
        should_shuffle_candidates = (
            self.evidence_candidates is not None
            and (self.corrupt_rows is None or self.corrupt_rows[index])
        )
        if should_shuffle_candidates:
            evidence = evidence[self.evidence_candidates]
        if self.evidence_control == "dropout" and self.corruption_rate > 0.0:
            # Generate the mask per record so Video does not require a second
            # dataset-sized boolean tensor. The mapping is deterministic across
            # workers, epochs, and reruns.
            row_seed = (self.control_seed + 104729 * index) % (2**32)
            rng = np.random.RandomState(row_seed)
            keep = rng.rand(*evidence.shape) >= self.corruption_rate
            evidence = evidence * keep
        # A cross-fit cache may be a read-only memory map. Copy only the
        # selected record (not the dataset-sized array) before wrapping it.
        item["structural"] = torch.from_numpy(
            np.array(evidence, dtype=np.float32, copy=True)
        )
        return item


def evaluate_backbone(bridge, loader, device: str) -> dict:
    bridge.model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            scores = bridge.score_candidates(batch, batch["candidates"])
            metrics = ranking_metrics(scores, batch["positive_mask"])
            accumulator.update(metrics, scores.size(0))
    return accumulator.compute()


def evaluate_adapter(bridge, adapter, loader, device: str):
    bridge.model.eval()
    adapter.eval()
    accumulator = MetricAccumulator()
    auxiliary = {"gate": 0.0, "shift_l2": 0.0, "need": 0.0, "trust": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            user = bridge.encode_user(batch)
            candidates = bridge.candidate_representations(batch["candidates"])
            base_scores = bridge.direct_scores_from_user(
                user, batch["candidates"]
            )
            adapted, details = adapter(
                user,
                candidates,
                base_scores,
                batch["structural"],
                batch["target_seq"],
            )
            if "score_shift" in details:
                scores = base_scores + details["score_shift"]
            else:
                scores = bridge.score_candidate_delta(
                    user, base_scores, details["shift"]
                )
            metrics = ranking_metrics(scores, batch["positive_mask"])
            size = scores.size(0)
            accumulator.update(metrics, size)
            count += size
            auxiliary["gate"] += float(details["gate"].mean().cpu()) * size
            auxiliary["shift_l2"] += (
                float(details["shift"].norm(dim=-1).mean().cpu()) * size
            )
            auxiliary["need"] += float(details["need"].mean().cpu()) * size
            auxiliary["trust"] += float(details["trust"].mean().cpu()) * size
    return accumulator.compute(), {
        key: value / max(count, 1) for key, value in auxiliary.items()
    }


def evaluate_by_target_length(bridge, adapter, loader, device: str) -> dict:
    """Evaluate fixed length bins and return paired per-record ranks."""
    bins = (
        ("1-2", 1, 2),
        ("3-5", 3, 5),
        ("6-10", 6, 10),
        ("11+", 11, None),
    )
    accumulators = {
        name: {"base": MetricAccumulator(), "adapted": MetricAccumulator()}
        for name, _, _ in bins
    }
    paired = {
        "base_rank": [],
        "adapted_rank": [],
        "target_length": [],
        "positive_trust": [],
        "positive_coverage": [],
        "positive_gate": [],
        "positive_shift_l2": [],
        "mean_trust": [],
        "mean_coverage": [],
    }
    bridge.model.eval()
    adapter.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            user = bridge.encode_user(batch)
            candidates = bridge.candidate_representations(batch["candidates"])
            base_scores = bridge.direct_scores_from_user(
                user, batch["candidates"]
            )
            _, details = adapter(
                user,
                candidates,
                base_scores,
                batch["structural"],
                batch["target_seq"],
            )
            if "score_shift" in details:
                adapted_scores = base_scores + details["score_shift"]
            else:
                adapted_scores = bridge.score_candidate_delta(
                    user, base_scores, details["shift"]
                )
            lengths = batch["target_seq"].ne(0).sum(dim=1)
            positive_scores = (
                base_scores * batch["positive_mask"]
            ).sum(dim=1, keepdim=True)
            base_rank = 1 + (
                (base_scores > positive_scores)
                & batch["positive_mask"].eq(0)
            ).sum(dim=1)
            adapted_positive_scores = (
                adapted_scores * batch["positive_mask"]
            ).sum(dim=1, keepdim=True)
            adapted_rank = 1 + (
                (adapted_scores > adapted_positive_scores)
                & batch["positive_mask"].eq(0)
            ).sum(dim=1)
            paired["base_rank"].append(base_rank.cpu().numpy())
            paired["adapted_rank"].append(adapted_rank.cpu().numpy())
            paired["target_length"].append(lengths.cpu().numpy())
            # Preserve complete sampled-candidate audit tensors, but avoid
            # materializing multi-gigabyte all-catalog score matrices during
            # Run19 full-ranking evaluation.
            if batch["candidates"].shape[1] <= 1000:
                paired.setdefault("candidate_ids", []).append(
                    batch["candidates"].cpu().numpy()
                )
                paired.setdefault("positive_ids", []).append(
                    (batch["candidates"] * batch["positive_mask"].long())
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                )
                paired.setdefault("base_scores", []).append(
                    base_scores.float().cpu().numpy()
                )
                paired.setdefault("adapted_scores", []).append(
                    adapted_scores.float().cpu().numpy()
                )
            positive_mask = batch["positive_mask"].float()
            coverage = batch["structural"].gt(0).float().mean(dim=-1)
            shift_l2 = details["shift"].norm(dim=-1)
            for name, values in (
                ("positive_trust", details["trust"]),
                ("positive_coverage", coverage),
                ("positive_gate", details["gate"]),
                ("positive_shift_l2", shift_l2),
            ):
                paired[name].append(
                    (values * positive_mask).sum(dim=1).cpu().numpy()
                )
            paired["mean_trust"].append(
                details["trust"].mean(dim=1).cpu().numpy()
            )
            paired["mean_coverage"].append(
                coverage.mean(dim=1).cpu().numpy()
            )
            for name, lower, upper in bins:
                mask = lengths.ge(lower)
                if upper is not None:
                    mask = mask & lengths.le(upper)
                size = int(mask.sum().item())
                if size == 0:
                    continue
                positives = batch["positive_mask"][mask]
                for method, scores in (
                    ("base", base_scores),
                    ("adapted", adapted_scores),
                ):
                    metrics = ranking_metrics(scores[mask], positives)
                    accumulators[name][method].update(metrics, size)
    breakdown = {
        name: {
            method: accumulator.compute()
            for method, accumulator in methods.items()
        }
        for name, methods in accumulators.items()
        if methods["base"].count > 0
    }
    integer_fields = {
        "base_rank", "adapted_rank", "target_length", "candidate_ids",
        "positive_ids",
    }
    paired_arrays = {
        name: np.concatenate(values).astype(
            np.int64 if name in integer_fields else np.float32
        )
        for name, values in paired.items()
    }
    paired_arrays["record_id"] = np.arange(
        len(paired_arrays["base_rank"]), dtype=np.int64
    )
    return breakdown, paired_arrays
