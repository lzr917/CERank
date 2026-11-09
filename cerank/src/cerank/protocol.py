"""Fixed-direction, target-visible CDR protocol.

Local item IDs are one-based:

- A items: 1 .. anum
- B items: anum + 1 .. anum + bnum
- padding: 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DirectionSpec:
    direction: str
    source_domain: str
    target_domain: str


def parse_direction(direction: str) -> DirectionSpec:
    value = direction.upper().replace("->", "2")
    if value == "A2B":
        return DirectionSpec("A2B", "A", "B")
    if value == "B2A":
        return DirectionSpec("B2A", "B", "A")
    raise ValueError("direction must be A2B or B2A, got %r" % direction)


def read_num_items(dataset_dir: Path) -> Tuple[int, int]:
    lines = (dataset_dir / "num.txt").read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("Expected two lines in %s" % (dataset_dir / "num.txt"))
    return int(lines[0]), int(lines[1])


def read_sequences(path: Path) -> List[List[int]]:
    sequences = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip().split("\t", 1)
            if len(fields) != 2:
                continue
            # Released data are zero-based; the clean runtime uses padding=0.
            sequences.append(
                [int(value) + 1 for value in fields[1].split(",") if value != ""]
            )
    return sequences


def load_splits(
    data_root: Path,
    dataset: str,
    session_limit: Optional[int] = None,
) -> Tuple[Path, int, int, Dict[str, List[List[int]]]]:
    dataset_dir = data_root / dataset
    anum, bnum = read_num_items(dataset_dir)
    names = ("train_A", "train_B", "train_overlap", "valid", "test")
    splits = {name: read_sequences(dataset_dir / (name + ".txt")) for name in names}
    if session_limit is not None:
        splits = {name: values[: int(session_limit)] for name, values in splits.items()}
    return dataset_dir, anum, bnum, splits


def is_domain(item: int, domain: str, anum: int) -> bool:
    return (int(item) <= anum) == (domain == "A")


def split_prefix(
    prefix: Sequence[int], spec: DirectionSpec, anum: int
) -> Tuple[List[int], List[int]]:
    source = [int(item) for item in prefix if is_domain(item, spec.source_domain, anum)]
    target = [int(item) for item in prefix if is_domain(item, spec.target_domain, anum)]
    return source, target


def pad_left(values: Sequence[int], length: int) -> np.ndarray:
    array = np.zeros(int(length), dtype=np.int64)
    visible = np.asarray(list(values)[-int(length) :], dtype=np.int64)
    if visible.size:
        array[-visible.size :] = visible
    return array


def pad_mixed_target_visible(
    mixed_sequence: Sequence[int],
    target_sequence: Sequence[int],
    length: int,
) -> np.ndarray:
    """Chronological truncation that preserves at least one visible target.

    The fixed protocol requires a nonempty target prefix. A long suffix can
    nevertheless contain only source events, so blindly taking the last
    ``length`` mixed events would violate that invariant inside a native
    backbone. In that rare case the latest visible target precedes the source
    suffix; retain it followed by the most recent ``length - 1`` mixed events.
    """
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    mixed = [int(item) for item in mixed_sequence]
    target = [int(item) for item in target_sequence]
    visible = mixed[-length:]
    if target:
        target_items = set(target)
        if not any(item in target_items for item in visible):
            visible = [target[-1]] + mixed[-max(length - 1, 0) :]
    return pad_left(visible, length)


def sample_target_negatives(
    positive: int,
    anum: int,
    bnum: int,
    count: int,
    rng: np.random.RandomState,
    excluded: Optional[Set[int]] = None,
) -> np.ndarray:
    start, end = (1, anum + 1) if positive <= anum else (anum + 1, anum + bnum + 1)
    blocked = {int(positive)}
    if excluded:
        blocked.update(item for item in excluded if start <= item < end)
    available = end - start - len(blocked)
    if available < count:
        raise ValueError(
            "Not enough target negatives: available=%d requested=%d" % (available, count)
        )
    # Match the released/current protocol exactly when only the positive is
    # blocked. This consumes the same RandomState draws as the legacy cache
    # builder and is required for byte-compatible validation/test candidates.
    if len(blocked) == 1:
        sampled = rng.choice(end - start - 1, size=int(count), replace=False) + start
        sampled += sampled >= int(positive)
        return sampled.astype(np.int64)
    pool = np.asarray(
        [item for item in range(start, end) if item not in blocked], dtype=np.int64
    )
    return rng.choice(pool, size=int(count), replace=False).astype(np.int64)


def make_train_samples(
    sessions: Sequence[Sequence[int]],
    anum: int,
    direction: str,
    session_uids: Optional[Sequence[str]] = None,
    session_folds: Optional[Sequence[int]] = None,
    position_protocol: str = "expanded",
) -> Tuple[List[dict], dict]:
    spec = parse_direction(direction)
    if session_uids is not None and len(session_uids) != len(sessions):
        raise ValueError("session_uids must align with sessions")
    if session_folds is not None and len(session_folds) != len(sessions):
        raise ValueError("session_folds must align with sessions")
    if position_protocol not in {"expanded", "terminal"}:
        raise ValueError("position_protocol must be expanded or terminal")
    records = []
    stats = {
        "skipped_wrong_label": 0,
        "skipped_empty_target": 0,
        "source_nonempty": 0,
    }
    for session_index, session in enumerate(sessions):
        positions = (
            range(1, len(session))
            if position_protocol == "expanded"
            else ([len(session) - 1] if len(session) >= 2 else [])
        )
        for index in positions:
            prefix = [int(item) for item in session[:index]]
            label = int(session[index])
            if not is_domain(label, spec.target_domain, anum):
                stats["skipped_wrong_label"] += 1
                continue
            source, target = split_prefix(prefix, spec, anum)
            if not target:
                stats["skipped_empty_target"] += 1
                continue
            if source:
                stats["source_nonempty"] += 1
            record = {
                    "mixed_seq": prefix,
                    "source_seq": source,
                    "target_seq": target,
                    "label": label,
                    "candidates": np.asarray([label], dtype=np.int64),
                    "positive_mask": np.asarray([1.0], dtype=np.float32),
                    "origin_session_index": int(session_index),
                    "origin_position": int(index),
                }
            if session_uids is not None:
                record["origin_session_uid"] = str(session_uids[session_index])
            if session_folds is not None:
                record["origin_fold"] = int(session_folds[session_index])
            records.append(record)
    stats["samples"] = len(records)
    stats["source_nonempty_ratio"] = stats["source_nonempty"] / max(len(records), 1)
    stats["position_protocol"] = position_protocol
    return records, stats


def make_train_records(
    sessions: Sequence[Sequence[int]],
    anum: int,
    bnum: int,
    direction: str,
    num_negatives: int,
    seed: int,
    session_uids: Optional[Sequence[str]] = None,
    session_folds: Optional[Sequence[int]] = None,
    position_protocol: str = "expanded",
) -> Tuple[List[dict], dict]:
    samples, stats = make_train_samples(
        sessions,
        anum,
        direction,
        session_uids=session_uids,
        session_folds=session_folds,
        position_protocol=position_protocol,
    )
    records = []
    rng = np.random.RandomState(seed)
    for sample in samples:
        record = dict(sample)
        negatives = sample_target_negatives(
            int(record["label"]),
            anum,
            bnum,
            num_negatives,
            rng,
            excluded=set(record["target_seq"]),
        )
        candidates = np.asarray(
            [int(record["label"])] + negatives.tolist(), dtype=np.int64
        )
        positive_mask = np.zeros(candidates.size, dtype=np.float32)
        positive_mask[0] = 1.0
        record["candidates"] = candidates
        record["positive_mask"] = positive_mask
        records.append(record)
    return records, stats


def make_eval_records(
    sessions: Sequence[Sequence[int]],
    anum: int,
    bnum: int,
    direction: str,
    num_negatives: int,
    seed: int,
) -> Tuple[List[dict], dict]:
    spec = parse_direction(direction)
    records = []
    stats = {
        "skipped_wrong_label": 0,
        "skipped_empty_target": 0,
        "source_nonempty": 0,
    }
    rng = np.random.RandomState(seed)
    for session in sessions:
        if len(session) < 2:
            continue
        prefix = [int(item) for item in session[:-1]]
        label = int(session[-1])
        if not is_domain(label, spec.target_domain, anum):
            stats["skipped_wrong_label"] += 1
            continue
        source, target = split_prefix(prefix, spec, anum)
        if not target:
            stats["skipped_empty_target"] += 1
            continue
        if source:
            stats["source_nonempty"] += 1
        negatives = sample_target_negatives(label, anum, bnum, num_negatives, rng)
        candidates = np.asarray([label] + negatives.tolist(), dtype=np.int64)
        positive_mask = np.zeros(candidates.size, dtype=np.float32)
        positive_mask[0] = 1.0
        records.append(
            {
                "mixed_seq": prefix,
                "source_seq": source,
                "target_seq": target,
                "label": label,
                "candidates": candidates,
                "positive_mask": positive_mask,
            }
        )
    stats["samples"] = len(records)
    stats["source_nonempty_ratio"] = stats["source_nonempty"] / max(len(records), 1)
    return records, stats


def make_full_rank_eval_records(
    sessions: Sequence[Sequence[int]],
    anum: int,
    bnum: int,
    direction: str,
) -> Tuple[List[dict], dict]:
    """Build deterministic all-target-item evaluation records.

    This is the exact-catalog counterpart of :func:`make_eval_records`: the
    positive remains candidate zero and every other item in the target domain
    is a negative. Previously observed items are intentionally retained so the
    eligible set matches the existing sampled protocol, which does not filter
    target-prefix items from its negative pool.
    """
    spec = parse_direction(direction)
    start, end = (
        (1, anum + 1)
        if spec.target_domain == "A"
        else (anum + 1, anum + bnum + 1)
    )
    target_items = np.arange(start, end, dtype=np.int64)
    records = []
    stats = {
        "skipped_wrong_label": 0,
        "skipped_empty_target": 0,
        "source_nonempty": 0,
        "candidate_protocol": "all target items; observed items retained",
        "candidates_per_record": int(target_items.size),
    }
    for session in sessions:
        if len(session) < 2:
            continue
        prefix = [int(item) for item in session[:-1]]
        label = int(session[-1])
        if not is_domain(label, spec.target_domain, anum):
            stats["skipped_wrong_label"] += 1
            continue
        source, target = split_prefix(prefix, spec, anum)
        if not target:
            stats["skipped_empty_target"] += 1
            continue
        if source:
            stats["source_nonempty"] += 1
        negatives = target_items[target_items != label]
        candidates = np.concatenate(
            [np.asarray([label], dtype=np.int64), negatives]
        )
        positive_mask = np.zeros(candidates.size, dtype=np.float32)
        positive_mask[0] = 1.0
        records.append(
            {
                "mixed_seq": prefix,
                "source_seq": source,
                "target_seq": target,
                "label": label,
                "candidates": candidates,
                "positive_mask": positive_mask,
            }
        )
    stats["samples"] = len(records)
    stats["source_nonempty_ratio"] = stats["source_nonempty"] / max(
        len(records), 1
    )
    return records, stats


class DirectionalDataset(Dataset):
    def __init__(self, records: Sequence[dict], maxlen: int) -> None:
        self.records = list(records)
        self.maxlen = int(maxlen)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "mixed_seq": torch.from_numpy(
                pad_mixed_target_visible(
                    record["mixed_seq"], record["target_seq"], self.maxlen
                )
            ).long(),
            "source_seq": torch.from_numpy(
                pad_left(record["source_seq"], self.maxlen)
            ).long(),
            "target_seq": torch.from_numpy(
                pad_left(record["target_seq"], self.maxlen)
            ).long(),
            "label": torch.tensor(record["label"], dtype=torch.long),
            "candidates": torch.from_numpy(record["candidates"]).long(),
            "positive_mask": torch.from_numpy(record["positive_mask"]).float(),
        }


class BackboneTrainDataset(DirectionalDataset):
    """Epoch-resampled 1+K target candidates used by external backbones."""

    def __init__(
        self,
        records: Sequence[dict],
        maxlen: int,
        anum: int,
        bnum: int,
        num_negatives: int,
        seed: int,
    ) -> None:
        super().__init__(records, maxlen)
        self.anum = int(anum)
        self.bnum = int(bnum)
        self.num_negatives = int(num_negatives)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        label = int(self.records[index]["label"])
        sample_seed = (
            (self.seed * 1000003) + (self.epoch * 9176) + int(index)
        ) % (2**32 - 1)
        rng = np.random.RandomState(sample_seed)
        negatives = sample_target_negatives(
            label,
            self.anum,
            self.bnum,
            self.num_negatives,
            rng,
        )
        candidates = np.asarray([label] + negatives.tolist(), dtype=np.int64)
        positive_mask = np.zeros(candidates.size, dtype=np.float32)
        positive_mask[0] = 1.0
        item["candidates"] = torch.from_numpy(candidates).long()
        item["positive_mask"] = torch.from_numpy(positive_mask).float()
        return item
