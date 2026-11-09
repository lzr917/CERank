"""Train-only target structural graph and raw multi-hop candidate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .protocol import is_domain


def build_cosine_graph(
    train_sessions: Sequence[Sequence[int]],
    item_count: int,
    anum: int,
    target_domain: str,
    topk: int = 50,
) -> sp.csr_matrix:
    """Build the active CERank target item graph.

    The final item of each training session is excluded so that a session-level
    label is never inserted into the structural memory for that session.
    """
    rows, cols = [], []
    for row_id, session in enumerate(train_sessions):
        items = sorted(
            {
                int(item)
                for item in session[:-1]
                if is_domain(int(item), target_domain, anum)
            }
        )
        rows.extend([row_id] * len(items))
        cols.extend(items)
    matrix = sp.coo_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(len(train_sessions), int(item_count)),
        dtype=np.float32,
    ).tocsr()
    user_degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    user_scale = np.zeros_like(user_degree, dtype=np.float32)
    user_scale[user_degree > 0] = 1.0 / np.sqrt(user_degree[user_degree > 0])
    matrix = sp.diags(user_scale).dot(matrix)
    item_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
    item_scale = np.zeros_like(item_degree, dtype=np.float32)
    item_scale[item_degree > 0] = 1.0 / np.sqrt(item_degree[item_degree > 0])
    matrix = matrix.dot(sp.diags(item_scale))
    graph = matrix.T.dot(matrix).tocsr().astype(np.float32)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    if topk > 0:
        graph = topk_rows(graph, int(topk))
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def topk_rows(matrix: sp.csr_matrix, topk: int) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    indptr = [0]
    indices, values = [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_values = matrix.data[start:end]
        if row_values.size > topk:
            keep = np.argpartition(row_values, -topk)[-topk:]
            keep = keep[np.argsort(-row_values[keep])]
            row_indices = row_indices[keep]
            row_values = row_values[keep]
        indices.extend(row_indices.tolist())
        values.extend(row_values.astype(np.float32).tolist())
        indptr.append(len(indices))
    return sp.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    )


def raw_multihop_evidence(
    records: Sequence[dict],
    graph: sp.csr_matrix,
    hops: Sequence[int] = (1, 2, 3),
    chunk_size: int = 1024,
) -> np.ndarray:
    if not records:
        raise ValueError("Cannot compute structural evidence for zero records")
    hops = sorted({int(hop) for hop in hops})
    if not hops or hops[0] < 1:
        raise ValueError("hops must contain positive integers")
    candidate_count = len(records[0]["candidates"])
    result = np.zeros(
        (len(records), candidate_count, len(hops)), dtype=np.float32
    )
    hop_to_channel = {hop: index for index, hop in enumerate(hops)}
    for start in range(0, len(records), max(int(chunk_size), 1)):
        end = min(start + int(chunk_size), len(records))
        rows, cols, values = [], [], []
        for row_id, record in enumerate(records[start:end]):
            counts = {}
            for item in record["target_seq"]:
                counts[int(item)] = counts.get(int(item), 0.0) + 1.0
            for item, value in counts.items():
                rows.append(row_id)
                cols.append(item)
                values.append(value)
        query = sp.coo_matrix(
            (
                np.asarray(values, dtype=np.float32),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=(end - start, graph.shape[0]),
            dtype=np.float32,
        ).tocsr()
        candidates = np.stack(
            [record["candidates"] for record in records[start:end]]
        )
        row_ids = np.repeat(np.arange(end - start), candidates.shape[1])
        candidate_ids = candidates.reshape(-1)
        state = query
        for hop in range(1, max(hops) + 1):
            state = state.dot(graph).tocsr()
            if hop in hop_to_channel:
                selected = np.asarray(state[row_ids, candidate_ids]).reshape(
                    end - start, candidates.shape[1]
                )
                result[start:end, :, hop_to_channel[hop]] = selected.astype(
                    np.float32
                )
    if not np.isfinite(result).all() or np.any(result < -1e-7):
        raise RuntimeError("Raw structural evidence must be finite and non-negative")
    return result


def evidence_statistics(evidence: np.ndarray) -> Dict[str, np.ndarray]:
    transformed = np.log1p(np.maximum(evidence, 0.0)).astype(np.float64)
    axes = tuple(range(transformed.ndim - 1))
    mean = transformed.mean(axis=axes)
    std = np.maximum(transformed.std(axis=axes), 1e-6)
    positive = transformed > 0
    positive_count = positive.sum(axis=axes)
    positive_mean = (
        np.where(positive, transformed, 0.0).sum(axis=axes)
        / np.maximum(positive_count, 1)
    )
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "positive_mean": np.maximum(positive_mean, 1e-6).astype(np.float32),
        "positive_count": positive_count.astype(np.int64),
    }


def evidence_statistics_streaming(
    evidence: np.ndarray, row_chunk: int = 4096
) -> Dict[str, np.ndarray]:
    """Population statistics without materializing a float64 evidence copy."""
    if evidence.ndim < 2 or evidence.shape[-1] <= 0:
        raise ValueError("evidence must have a nonempty channel dimension")
    channels = int(evidence.shape[-1])
    total = 0
    value_sum = np.zeros(channels, dtype=np.float64)
    square_sum = np.zeros(channels, dtype=np.float64)
    positive_sum = np.zeros(channels, dtype=np.float64)
    positive_count = np.zeros(channels, dtype=np.int64)
    for start in range(0, int(evidence.shape[0]), max(int(row_chunk), 1)):
        values = np.log1p(
            np.maximum(evidence[start : start + int(row_chunk)], 0.0)
        ).astype(np.float64)
        axes = tuple(range(values.ndim - 1))
        count = int(np.prod(values.shape[:-1]))
        total += count
        value_sum += values.sum(axis=axes)
        square_sum += np.square(values).sum(axis=axes)
        positive = values > 0
        positive_count += positive.sum(axis=axes).astype(np.int64)
        positive_sum += np.where(positive, values, 0.0).sum(axis=axes)
    if total <= 0:
        raise ValueError("Cannot compute statistics for empty evidence")
    mean = value_sum / float(total)
    variance = np.maximum(square_sum / float(total) - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    positive_mean = positive_sum / np.maximum(positive_count, 1)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "positive_mean": np.maximum(positive_mean, 1e-6).astype(np.float32),
        "positive_count": positive_count,
    }


def build_session_fold_manifest(
    split_sizes: Mapping[str, int],
    num_folds: int = 5,
    seed: int = 20260805,
) -> Tuple[List[str], np.ndarray, dict]:
    """Deterministic split-stratified session folds shared by all backbones."""
    num_folds = int(num_folds)
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    rng = np.random.RandomState(int(seed))
    session_uids: List[str] = []
    assignments: List[int] = []
    split_counts = {}
    for split_name in ("train_A", "train_B", "train_overlap"):
        size = int(split_sizes.get(split_name, 0))
        local = np.empty(size, dtype=np.int16)
        if size:
            permutation = rng.permutation(size)
            local[permutation] = np.arange(size, dtype=np.int64) % num_folds
        counts = [int(np.sum(local == fold)) for fold in range(num_folds)]
        split_counts[split_name] = counts
        session_uids.extend(
            "%s:%d" % (split_name, index) for index in range(size)
        )
        assignments.extend(int(value) for value in local.tolist())
    fold_array = np.asarray(assignments, dtype=np.int16)
    manifest_payload = {
        "schema": 1,
        "protocol": "session_crossfit_k%d_v1" % num_folds,
        "seed": int(seed),
        "num_folds": num_folds,
        "split_sizes": {
            name: int(split_sizes.get(name, 0))
            for name in ("train_A", "train_B", "train_overlap")
        },
        "split_fold_counts": split_counts,
    }
    digest_payload = dict(
        manifest_payload, assignments=fold_array.astype(int).tolist()
    )
    serialized = json.dumps(
        digest_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_payload["sha256"] = hashlib.sha256(serialized).hexdigest()
    return session_uids, fold_array, manifest_payload


def cache_key(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()[:12]


def save_evidence_cache(path: Path, evidence: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        evidence=evidence.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def load_evidence_cache(
    path: Path, expected_shape: Sequence[int], metadata: dict
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        cached_metadata = json.loads(str(payload["metadata"].item()))
        if cached_metadata != metadata:
            raise ValueError("Structural cache metadata mismatch at %s" % path)
        evidence = np.asarray(payload["evidence"], dtype=np.float32)
    if tuple(evidence.shape) != tuple(expected_shape):
        raise ValueError(
            "Structural cache shape %s != %s" % (evidence.shape, expected_shape)
        )
    return evidence


def legacy_v3_cache_key(direction: str, target_domain: str) -> str:
    """Reproduce the audited cache key used by the current v3 implementation."""
    payload = {
        "direction": direction,
        "target_domain": target_domain,
        "hops": [1, 2, 3],
        "train_neg": 100,
        "structural_hard_neg": 0,
        "eval_neg": 999,
        "evidence_sources": "graph",
        "transition_recency_decay": 0.8,
        "semantic_evidence": "none",
        "semantic_score_mode": "mean_topm",
        "semantic_top_m": 5,
        "structural_representation": "raw_graph_evidence",
        "query": None,
        "graph_mode": "cosine",
        # This value was part of the old key/file name but was not applied by
        # the cosine branch. The clean graph_topk=0 reproduces its computation.
        "graph_topk": 50,
        "pmi_eps": 1e-8,
        "check_only": False,
        "v": 6,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def legacy_graph_path(
    legacy_root: Path,
    dataset: str,
    direction: str,
    target_domain: str,
) -> Path:
    return (
        Path(legacy_root)
        / dataset
        / "graphs"
        / (
            "%s_%s_target_history_full_cosine_topk50_eps1e-08.npz"
            % (direction, target_domain)
        )
    )


def legacy_evidence_path(
    legacy_root: Path,
    dataset: str,
    seed: int,
    direction: str,
    split: str,
    target_domain: str,
) -> Path:
    key = legacy_v3_cache_key(direction, target_domain)
    return (
        Path(legacy_root)
        / dataset
        / ("seed%d" % int(seed))
        / direction
        / "k100"
        / ("%s_%s.pt" % (split, key))
    )


def load_legacy_v3_evidence(
    path: Path,
    expected_shape: Sequence[int],
) -> np.ndarray:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "structural" not in payload:
        raise ValueError("Not a legacy structural evidence cache: %s" % path)
    metadata = payload.get("metadata")
    expected_metadata = {
        "cache_schema_version": 3,
        "representation": "raw_graph_evidence",
        "storage_dtype": "float32",
        "hops": [1, 2, 3],
        "query_weight_mode": "uniform",
        "query_recent_k": 3,
        "query_recency_gamma": 0.7,
        "query_long_gamma": 0.98,
        "query_short_gamma": 0.7,
    }
    if metadata != expected_metadata:
        raise ValueError("Legacy raw-v3 metadata mismatch at %s" % path)
    evidence = np.asarray(payload["structural"], dtype=np.float32)
    if tuple(evidence.shape) != tuple(expected_shape):
        raise ValueError(
            "Legacy evidence shape %s != %s" % (evidence.shape, expected_shape)
        )
    if not np.isfinite(evidence).all() or np.any(evidence < -1e-7):
        raise ValueError("Legacy evidence is non-finite or negative at %s" % path)
    return evidence


def validate_evidence_alignment(
    records: Sequence[dict],
    graph: sp.csr_matrix,
    evidence: np.ndarray,
    sample_count: int = 16,
    tolerance: float = 1e-6,
) -> float:
    """Spot-check that cached rows use the current records/candidate ordering."""
    count = min(max(int(sample_count), 1), len(records))
    indices = np.linspace(0, len(records) - 1, num=count, dtype=np.int64)
    sampled_records = [records[int(index)] for index in indices]
    recomputed = raw_multihop_evidence(
        sampled_records, graph, hops=(1, 2, 3), chunk_size=count
    )
    selected = evidence[indices]
    error = float(np.max(np.abs(recomputed - selected)))
    if error > tolerance:
        raise ValueError(
            "Legacy evidence/candidate alignment error %g > %g"
            % (error, tolerance)
        )
    return error
