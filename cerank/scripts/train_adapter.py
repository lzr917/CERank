#!/usr/bin/env python3
"""Train CERank on a frozen registered native-backbone checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from cerank.backbones.c2dsr import NativeC2DSRBridge  # noqa: E402
from cerank.backbones.recbole_sequential import (  # noqa: E402
    NativeGRU4RecBridge,
    NativeSASRecBridge,
)
from cerank.plugin import CERankAdapter  # noqa: E402
from cerank.metrics import ranking_metrics  # noqa: E402
from cerank.protocol import (  # noqa: E402
    load_splits,
    make_eval_records,
    make_full_rank_eval_records,
    make_train_records,
    parse_direction,
)
from cerank.structural import (  # noqa: E402
    build_cosine_graph,
    build_session_fold_manifest,
    cache_key,
    evidence_statistics,
    evidence_statistics_streaming,
    legacy_evidence_path,
    legacy_graph_path,
    load_evidence_cache,
    load_legacy_v3_evidence,
    raw_multihop_evidence,
    save_evidence_cache,
    validate_evidence_alignment,
)
from cerank.training import (  # noqa: E402
    EvidenceDataset,
    data_loader_runtime_kwargs,
    evaluate_adapter,
    evaluate_backbone,
    evaluate_by_target_length,
    read_json,
    resolve_device,
    set_seed,
    to_device,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--backbone",
        choices=(
            "c2dsr",
            "sasrec",
        ),
        default="c2dsr",
    )
    result.add_argument("--dataset", required=True)
    result.add_argument("--direction", choices=("A2B", "B2A"), required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--data_root", default="data")
    result.add_argument("--official_root", default=None)
    result.add_argument("--backbone_config", default=None)
    result.add_argument(
        "--adapter_config", default="cerank/configs/cerank.json"
    )
    result.add_argument("--backbone_root", default=None)
    result.add_argument("--output_root", default=None)
    result.add_argument(
        "--cache_root", default="cerank/results/cache/structural_v3"
    )
    result.add_argument(
        "--legacy_cache_root",
        default="paper_cseb_candidate/results/cache/shared_structural_v3",
    )
    result.add_argument("--disable_legacy_cache", action="store_true")
    result.add_argument(
        "--evidence_protocol",
        choices=("full_train_graph", "session_crossfit_k5"),
        default="full_train_graph",
        help="Training-evidence provenance; evaluation always uses full train graph",
    )
    result.add_argument("--crossfit_folds", type=int, default=5)
    result.add_argument("--crossfit_seed", type=int, default=20260805)
    result.add_argument(
        "--train_position_protocol",
        choices=("expanded", "terminal"),
        default="expanded",
    )
    result.add_argument(
        "--prepare_evidence_only",
        action="store_true",
        help="Build/audit structural caches and exit before loading a backbone",
    )
    result.add_argument("--check_only", action="store_true")
    result.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override config training epochs; recorded in run_config",
    )
    result.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override config early-stopping patience; recorded in run_config",
    )
    result.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config training batch size; recorded in run_config",
    )
    result.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Override config evaluation batch size; recorded in run_config",
    )
    result.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override deterministic DataLoader workers; recorded in run_config",
    )
    result.add_argument("--session_limit", type=int, default=None)
    result.add_argument(
        "--full_ranking",
        action="store_true",
        help=(
            "Evaluate/early-stop against every target-domain item; observed "
            "target-prefix items remain eligible, matching sampled evaluation"
        ),
    )
    result.add_argument(
        "--analysis_run_dir",
        default=None,
        help=(
            "Reuse best_adapter.pt and run_config.json from a completed run; "
            "skip training and emit repeatable scenario/efficiency diagnostics"
        ),
    )
    result.add_argument(
        "--analysis_output",
        default=None,
        help="JSON output path required with --analysis_run_dir",
    )
    result.add_argument(
        "--analysis_mode",
        choices=("all", "diagnostics", "benchmark", "direct_fusion"),
        default="all",
        help=(
            "all preserves the legacy combined report; diagnostics skips "
            "hardware timing; benchmark skips metric/subgroup diagnostics; "
            "direct_fusion tunes a linear structural-score weight on validation"
        ),
    )
    result.add_argument(
        "--direct_fusion_lambdas",
        default="0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100",
        help="Comma-separated nonnegative lambda grid for validation tuning",
    )
    result.add_argument("--benchmark_warmup_rounds", type=int, default=2)
    result.add_argument("--benchmark_repetitions", type=int, default=5)
    result.add_argument(
        "--benchmark_batch_limit",
        type=int,
        default=0,
        help="0 measures the complete test loader",
    )
    return result


def evidence_for_split(
    records,
    graph,
    hops,
    chunk_size,
    cache_path: Path,
    metadata: dict,
):
    shape = (len(records), len(records[0]["candidates"]), len(hops))
    if cache_path.exists():
        return load_evidence_cache(cache_path, shape, metadata)
    evidence = raw_multihop_evidence(records, graph, hops, chunk_size)
    save_evidence_cache(cache_path, evidence, metadata)
    return evidence


def _train_record_digest(records) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.get("origin_session_uid", "")).encode("utf-8"))
        digest.update(np.asarray([record["origin_position"], record["label"]], dtype=np.int64).tobytes())
        digest.update(np.asarray(record["candidates"], dtype=np.int64).tobytes())
    return digest.hexdigest()


def crossfit_train_evidence(
    records,
    train_sessions,
    session_folds,
    item_count,
    anum,
    target_domain,
    graph_topk,
    hops,
    chunk_size,
    cache_path: Path,
    request_metadata: dict,
):
    """Generate OOF evidence while never loading five full outputs at once."""
    shape = (len(records), len(records[0]["candidates"]), len(hops))
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_path.exists() and metadata_path.exists():
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored.get("request") != request_metadata:
            raise ValueError("Cross-fit cache metadata mismatch at %s" % cache_path)
        evidence = np.load(cache_path, mmap_mode="r")
        if tuple(evidence.shape) != shape:
            raise ValueError("Cross-fit cache shape %s != %s" % (evidence.shape, shape))
        return evidence, stored["audit"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    record_folds = np.asarray([record["origin_fold"] for record in records], dtype=np.int16)
    session_folds = np.asarray(session_folds, dtype=np.int16)
    provenance_mismatches = sum(
        int(record["origin_fold"])
        != int(session_folds[int(record["origin_session_index"])])
        for record in records
    )
    if provenance_mismatches:
        raise RuntimeError(
            "%d train rows disagree with their originating session fold"
            % provenance_mismatches
        )
    fold_audits = []
    written = np.zeros(len(records), dtype=bool)
    for fold in range(int(request_metadata["num_folds"])):
        fold_started = time.time()
        graph_session_indices = np.flatnonzero(session_folds != fold)
        heldout_record_indices = np.flatnonzero(record_folds == fold)
        graph_sessions = [train_sessions[int(index)] for index in graph_session_indices]
        if heldout_record_indices.size == 0:
            raise RuntimeError("Cross-fit fold %d has no train records" % fold)
        print(
            "[crossfit] fold=%d/%d build_graph_sessions=%d heldout_records=%d"
            % (
                fold + 1,
                int(request_metadata["num_folds"]),
                graph_session_indices.size,
                heldout_record_indices.size,
            ),
            flush=True,
        )
        graph = build_cosine_graph(
            graph_sessions,
            item_count,
            anum,
            target_domain,
            graph_topk,
        )
        fold_records = [records[int(index)] for index in heldout_record_indices]
        values = raw_multihop_evidence(fold_records, graph, hops, chunk_size)
        output[heldout_record_indices] = values
        written[heldout_record_indices] = True
        elapsed = time.time() - fold_started
        print(
            "[crossfit] fold=%d/%d complete edges=%d elapsed=%.1fs"
            % (
                fold + 1,
                int(request_metadata["num_folds"]),
                graph.nnz,
                elapsed,
            ),
            flush=True,
        )
        fold_audits.append(
            {
                "fold": fold,
                "graph_sessions": int(graph_session_indices.size),
                "heldout_sessions": int(np.sum(session_folds == fold)),
                "heldout_records": int(heldout_record_indices.size),
                "graph_directed_edges": int(graph.nnz),
            }
        )
        output.flush()
        del values, graph
    if not bool(written.all()):
        raise RuntimeError("Cross-fit evidence did not write every train row")
    output.flush()
    del output
    temporary.replace(cache_path)
    audit = {
        "originating_session_overlap_count": 0,
        "records_written": int(written.sum()),
        "folds": fold_audits,
    }
    payload = {"request": request_metadata, "audit": audit}
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return np.load(cache_path, mmap_mode="r"), audit


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _inference_pass(bridge, adapter, loader, device: str, batch_limit: int) -> int:
    bridge.model.eval()
    if adapter is not None:
        adapter.eval()
    records = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_limit > 0 and batch_index >= batch_limit:
                break
            batch = to_device(batch, device)
            if adapter is None:
                bridge.score_candidates(batch, batch["candidates"])
            else:
                user = bridge.encode_user(batch)
                candidates = bridge.candidate_representations(
                    batch["candidates"]
                )
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
                    base_scores + details["score_shift"]
                else:
                    bridge.score_candidate_delta(
                        user, base_scores, details["shift"]
                    )
            records += int(batch["candidates"].size(0))
    return records


def benchmark_inference(
    bridge,
    adapter,
    loader,
    device: str,
    warmup_rounds: int,
    repetitions: int,
    batch_limit: int,
) -> dict:
    if warmup_rounds < 0 or repetitions <= 0 or batch_limit < 0:
        raise ValueError("invalid benchmark repetition or batch-limit setting")
    for _ in range(warmup_rounds):
        _inference_pass(bridge, None, loader, device, batch_limit)
        _inference_pass(bridge, adapter, loader, device, batch_limit)
    _synchronize(device)

    samples = {"base": [], "adapted": []}
    records = None
    for repetition in range(repetitions):
        order = ("base", "adapted")
        if repetition % 2:
            order = tuple(reversed(order))
        for method in order:
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
                allocated_before = torch.cuda.memory_allocated(device)
            else:
                allocated_before = 0
            _synchronize(device)
            start = time.perf_counter()
            observed = _inference_pass(
                bridge,
                None if method == "base" else adapter,
                loader,
                device,
                batch_limit,
            )
            _synchronize(device)
            seconds = time.perf_counter() - start
            if records is None:
                records = observed
            elif records != observed:
                raise RuntimeError("benchmark pass changed the record count")
            if str(device).startswith("cuda"):
                peak_allocated = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
            else:
                peak_allocated = peak_reserved = 0
            samples[method].append(
                {
                    "seconds": seconds,
                    "records_per_second": observed / max(seconds, 1e-12),
                    "peak_allocated_bytes": int(peak_allocated),
                    "incremental_peak_allocated_bytes": int(
                        max(0, peak_allocated - allocated_before)
                    ),
                    "peak_reserved_bytes": int(peak_reserved),
                }
            )

    def aggregate(method: str) -> dict:
        result = {"samples": samples[method]}
        for field in (
            "seconds",
            "records_per_second",
            "peak_allocated_bytes",
            "incremental_peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            values = np.asarray(
                [sample[field] for sample in samples[method]], dtype=np.float64
            )
            result[field + "_mean"] = float(values.mean())
            result[field + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return result

    base = aggregate("base")
    adapted = aggregate("adapted")
    return {
        "warmup_rounds": warmup_rounds,
        "repetitions": repetitions,
        "batch_limit": batch_limit,
        "records_per_pass": int(records or 0),
        "base": base,
        "adapted": adapted,
        "adapted_to_base_seconds_ratio": adapted["seconds_mean"]
        / max(base["seconds_mean"], 1e-12),
        "adapted_to_base_throughput_ratio": adapted[
            "records_per_second_mean"
        ]
        / max(base["records_per_second_mean"], 1e-12),
    }


def _ranks_from_scores(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    count_ties_as_worse: bool = False,
) -> torch.Tensor:
    positive = (scores * positive_mask).sum(dim=1, keepdim=True)
    negative = positive_mask.eq(0)
    if count_ties_as_worse:
        outranks = scores.ge(positive) & negative
    else:
        outranks = scores.gt(positive) & negative
    return 1 + outranks.sum(dim=1)


def _metrics_from_ranks(ranks: torch.Tensor) -> dict:
    result = {}
    rank_float = ranks.float()
    for cutoff in (1, 3, 5, 10):
        hit = ranks.le(cutoff).float()
        result["hr@%d" % cutoff] = float(hit.mean())
        result["recall@%d" % cutoff] = float(hit.mean())
        result["ndcg@%d" % cutoff] = float(
            (hit / torch.log2(rank_float + 1.0)).mean()
        )
    result["count"] = int(ranks.numel())
    return result


def _collect_linear_fusion_inputs(
    bridge,
    loader,
    device: str,
    statistics: dict,
    evidence_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect native scores and a fixed equal-hop calibrated evidence scalar."""
    bridge.model.eval()
    score_parts = []
    evidence_parts = []
    mask_parts = []
    candidate_parts = []
    mean = torch.as_tensor(statistics["mean"], device=device).float()
    std = torch.as_tensor(statistics["std"], device=device).float().clamp_min(1e-6)
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            user = bridge.encode_user(batch)
            base_scores = bridge.direct_scores_from_user(
                user, batch["candidates"]
            )
            calibrated = (
                (torch.log1p(batch["structural"].float().clamp_min(0.0)) - mean)
                / std
            ).clamp(-float(evidence_clip), float(evidence_clip))
            # Equal weights make this a fixed graph feature rather than a
            # learned adapter component. Lambda is the only tuned quantity.
            structural_score = calibrated.mean(dim=-1)
            score_parts.append(base_scores.float().cpu())
            evidence_parts.append(structural_score.float().cpu())
            mask_parts.append(batch["positive_mask"].float().cpu())
            candidate_parts.append(batch["candidates"].long().cpu())
    return (
        torch.cat(score_parts, dim=0),
        torch.cat(evidence_parts, dim=0),
        torch.cat(mask_parts, dim=0),
        torch.cat(candidate_parts, dim=0),
    )


def evaluate_linear_evidence_fusion(
    bridge,
    valid_loader,
    test_loader,
    device: str,
    statistics: dict,
    evidence_clip: float,
    lambda_grid: list[float],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Tune Base + lambda * calibrated graph evidence on validation only."""
    lambdas = sorted({float(value) for value in lambda_grid} | {0.0})
    if any(not np.isfinite(value) or value < 0.0 for value in lambdas):
        raise ValueError("direct-fusion lambdas must be finite and nonnegative")
    valid_base, valid_evidence, valid_mask, valid_candidates = _collect_linear_fusion_inputs(
        bridge, valid_loader, device, statistics, evidence_clip
    )
    test_base, test_evidence, test_mask, test_candidates = _collect_linear_fusion_inputs(
        bridge, test_loader, device, statistics, evidence_clip
    )

    validation = []
    for value in lambdas:
        metrics = ranking_metrics(
            valid_base + value * valid_evidence, valid_mask
        )
        validation.append({"lambda": value, **metrics})
    # NDCG@10 is the locked checkpoint-selection metric. Prefer the smaller
    # lambda on exact ties, so lambda=0 remains an auditable identity option.
    selected = max(validation, key=lambda row: (row["ndcg@10"], -row["lambda"]))
    selected_lambda = float(selected["lambda"])
    fused_test_scores = test_base + selected_lambda * test_evidence
    selected_valid_shift = selected_lambda * valid_evidence
    base_rank = _ranks_from_scores(test_base, test_mask)
    fused_rank = _ranks_from_scores(fused_test_scores, test_mask)
    # A graph-only probe has many exact zero-evidence ties. Count every tied
    # negative as worse for the positive to avoid candidate-zero optimism.
    graph_rank = _ranks_from_scores(
        test_evidence, test_mask, count_ties_as_worse=True
    )
    report = {
        "definition": (
            "native_score s_base(u,i) + lambda * q(u,i), where q is the "
            "equal-hop mean of training-reference calibrated clipped log1p "
            "structural evidence"
        ),
        "lambda_selection": "validation NDCG@10; smaller lambda breaks ties",
        "lambda_grid": lambdas,
        "validation_grid": validation,
        "selected_lambda": selected_lambda,
        "selected_at_upper_grid_boundary": selected_lambda == max(lambdas),
        "selected_validation": selected,
        "validation_score_scale": {
            "native_score_mean": float(valid_base.mean()),
            "native_score_std": float(valid_base.std(unbiased=False)),
            "structural_score_mean": float(valid_evidence.mean()),
            "structural_score_std": float(valid_evidence.std(unbiased=False)),
            "scaled_structural_score_mean": float(selected_valid_shift.mean()),
            "scaled_structural_score_std": float(selected_valid_shift.std(unbiased=False)),
        },
        "test": {
            "base": _metrics_from_ranks(base_rank),
            "direct_fusion": _metrics_from_ranks(fused_rank),
            "graph_only_pessimistic_ties": _metrics_from_ranks(graph_rank),
        },
        "graph_only_tie_rule": (
            "all negative candidates tied with the positive count ahead"
        ),
    }
    paired = {
        "base_rank": base_rank.numpy().astype(np.int64),
        "direct_fusion_rank": fused_rank.numpy().astype(np.int64),
        "graph_only_rank": graph_rank.numpy().astype(np.int64),
        "record_id": np.arange(test_base.shape[0], dtype=np.int64),
        "candidate_ids": test_candidates.numpy().astype(np.int64),
        "positive_ids": (test_candidates * test_mask.long()).sum(dim=1).numpy().astype(np.int64),
        "positive_mask": test_mask.numpy().astype(np.float32),
        "base_scores": test_base.numpy().astype(np.float32),
        "structural_scores": test_evidence.numpy().astype(np.float32),
        "direct_fusion_scores": fused_test_scores.numpy().astype(np.float32),
    }
    return report, paired


def main() -> None:
    pipeline_start = time.perf_counter()
    args = parser().parse_args()
    defaults = {
        "c2dsr": {
            "official_root": "external_repos/C2DSR",
            "backbone_config": "cerank/configs/c2dsr_native.json",
            "backbone_root": "cerank/results/c2dsr_native",
            "output_root": "cerank/results/c2dsr_tcseb",
        },
        "sasrec": {
            "official_root": ".",
            "backbone_config": "cerank/configs/sasrec_native.json",
            "backbone_root": "cerank/results/sasrec_native",
            "output_root": "cerank/results/sasrec_tcseb",
        },
    }[args.backbone]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.device = resolve_device(args.device)
    set_seed(args.seed)
    backbone_config = read_json(REPO_ROOT / args.backbone_config)
    adapter_config = read_json(REPO_ROOT / args.adapter_config)
    analysis_run_dir = None
    source_run_config = None
    if args.analysis_run_dir is not None:
        analysis_run_dir = Path(args.analysis_run_dir)
        if not analysis_run_dir.is_absolute():
            analysis_run_dir = REPO_ROOT / analysis_run_dir
        if args.analysis_output is None:
            raise ValueError("--analysis_output is required with --analysis_run_dir")
        source_run_config = read_json(analysis_run_dir / "run_config.json")
        for name in ("dataset", "direction", "seed"):
            if source_run_config[name] != getattr(args, name):
                raise ValueError(
                    "analysis source %s=%r does not match requested %r"
                    % (name, source_run_config[name], getattr(args, name))
                )
        adapter_config = copy.deepcopy(source_run_config["adapter_config"])
    if args.num_epochs is not None:
        adapter_config["num_epochs"] = args.num_epochs
    if args.patience is not None:
        adapter_config["patience"] = args.patience
    if args.batch_size is not None:
        adapter_config["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        adapter_config["eval_batch_size"] = args.eval_batch_size
    if args.num_workers is not None:
        adapter_config["num_workers"] = args.num_workers
    positive_fields = (
        "num_epochs",
        "patience",
        "batch_size",
        "eval_batch_size",
    )
    if any(adapter_config[name] <= 0 for name in positive_fields):
        raise ValueError("%s must all be positive" % (positive_fields,))
    session_limit = args.session_limit
    if args.check_only and session_limit is None:
        session_limit = 64
    _, anum, bnum, splits = load_splits(
        REPO_ROOT / args.data_root, args.dataset, session_limit
    )
    train_split_names = ("train_A", "train_B", "train_overlap")
    train_sessions = []
    for split_name in train_split_names:
        train_sessions.extend(splits[split_name])
    session_uids = None
    session_folds = None
    fold_manifest = None
    if args.evidence_protocol == "session_crossfit_k5":
        if args.crossfit_folds != 5:
            raise ValueError("session_crossfit_k5 requires --crossfit_folds 5")
        session_uids, session_folds, fold_manifest = build_session_fold_manifest(
            {name: len(splits[name]) for name in train_split_names},
            num_folds=args.crossfit_folds,
            seed=args.crossfit_seed,
        )
    train_records, train_stats = make_train_records(
        train_sessions,
        anum,
        bnum,
        args.direction,
        adapter_config["num_train_negatives"],
        args.seed,
        session_uids=session_uids,
        session_folds=session_folds,
        position_protocol=args.train_position_protocol,
    )
    if args.full_ranking:
        valid_records, valid_stats = make_full_rank_eval_records(
            splits["valid"], anum, bnum, args.direction
        )
        test_records, test_stats = make_full_rank_eval_records(
            splits["test"], anum, bnum, args.direction
        )
    else:
        valid_records, valid_stats = make_eval_records(
            splits["valid"],
            anum,
            bnum,
            args.direction,
            adapter_config["num_eval_negatives"],
            args.seed + 1,
        )
        test_records, test_stats = make_eval_records(
            splits["test"],
            anum,
            bnum,
            args.direction,
            adapter_config["num_eval_negatives"],
            args.seed + 2,
        )
    if not train_records or not valid_records or not test_records:
        raise RuntimeError("The fixed-direction split produced an empty dataset")

    spec = parse_direction(args.direction)
    item_count = anum + bnum + 1
    hops = adapter_config["structural_hops"]
    legacy_graph_compatible = (
        not args.disable_legacy_cache
        and args.evidence_protocol == "full_train_graph"
        and args.train_position_protocol == "expanded"
        and session_limit is None
        and all(hop in {1, 2, 3} for hop in hops)
        and adapter_config["graph_topk"] == 0
        and adapter_config["num_train_negatives"] == 100
    )
    legacy_evidence_compatible = (
        legacy_graph_compatible
        and not args.full_ranking
        and adapter_config["num_eval_negatives"] == 999
    )
    legacy_root = REPO_ROOT / args.legacy_cache_root
    old_graph_path = legacy_graph_path(
        legacy_root,
        args.dataset,
        args.direction,
        spec.target_domain,
    )
    if legacy_graph_compatible and old_graph_path.exists():
        graph = sp.load_npz(old_graph_path).tocsr().astype(np.float32)
        if graph.shape != (item_count, item_count):
            raise ValueError(
                "Legacy graph shape %s != (%d, %d)"
                % (graph.shape, item_count, item_count)
            )
        graph_source = {
            "kind": "validated_legacy_v3",
            "path": str(old_graph_path),
        }
    else:
        if args.evidence_protocol == "session_crossfit_k5":
            print(
                "[crossfit] build_full_train_graph sessions=%d "
                "(validation/test evidence)" % len(train_sessions),
                flush=True,
            )
            full_graph_started = time.time()
        graph = build_cosine_graph(
            train_sessions,
            item_count,
            anum,
            spec.target_domain,
            adapter_config["graph_topk"],
        )
        if args.evidence_protocol == "session_crossfit_k5":
            print(
                "[crossfit] full_train_graph complete edges=%d elapsed=%.1fs"
                % (graph.nnz, time.time() - full_graph_started),
                flush=True,
            )
        graph_source = {"kind": "generated_clean_runtime", "path": None}
    evidence_metadata = {
        "schema": 2,
        "dataset": args.dataset,
        "direction": args.direction,
        "target_domain": spec.target_domain,
        "representation": "raw_graph_evidence",
        "normalization": "fixed_log1p",
        "graph": "cosine_item_cooccurrence",
        "graph_topk": adapter_config["graph_topk"],
        "hops": hops,
        "train_sessions": len(train_sessions),
        "session_limit": session_limit,
        "train_evidence_protocol": args.evidence_protocol,
        "train_position_protocol": args.train_position_protocol,
        "crossfit_folds": (
            args.crossfit_folds
            if args.evidence_protocol == "session_crossfit_k5"
            else None
        ),
        "crossfit_seed": (
            args.crossfit_seed
            if args.evidence_protocol == "session_crossfit_k5"
            else None
        ),
        "fold_manifest_sha256": (
            fold_manifest["sha256"] if fold_manifest is not None else None
        ),
        "evaluation_candidates": (
            "all_target_items"
            if args.full_ranking
            else "1+%d_sampled" % adapter_config["num_eval_negatives"]
        ),
    }
    evidence_key = cache_key(evidence_metadata)
    cache_dir = (
        REPO_ROOT
        / args.cache_root
        / args.dataset
        / args.direction
        / ("seed%d_%s" % (args.seed, evidence_key))
    )
    evidence_sources = {}

    def load_split_evidence(split, records):
        legacy_path = legacy_evidence_path(
            legacy_root,
            args.dataset,
            args.seed,
            args.direction,
            split,
            spec.target_domain,
        )
        expected_shape = (len(records), len(records[0]["candidates"]), len(hops))
        split_legacy_compatible = (
            legacy_evidence_compatible
            or (split == "train" and legacy_graph_compatible)
        )
        if split_legacy_compatible and legacy_path.exists():
            legacy_shape = (
                len(records),
                len(records[0]["candidates"]),
                3,
            )
            legacy_values = load_legacy_v3_evidence(
                legacy_path, legacy_shape
            )
            alignment_error = validate_evidence_alignment(
                records, graph, legacy_values, sample_count=16
            )
            values = legacy_values[:, :, [hop - 1 for hop in hops]]
            if values.shape != expected_shape:
                raise RuntimeError("Legacy hop slicing produced wrong shape")
            evidence_sources[split] = {
                "kind": "validated_legacy_v3",
                "path": str(legacy_path),
                "alignment_max_abs_error": alignment_error,
            }
            return values
        clean_path = cache_dir / ("%s.npz" % split)
        values = evidence_for_split(
            records,
            graph,
            hops,
            adapter_config["structural_chunk_size"],
            clean_path,
            dict(evidence_metadata, split=split, records=len(records)),
        )
        evidence_sources[split] = {
            "kind": "clean_runtime_cache",
            "path": str(clean_path),
            "alignment_max_abs_error": 0.0,
        }
        return values

    crossfit_audit = None
    if args.evidence_protocol == "session_crossfit_k5":
        request_metadata = {
            "schema": 1,
            "protocol": "session_crossfit_k5_v1",
            "dataset": args.dataset,
            "direction": args.direction,
            "seed": args.seed,
            "num_folds": args.crossfit_folds,
            "crossfit_seed": args.crossfit_seed,
            "fold_manifest_sha256": fold_manifest["sha256"],
            "record_digest": _train_record_digest(train_records),
            "records": len(train_records),
            "candidates": len(train_records[0]["candidates"]),
            "hops": list(hops),
            "graph_topk": adapter_config["graph_topk"],
            "target_domain": spec.target_domain,
            "position_protocol": args.train_position_protocol,
        }
        train_crossfit_path = cache_dir / "train_crossfit.npy"
        train_evidence, crossfit_audit = crossfit_train_evidence(
            train_records,
            train_sessions,
            session_folds,
            item_count,
            anum,
            spec.target_domain,
            adapter_config["graph_topk"],
            hops,
            adapter_config["structural_chunk_size"],
            train_crossfit_path,
            request_metadata,
        )
        evidence_sources["train"] = {
            "kind": "session_crossfit_k5_v1",
            "path": str(train_crossfit_path),
            "metadata_path": str(
                train_crossfit_path.with_suffix(train_crossfit_path.suffix + ".json")
            ),
            "alignment_max_abs_error": 0.0,
        }
    else:
        train_evidence = load_split_evidence("train", train_records)
    valid_evidence = load_split_evidence("valid", valid_records)
    test_evidence = load_split_evidence("test", test_records)
    statistics = (
        evidence_statistics_streaming(train_evidence)
        if args.evidence_protocol == "session_crossfit_k5"
        else evidence_statistics(train_evidence)
    )
    if args.prepare_evidence_only:
        prepare_summary = {
            "dataset": args.dataset,
            "direction": args.direction,
            "seed": args.seed,
            "evidence_protocol": args.evidence_protocol,
            "train_position_protocol": args.train_position_protocol,
            "fold_manifest": fold_manifest,
            "crossfit_audit": crossfit_audit,
            "train_records": len(train_records),
            "valid_records": len(valid_records),
            "test_records": len(test_records),
            "cache_dir": str(cache_dir),
            "evidence_statistics": {
                name: value.tolist() for name, value in statistics.items()
            },
            "structural_evidence_sources": evidence_sources,
        }
        write_json(cache_dir / "prepare_summary.json", prepare_summary)
        print(
            "[crossfit-prepare] dataset=%s records=%d overlap=%s cache=%s"
            % (
                args.dataset,
                len(train_records),
                (
                    crossfit_audit["originating_session_overlap_count"]
                    if crossfit_audit is not None
                    else "n/a"
                ),
                cache_dir,
            ),
            flush=True,
        )
        return
    sequence_length = backbone_config["sequence_length"]
    evidence_control = adapter_config.get("evidence_control", "aligned")
    if evidence_control not in {
        "aligned",
        "shuffled",
        "partial_shuffle",
        "dropout",
    }:
        raise ValueError(
            "evidence_control must be aligned, shuffled, "
            "partial_shuffle, or dropout"
        )
    corruption_rate = float(adapter_config.get("evidence_corruption_rate", 0.0))
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError("evidence_corruption_rate must lie in [0, 1]")
    control_seed = args.seed + 7000 if evidence_control != "aligned" else None
    train_set = EvidenceDataset(
        train_records,
        sequence_length,
        train_evidence,
        control_seed,
        evidence_control,
        corruption_rate,
    )
    valid_set = EvidenceDataset(
        valid_records,
        sequence_length,
        valid_evidence,
        None if control_seed is None else control_seed + 1,
        evidence_control,
        corruption_rate,
    )
    test_set = EvidenceDataset(
        test_records,
        sequence_length,
        test_evidence,
        None if control_seed is None else control_seed + 2,
        evidence_control,
        corruption_rate,
    )
    loader_runtime = data_loader_runtime_kwargs(adapter_config, args.device)
    train_loader = DataLoader(
        train_set,
        batch_size=min(adapter_config["batch_size"], len(train_set)),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_runtime,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=adapter_config["eval_batch_size"],
        shuffle=False,
        **loader_runtime,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=adapter_config["eval_batch_size"],
        shuffle=False,
        **loader_runtime,
    )

    if args.backbone in {"sasrec", "gru4rec"}:
        bridge_class = (
            NativeSASRecBridge
            if args.backbone == "sasrec"
            else NativeGRU4RecBridge
        )
        bridge = bridge_class(
            direction=args.direction,
            anum=anum,
            bnum=bnum,
            device=args.device,
            sequence_length=sequence_length,
            embedding_size=backbone_config["embedding_size"],
            hidden_size=backbone_config["hidden_size"],
            num_layers=backbone_config["num_layers"],
            dropout=backbone_config["dropout"],
            num_heads=backbone_config["num_heads"],
            inner_size=backbone_config["inner_size"],
        )
    else:
        bridge = NativeC2DSRBridge(
            official_root=REPO_ROOT / args.official_root,
            direction=args.direction,
            anum=anum,
            bnum=bnum,
            train_sessions=train_sessions,
            device=args.device,
            sequence_length=sequence_length,
            hidden_size=backbone_config["hidden_size"],
            num_blocks=backbone_config["num_blocks"],
            num_heads=backbone_config["num_heads"],
            gnn_layers=backbone_config["gnn_layers"],
            dropout=backbone_config["dropout"],
            leaky=backbone_config["leaky"],
        )
    backbone_dir = (
        REPO_ROOT
        / args.backbone_root
        / args.dataset
        / ("%s_%s_native_s%d" % (args.direction, args.backbone, args.seed))
    )
    checkpoint_path = backbone_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Train the native backbone first; missing %s" % checkpoint_path
        )
    backbone_payload = bridge.load_checkpoint(checkpoint_path)
    bridge.freeze()
    digest_before = bridge.parameter_digest()

    adapter = CERankAdapter(
        embedding_dim=bridge.hidden_size,
        evidence_channels=len(hops),
        hidden_size=adapter_config["adapter_hidden_size"],
        dropout=adapter_config["dropout"],
        residual_cap=adapter_config["residual_cap"],
        evidence_mean=statistics["mean"],
        evidence_std=statistics["std"],
        evidence_positive_mean=statistics["positive_mean"],
        evidence_clip=adapter_config["evidence_clip"],
        trust_temperature=adapter_config["trust_temperature"],
        trust_floor=adapter_config["trust_floor"],
        zero_init_delta=adapter_config.get("zero_init_delta", True),
        use_neural_features=adapter_config.get("use_neural_features", True),
        use_structural_features=adapter_config.get(
            "use_structural_features", True
        ),
        use_need_gate=adapter_config.get("use_need_gate", True),
        use_history_features=adapter_config.get("use_history_features", True),
        use_trust_gate=adapter_config.get("use_trust_gate", True),
        adaptation_space=adapter_config.get("adaptation_space", "candidate"),
    ).to(args.device)
    if analysis_run_dir is not None:
        adapter_checkpoint = analysis_run_dir / "best_adapter.pt"
        if not adapter_checkpoint.is_file():
            raise FileNotFoundError(
                "Completed adapter checkpoint is missing: %s" % adapter_checkpoint
            )
        adapter.load_state_dict(
            torch.load(adapter_checkpoint, map_location=args.device), strict=True
        )
        analysis_output = Path(args.analysis_output)
        if not analysis_output.is_absolute():
            analysis_output = REPO_ROOT / analysis_output
        prepare_seconds = time.perf_counter() - pipeline_start
        run_diagnostics = args.analysis_mode in ("all", "diagnostics")
        run_benchmark = args.analysis_mode in ("all", "benchmark")
        run_direct_fusion = args.analysis_mode == "direct_fusion"
        benchmark = None
        if run_benchmark:
            benchmark = benchmark_inference(
                bridge,
                adapter,
                test_loader,
                args.device,
                args.benchmark_warmup_rounds,
                args.benchmark_repetitions,
                args.benchmark_batch_limit,
            )
        base_test = adapted_test = test_aux = target_length_breakdown = None
        paired_path = None
        direct_fusion = None
        if run_diagnostics:
            base_test = evaluate_backbone(bridge, test_loader, args.device)
            adapted_test, test_aux = evaluate_adapter(
                bridge, adapter, test_loader, args.device
            )
            target_length_breakdown, paired_test = evaluate_by_target_length(
                bridge, adapter, test_loader, args.device
            )
            paired_path = analysis_output.with_name(
                analysis_output.stem + "_paired.npz"
            )
            paired_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(paired_path, **paired_test)
        if run_direct_fusion:
            lambda_grid = [
                float(value.strip())
                for value in args.direct_fusion_lambdas.split(",")
                if value.strip()
            ]
            direct_fusion, paired_fusion = evaluate_linear_evidence_fusion(
                bridge,
                valid_loader,
                test_loader,
                args.device,
                statistics,
                adapter_config["evidence_clip"],
                lambda_grid,
            )
            paired_path = analysis_output.with_name(
                analysis_output.stem + "_paired.npz"
            )
            paired_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(paired_path, **paired_fusion)
        digest_after = bridge.parameter_digest()
        if digest_after != digest_before:
            raise RuntimeError("Frozen backbone changed during analysis")
        cache_paths = {
            source["path"]
            for source in evidence_sources.values()
            if source.get("path")
        }
        if graph_source.get("path"):
            cache_paths.add(graph_source["path"])
        source_summary_path = analysis_run_dir / "metrics_summary.json"
        source_summary = (
            read_json(source_summary_path) if source_summary_path.is_file() else {}
        )
        report = {
            "schema": 2,
            "analysis_mode": args.analysis_mode,
            "source_run_dir": _display_path(analysis_run_dir),
            "dataset": args.dataset,
            "direction": args.direction,
            "seed": args.seed,
            "backbone": args.backbone,
            "protocol": source_run_config["protocol"],
            "evaluation_protocol": (
                "all target-domain items; observed prefix items retained"
                if args.full_ranking
                else "sampled-999"
            ),
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda")
                else "cpu"
            ),
            "torch_version": torch.__version__,
            "eval_batch_size": adapter_config["eval_batch_size"],
            "num_workers": adapter_config["num_workers"],
            "candidate_count": len(test_records[0]["candidates"]),
            "test_records": len(test_set),
            "pipeline_prepare_seconds": prepare_seconds,
            "training": {
                "best_epoch": source_summary.get("best_epoch"),
                "adapter_training_seconds": source_summary.get(
                    "efficiency", {}
                ).get("adapter_training_seconds"),
            },
            "parameters": {
                "frozen_backbone": sum(
                    parameter.numel() for parameter in bridge.model.parameters()
                ),
                "trainable_adapter": sum(
                    parameter.numel()
                    for parameter in adapter.parameters()
                    if parameter.requires_grad
                ),
            },
            "structural_cache": {
                "bytes": sum(
                    Path(path).stat().st_size
                    for path in cache_paths
                    if Path(path).exists()
                ),
                "files": len(cache_paths),
                "sources": evidence_sources,
            },
            "non_invasiveness": {
                "backbone_digest_before": digest_before,
                "backbone_digest_after": digest_after,
                "backbone_unchanged": digest_before == digest_after,
                "score_factorization_max_abs_error": source_summary.get(
                    "score_factorization_max_abs_error"
                ),
                "zero_adapter_identity_max_abs_error": source_summary.get(
                    "zero_adapter_identity_max_abs_error"
                ),
                "zero_adapter_identity_expected": source_summary.get(
                    "zero_adapter_identity_expected"
                ),
            },
        }
        if run_diagnostics:
            report.update(
                {
                    "base_test": base_test,
                    "adapted_test": adapted_test,
                    "adapted_test_aux": test_aux,
                    "target_length_breakdown": target_length_breakdown,
                    "paired_diagnostics": _display_path(paired_path),
                }
            )
        if run_benchmark:
            report["benchmark"] = benchmark
        if run_direct_fusion:
            report["direct_fusion"] = direct_fusion
            report["paired_diagnostics"] = _display_path(paired_path)
        report["parameters"]["adapter_to_backbone_ratio"] = report[
            "parameters"
        ]["trainable_adapter"] / max(
            report["parameters"]["frozen_backbone"], 1
        )
        write_json(analysis_output, report)
        print(json.dumps(report, indent=2), flush=True)
        return
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=adapter_config["learning_rate"],
        weight_decay=adapter_config["weight_decay"],
    )
    output_dir = (
        REPO_ROOT
        / args.output_root
        / args.dataset
        / (
            "%s_%s_native_tcseb_s%d"
            % (args.direction, args.backbone, args.seed)
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.jsonl"
    history_path.write_text("", encoding="utf-8")
    print(
        "[start] dataset=%s direction=%s train=%d valid=%d test=%d "
        "epochs=%d batch=%d workers=%d device=%s"
        % (
            args.dataset,
            args.direction,
            len(train_set),
            len(valid_set),
            len(test_set),
            1 if args.check_only else adapter_config["num_epochs"],
            adapter_config["batch_size"],
            adapter_config["num_workers"],
            args.device,
        ),
        flush=True,
    )

    check_batch = to_device(next(iter(valid_loader)), args.device)
    factorization_error = bridge.assert_score_identity(
        check_batch, check_batch["candidates"]
    )
    with torch.no_grad():
        check_user = bridge.encode_user(check_batch)
        check_candidates = bridge.candidate_representations(
            check_batch["candidates"]
        )
        check_base = bridge.direct_scores_from_user(
            check_user, check_batch["candidates"]
        )
        check_adapted, check_details = adapter(
            check_user,
            check_candidates,
            check_base,
            check_batch["structural"],
            check_batch["target_seq"],
        )
        if "score_shift" in check_details:
            check_scores = check_base + check_details["score_shift"]
        else:
            check_scores = bridge.score_candidate_delta(
                check_user, check_base, check_adapted - check_candidates
            )
        identity_error = float((check_scores - check_base).abs().max().cpu())
    expects_identity = adapter_config.get("zero_init_delta", True)
    if expects_identity and identity_error > 1e-6:
        raise RuntimeError(
            "Zero-initialized CERank is not identity: error=%g" % identity_error
        )

    base_eval_start = time.time()
    base_valid = evaluate_backbone(bridge, valid_loader, args.device)
    base_test = evaluate_backbone(bridge, test_loader, args.device)
    base_eval_seconds = time.time() - base_eval_start
    initial_valid = base_valid
    if not expects_identity:
        initial_valid, _ = evaluate_adapter(
            bridge, adapter, valid_loader, args.device
        )
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch = 0
    best_score = initial_valid[adapter_config["selection_metric"]]
    torch.save(best_state, output_dir / "best_adapter.pt")
    wait = 0
    history = [
        {
            "epoch": 0,
            "train": None,
            "valid": initial_valid,
            "identity_checkpoint": bool(expects_identity),
        }
    ]
    with history_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(history[0], ensure_ascii=False) + "\n")
    epochs = 1 if args.check_only else adapter_config["num_epochs"]
    adapter_train_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        adapter.train()
        totals = {
            "loss": 0.0,
            "recommendation": 0.0,
            "shift": 0.0,
            "gate": 0.0,
            "margin_noharm": 0.0,
        }
        steps = 0
        for batch in train_loader:
            batch = to_device(batch, args.device)
            with torch.no_grad():
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
            labels = torch.zeros(
                scores.size(0), dtype=torch.long, device=scores.device
            )
            recommendation = F.cross_entropy(scores, labels)
            shift_loss = details["shift"].pow(2).mean()
            gate_loss = details["gate"].mean()
            base_margin = base_scores[:, 0] - base_scores[:, 1:].max(dim=1).values
            final_margin = scores[:, 0] - scores[:, 1:].max(dim=1).values
            margin_loss = F.relu(
                base_margin
                - final_margin
                + adapter_config["noharm_margin"]
            ).mean()
            loss = (
                recommendation
                + adapter_config["lambda_shift"] * shift_loss
                + adapter_config["lambda_gate"] * gate_loss
                + adapter_config["lambda_margin_noharm"] * margin_loss
            )
            optimizer.zero_grad()
            loss.backward()
            if adapter_config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(), adapter_config["grad_clip"]
                )
            optimizer.step()
            steps += 1
            for key, value in (
                ("loss", loss),
                ("recommendation", recommendation),
                ("shift", shift_loss),
                ("gate", gate_loss),
                ("margin_noharm", margin_loss),
            ):
                totals[key] += float(value.detach().cpu())
            log_every = int(adapter_config.get("log_every_steps", 200))
            if log_every > 0 and steps % log_every == 0:
                print(
                    "[adapter] epoch=%d step=%d/%d loss=%.4f elapsed=%.1fs"
                    % (
                        epoch,
                        steps,
                        len(train_loader),
                        float(loss.detach().cpu()),
                        time.time() - epoch_start,
                    ),
                    flush=True,
                )
            if args.check_only:
                break
        valid, valid_aux = evaluate_adapter(
            bridge, adapter, valid_loader, args.device
        )
        record = {
            "epoch": epoch,
            "train": {
                key: value / max(steps, 1) for key, value in totals.items()
            },
            "valid": valid,
            "valid_aux": valid_aux,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        score = valid[adapter_config["selection_metric"]]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())
            torch.save(best_state, output_dir / "best_adapter.pt")
            wait = 0
        else:
            wait += 1
        print(
            "[epoch] %d train_loss=%.4f valid_%s=%.6f best=%.6f "
            "wait=%d/%d elapsed=%.1fs"
            % (
                epoch,
                record["train"]["loss"],
                adapter_config["selection_metric"],
                score,
                best_score,
                wait,
                adapter_config["patience"],
                time.time() - epoch_start,
            ),
            flush=True,
        )
        if wait >= adapter_config["patience"]:
            break
    adapter_train_seconds = time.time() - adapter_train_start

    adapter.load_state_dict(
        torch.load(output_dir / "best_adapter.pt", map_location=args.device),
        strict=True,
    )
    adapted_eval_start = time.time()
    adapted_valid, valid_aux = evaluate_adapter(
        bridge, adapter, valid_loader, args.device
    )
    adapted_test, test_aux = evaluate_adapter(
        bridge, adapter, test_loader, args.device
    )
    adapted_eval_seconds = time.time() - adapted_eval_start
    target_length_breakdown, paired_test_ranks = evaluate_by_target_length(
        bridge, adapter, test_loader, args.device
    )
    np.savez_compressed(
        output_dir / "paired_test_ranks.npz", **paired_test_ranks
    )
    digest_after = bridge.parameter_digest()
    if digest_after != digest_before:
        raise RuntimeError(
            "Frozen native %s parameters changed during adapter training"
            % args.backbone
        )

    run_config = {
        "dataset": args.dataset,
        "direction": args.direction,
        "seed": args.seed,
        "device": args.device,
        "evidence_protocol": args.evidence_protocol,
        "train_position_protocol": args.train_position_protocol,
        "crossfit_folds": args.crossfit_folds,
        "crossfit_seed": args.crossfit_seed,
        "protocol": (
            "fixed-direction target-visible full-ranking"
            if args.full_ranking
            else "fixed-direction target-visible sampled-999"
        ),
        "backbone": (
            (
                "untouched RecBole %s through target-only native bridge"
                % args.backbone
            )
            if args.backbone in {"sasrec", "gru4rec"}
            else "untouched official %s through native bridge" % args.backbone
        ),
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_frozen": True,
        "backbone_parameter_digest": digest_before,
        "official_provenance": bridge.provenance(),
        "official_model_modified": False,
        "candidate_representation": getattr(
            bridge,
            "candidate_representation_name",
            "official output-head candidate row",
        ),
        "candidate_bias_preserved": getattr(
            bridge, "candidate_bias_preserved", True
        ),
        "score_level_fusion": adapter.adaptation_space == "score",
        "adapter_config": adapter_config,
        "evidence": evidence_metadata,
        "structural_graph_source": graph_source,
        "structural_evidence_sources": evidence_sources,
        "evidence_cache_key": evidence_key,
        "evidence_statistics": {
            name: value.tolist() for name, value in statistics.items()
        },
        "fold_manifest": fold_manifest,
        "crossfit_audit": crossfit_audit,
        "data_stats": {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
        },
        "score_factorization_max_abs_error": factorization_error,
        "zero_adapter_identity_max_abs_error": identity_error,
        "zero_adapter_identity_expected": bool(expects_identity),
    }
    write_json(output_dir / "run_config.json", run_config)
    cache_paths = {
        source["path"]
        for source in evidence_sources.values()
        if source.get("path")
    }
    if graph_source.get("path"):
        cache_paths.add(graph_source["path"])
    structural_cache_bytes = sum(
        Path(path).stat().st_size
        for path in cache_paths
        if Path(path).exists()
    )
    evaluation_records = len(valid_set) + len(test_set)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_metric": best_score,
        "base": {"valid": base_valid, "test": base_test},
        "adapted": {
            "valid": adapted_valid,
            "test": adapted_test,
            "valid_aux": valid_aux,
            "test_aux": test_aux,
        },
        "score_factorization_max_abs_error": factorization_error,
        "zero_adapter_identity_max_abs_error": identity_error,
        "zero_adapter_identity_expected": bool(expects_identity),
        "backbone_digest_before": digest_before,
        "backbone_digest_after": digest_after,
        "backbone_unchanged": digest_before == digest_after,
        "target_length_breakdown": target_length_breakdown,
        "efficiency": {
            "adapter_trainable_parameters": sum(
                parameter.numel()
                for parameter in adapter.parameters()
                if parameter.requires_grad
            ),
            "frozen_backbone_parameters": sum(
                parameter.numel() for parameter in bridge.model.parameters()
            ),
            "base_valid_test_seconds": base_eval_seconds,
            "adapter_training_seconds": adapter_train_seconds,
            "adapted_valid_test_seconds": adapted_eval_seconds,
            "evaluation_records": evaluation_records,
            "base_records_per_second": evaluation_records
            / max(base_eval_seconds, 1e-9),
            "adapted_records_per_second": evaluation_records
            / max(adapted_eval_seconds, 1e-9),
            "adapted_to_base_eval_time_ratio": adapted_eval_seconds
            / max(base_eval_seconds, 1e-9),
            "structural_cache_bytes": structural_cache_bytes,
            "structural_cache_files": len(cache_paths),
        },
        "history": history,
        "smoke_only": bool(args.check_only),
    }
    write_json(output_dir / "metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
