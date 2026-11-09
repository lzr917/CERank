#!/usr/bin/env python3
"""Train a registered untouched native backbone under the fixed protocol."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
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
from cerank.protocol import (  # noqa: E402
    BackboneTrainDataset,
    DirectionalDataset,
    load_splits,
    make_eval_records,
    make_train_samples,
)
from cerank.training import (  # noqa: E402
    data_loader_runtime_kwargs,
    evaluate_backbone,
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
    result.add_argument("--config", default=None)
    result.add_argument("--output_root", default=None)
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
    result.add_argument(
        "--finalize_checkpoint",
        action="store_true",
        help="Audit/evaluate an existing best_model.pt without retraining",
    )
    result.add_argument("--session_limit", type=int, default=None)
    return result


def make_bridge(args, config, anum, bnum, train_sessions):
    if args.backbone in {"sasrec", "gru4rec"}:
        bridge_class = (
            NativeSASRecBridge
            if args.backbone == "sasrec"
            else NativeGRU4RecBridge
        )
        return bridge_class(
            direction=args.direction,
            anum=anum,
            bnum=bnum,
            device=args.device,
            sequence_length=config["sequence_length"],
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_heads=config["num_heads"],
            inner_size=config["inner_size"],
        )
    return NativeC2DSRBridge(
        official_root=REPO_ROOT / args.official_root,
        direction=args.direction,
        anum=anum,
        bnum=bnum,
        train_sessions=train_sessions,
        device=args.device,
        sequence_length=config["sequence_length"],
        hidden_size=config["hidden_size"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        gnn_layers=config["gnn_layers"],
        dropout=config["dropout"],
        leaky=config["leaky"],
    )


def main() -> None:
    args = parser().parse_args()
    defaults = {
        "c2dsr": {
            "official_root": "external_repos/C2DSR",
            "config": "cerank/configs/c2dsr_native.json",
            "output_root": "cerank/results/c2dsr_native",
        },
        "sasrec": {
            "official_root": ".",
            "config": "cerank/configs/sasrec_native.json",
            "output_root": "cerank/results/sasrec_native",
        },
    }[args.backbone]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.device = resolve_device(args.device)
    set_seed(args.seed)
    config = read_json(REPO_ROOT / args.config)
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.patience is not None:
        config["patience"] = args.patience
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    positive_fields = (
        "num_epochs",
        "patience",
        "batch_size",
        "eval_batch_size",
    )
    if any(config[name] <= 0 for name in positive_fields):
        raise ValueError("%s must all be positive" % (positive_fields,))
    session_limit = args.session_limit
    if args.check_only and session_limit is None:
        session_limit = 64
    _, anum, bnum, splits = load_splits(
        REPO_ROOT / args.data_root, args.dataset, session_limit
    )
    train_sessions = (
        splits["train_A"] + splits["train_B"] + splits["train_overlap"]
    )
    train_records, train_stats = make_train_samples(
        train_sessions, anum, args.direction
    )
    valid_records, valid_stats = make_eval_records(
        splits["valid"],
        anum,
        bnum,
        args.direction,
        config["num_eval_negatives"],
        args.seed + 1,
    )
    test_records, test_stats = make_eval_records(
        splits["test"],
        anum,
        bnum,
        args.direction,
        config["num_eval_negatives"],
        args.seed + 2,
    )
    if not train_records or not valid_records or not test_records:
        raise RuntimeError("The fixed-direction split produced an empty dataset")

    standard_target_only = args.backbone in {"sasrec", "gru4rec"}
    if standard_target_only:
        train_set = DirectionalDataset(
            train_records, config["sequence_length"]
        )
    else:
        train_set = BackboneTrainDataset(
            train_records,
            config["sequence_length"],
            anum,
            bnum,
            config["num_train_negatives"],
            args.seed,
        )
    valid_set = DirectionalDataset(valid_records, config["sequence_length"])
    test_set = DirectionalDataset(test_records, config["sequence_length"])
    loader_runtime = data_loader_runtime_kwargs(config, args.device)
    train_loader = DataLoader(
        train_set,
        batch_size=min(config["batch_size"], len(train_set)),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_runtime,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **loader_runtime,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **loader_runtime,
    )
    bridge = make_bridge(args, config, anum, bnum, train_sessions)
    output_dir = (
        REPO_ROOT
        / args.output_root
        / args.dataset
        / ("%s_%s_native_s%d" % (args.direction, args.backbone, args.seed))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    if args.finalize_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Cannot finalize missing checkpoint: %s" % checkpoint_path
            )
        payload = bridge.load_checkpoint(checkpoint_path)
        stored_run_config = payload.get("run_config", {})
        write_json(output_dir / "run_config.json", stored_run_config)
        check_batch = to_device(next(iter(valid_loader)), args.device)
        identity_error = bridge.assert_score_identity(
            check_batch, check_batch["candidates"]
        )
        valid = evaluate_backbone(bridge, valid_loader, args.device)
        test = evaluate_backbone(bridge, test_loader, args.device)
        summary = {
            "best_epoch": payload.get("checkpoint_epoch"),
            "best_validation_selection_metric": valid[
                config["selection_metric"]
            ],
            "valid": valid,
            "test": test,
            "score_factorization_max_abs_error": identity_error,
            "parameter_digest": bridge.parameter_digest(),
            "history": [],
            "history_unavailable": True,
            "recovered_checkpoint_audit": True,
            "smoke_only": False,
        }
        write_json(output_dir / "metrics_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return
    run_config = {
        "dataset": args.dataset,
        "direction": args.direction,
        "backbone": args.backbone,
        "seed": args.seed,
        "device": args.device,
        "protocol": "fixed-direction target-visible",
        "training_samples": "every valid target-domain next-item position",
        "training_objective": {
            "sasrec": (
                "untouched RecBole SASRec target-only encoder with "
                "target-domain full-softmax CE"
            ),
            "c2dsr": (
                "protocol-matched sampled softmax on official shared/target-"
                "specific heads plus paired-prefix MI"
            ),
        }[args.backbone],
        "training_candidates": (
            "full target-domain catalog"
            if standard_target_only
            else "1 positive + %d target negatives"
            % config["num_train_negatives"]
        ),
        "source_label_loss": False,
        "official_model_modified": False,
        "config": config,
        "data_stats": {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
        },
        "official_provenance": bridge.provenance(),
    }
    write_json(output_dir / "run_config.json", run_config)
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
            1 if args.check_only else config["num_epochs"],
            config["batch_size"],
            config["num_workers"],
            args.device,
        ),
        flush=True,
    )

    optimizer_name = config.get("optimizer", "adam").lower()
    optimizer_classes = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    if optimizer_name not in optimizer_classes:
        raise ValueError("Unsupported optimizer: %s" % optimizer_name)
    optimizer = optimizer_classes[optimizer_name](
        bridge.model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup_epochs = int(config.get("warmup_epochs", 0))
    warmup = None
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(config.get("warmup_start_factor", 1e-5)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
    epochs = 1 if args.check_only else config["num_epochs"]
    best_score = float("-inf")
    best_epoch = None
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        bridge.model.train()
        totals = {}
        steps = 0
        for batch in train_loader:
            batch = to_device(batch, args.device)
            optimizer.zero_grad()
            loss, parts = bridge.directional_training_loss(
                batch, config.get("native_lambda")
            )
            if hasattr(bridge, "pareto_objective"):
                loss = bridge.pareto_objective(loss, optimizer)
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    bridge.model.parameters(), config["grad_clip"]
                )
            optimizer.step()
            steps += 1
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            log_every = int(config.get("log_every_steps", 50))
            if log_every > 0 and steps % log_every == 0:
                print(
                    "[train] epoch=%d step=%d/%d loss=%.4f elapsed=%.1fs"
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
        valid = evaluate_backbone(bridge, valid_loader, args.device)
        if warmup is not None and epoch <= warmup_epochs:
            warmup.step()
        record = {
            "epoch": epoch,
            "train": {key: value / max(steps, 1) for key, value in totals.items()},
            "valid": valid,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        score = valid[config["selection_metric"]]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            checkpoint = bridge.checkpoint_payload(run_config)
            checkpoint["checkpoint_epoch"] = epoch
            checkpoint["checkpoint_validation"] = valid
            torch.save(checkpoint, checkpoint_path)
        else:
            wait += 1
        print(
            "[epoch] %d train_loss=%.4f valid_%s=%.6f best=%.6f "
            "wait=%d/%d elapsed=%.1fs"
            % (
                epoch,
                record["train"]["loss"],
                config["selection_metric"],
                score,
                best_score,
                wait,
                config["patience"],
                time.time() - epoch_start,
            ),
            flush=True,
        )
        if wait >= config["patience"]:
            break

    bridge.load_checkpoint(checkpoint_path)
    check_batch = to_device(next(iter(valid_loader)), args.device)
    identity_error = bridge.assert_score_identity(
        check_batch, check_batch["candidates"]
    )
    summary = {
        "best_epoch": best_epoch,
        "best_validation_selection_metric": best_score,
        "valid": evaluate_backbone(bridge, valid_loader, args.device),
        "test": evaluate_backbone(bridge, test_loader, args.device),
        "score_factorization_max_abs_error": identity_error,
        "parameter_digest": bridge.parameter_digest(),
        "history": history,
        "smoke_only": bool(args.check_only),
    }
    write_json(output_dir / "metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
