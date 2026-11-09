"""Minimal bridge to the untouched official C2DSR implementation."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from ..protocol import DirectionSpec, parse_direction
from .base import NativeBackboneBridge


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    inverse[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    return sp.diags(inverse).dot(matrix).tocsr().astype(np.float32)


def _to_torch_sparse(matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    )
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, dtype=torch.float32, device=device
    ).coalesce()


class NativeC2DSRBridge(NativeBackboneBridge):
    """Protocol and scoring shell; the official C2DSR model is imported as-is."""

    OFFICIAL_FILES = (
        "model/C2DSR.py",
        "model/GNN.py",
    )

    def __init__(
        self,
        official_root: Path,
        direction: str,
        anum: int,
        bnum: int,
        train_sessions: Sequence[Sequence[int]],
        device: str = "cpu",
        sequence_length: int = 50,
        hidden_size: int = 256,
        num_blocks: int = 2,
        num_heads: int = 1,
        gnn_layers: int = 1,
        dropout: float = 0.2,
        leaky: float = 0.1,
    ) -> None:
        self.official_root = Path(official_root).resolve()
        self.official_src = self.official_root / "C2DSR_src"
        self.spec = parse_direction(direction)
        self.anum = int(anum)
        self.bnum = int(bnum)
        self.source_item_num = self.anum if self.spec.source_domain == "A" else self.bnum
        self.target_item_num = self.bnum if self.spec.target_domain == "B" else self.anum
        self.itemnum = self.source_item_num + self.target_item_num + 1
        self.pad_id = self.itemnum - 1
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            # The untouched official attention code calls `.cuda()` without an
            # explicit index. Pinning the process default keeps those temporary
            # masks on the requested device without patching official sources.
            torch.cuda.set_device(self.device)
        if not self.official_src.exists():
            raise FileNotFoundError("Missing official C2DSR source: %s" % self.official_src)

        # C2DSR.py uses absolute imports such as `from model.GNN import ...`.
        official_src_text = str(self.official_src)
        if official_src_text not in sys.path:
            sys.path.insert(0, official_src_text)
        official_module = importlib.import_module("model.C2DSR")
        official_class = getattr(official_module, "C2DSR")

        adjacency, single_adjacency = self._build_native_graphs(train_sessions)
        self.opt = {
            "model": "C2DSR",
            "source_item_num": self.source_item_num,
            "target_item_num": self.target_item_num,
            "itemnum": self.itemnum,
            "hidden_units": self.hidden_size,
            "num_blocks": int(num_blocks),
            "num_heads": int(num_heads),
            "GNN": int(gnn_layers),
            "dropout": float(dropout),
            "leakey": float(leaky),
            # Official positions are one-based. One extra embedding entry lets
            # the fixed protocol expose exactly sequence_length visible items.
            "maxlen": self.sequence_length + 1,
            "cuda": self.device.type == "cuda",
        }
        adjacency_tensor = _to_torch_sparse(adjacency, self.device)
        single_tensor = _to_torch_sparse(single_adjacency, self.device)
        self.model = official_class(self.opt, adjacency_tensor, single_tensor).to(
            self.device
        )
        # These tensors are plain attributes in the official implementation.
        self.model.adj = adjacency_tensor
        self.model.adj_single = single_tensor
        self.model.source_item_index = self.model.source_item_index.to(self.device)
        self.model.target_item_index = self.model.target_item_index.to(self.device)
        self.model.item_index = self.model.item_index.to(self.device)

    def _local_to_native_numpy(self, item: int) -> int:
        item = int(item)
        if item == 0:
            return self.pad_id
        if self.spec.direction == "A2B":
            if item <= self.anum:
                return item - 1
            return self.source_item_num + (item - self.anum - 1)
        if item > self.anum:
            return item - self.anum - 1
        return self.source_item_num + (item - 1)

    def _build_native_graphs(
        self, train_sessions: Sequence[Sequence[int]]
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        cross_rows, cross_cols = [], []
        single_rows, single_cols = [], []
        for session in train_sessions:
            mapped = [self._local_to_native_numpy(item) for item in session]
            previous = None
            previous_by_domain = {"X": None, "Y": None}
            for item in mapped:
                domain = "X" if item < self.source_item_num else "Y"
                if previous is not None:
                    cross_rows.append(previous)
                    cross_cols.append(item)
                domain_previous = previous_by_domain[domain]
                if domain_previous is not None:
                    single_rows.append(domain_previous)
                    single_cols.append(item)
                previous = item
                previous_by_domain[domain] = item

        def matrix(rows, cols):
            if not rows:
                return sp.csr_matrix((self.itemnum, self.itemnum), dtype=np.float32)
            value = sp.coo_matrix(
                (
                    np.ones(len(rows), dtype=np.float32),
                    (np.asarray(rows), np.asarray(cols)),
                ),
                shape=(self.itemnum, self.itemnum),
                dtype=np.float32,
            ).tocsr()
            return _row_normalize(value)

        return matrix(cross_rows, cross_cols), matrix(single_rows, single_cols)

    def _map_local_tensor(self, local_ids: torch.Tensor) -> torch.Tensor:
        local = local_ids.long()
        mapped = torch.full_like(local, self.pad_id)
        nonpad = local.ne(0)
        if self.spec.direction == "A2B":
            source = nonpad & local.le(self.anum)
            target = nonpad & local.gt(self.anum)
            mapped[source] = local[source] - 1
            mapped[target] = self.source_item_num + local[target] - self.anum - 1
        else:
            source = nonpad & local.gt(self.anum)
            target = nonpad & local.le(self.anum)
            mapped[source] = local[source] - self.anum - 1
            mapped[target] = self.source_item_num + local[target] - 1
        return mapped

    def _target_class_indices(self, local_ids: torch.Tensor) -> torch.Tensor:
        local = local_ids.long()
        if self.spec.target_domain == "A":
            if bool((local.gt(self.anum) | local.eq(0)).any()):
                raise ValueError("Non-target candidate supplied to B2A C2DSR bridge")
            return local - 1
        if bool((local.le(self.anum)).any()):
            raise ValueError("Non-target candidate supplied to A2B C2DSR bridge")
        return local - self.anum - 1

    def _native_inputs(
        self,
        mixed_local: torch.Tensor,
        target_local: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, ...]:
        mixed_local = mixed_local.to(self.device).long()
        if target_local is not None:
            target_local = target_local.to(self.device).long()
            if self.spec.target_domain == "A":
                local_target_mask = mixed_local.gt(0) & mixed_local.le(self.anum)
            else:
                local_target_mask = mixed_local.gt(self.anum)
            missing_target = ~local_target_mask.any(dim=1)
            if bool(missing_target.any()):
                mixed_local = mixed_local.clone()
                for row in missing_target.nonzero(as_tuple=False).flatten().tolist():
                    target_values = target_local[row][target_local[row].ne(0)]
                    if target_values.numel() == 0:
                        raise ValueError(
                            "Fixed protocol supplied an empty target_seq to Native C2DSR"
                        )
                    visible = mixed_local[row][mixed_local[row].ne(0)]
                    repaired = torch.cat(
                        [target_values[-1:], visible[-(mixed_local.size(1) - 1) :]]
                    )
                    mixed_local[row].zero_()
                    mixed_local[row, -repaired.numel() :] = repaired

        mixed = self._map_local_tensor(mixed_local)
        x_seq = torch.full_like(mixed, self.pad_id)
        y_seq = torch.full_like(mixed, self.pad_id)
        x_mask = mixed.lt(self.source_item_num)
        y_mask = mixed.ge(self.source_item_num) & mixed.lt(self.pad_id)
        x_seq[x_mask] = mixed[x_mask]
        y_seq[y_mask] = mixed[y_mask]
        visible = mixed.ne(self.pad_id)
        position = visible.long().cumsum(dim=1) * visible.long()
        x_position = x_mask.long().cumsum(dim=1) * x_mask.long()
        y_position = y_mask.long().cumsum(dim=1) * y_mask.long()
        if bool((~y_mask.any(dim=1)).any()):
            rows = (~y_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Native C2DSR target sequence is empty after bridge repair; rows=%s"
                % rows[:8]
            )
        return mixed, x_seq, y_seq, position, x_position, y_position, x_mask, y_mask

    def _features(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            mixed,
            x_seq,
            y_seq,
            position,
            x_position,
            y_position,
            x_mask,
            y_mask,
        ) = self._native_inputs(batch["mixed_seq"], batch.get("target_seq"))
        self.model.graph_convolution()
        shared, source_specific, target_specific = self.model(
            mixed, x_seq, y_seq, position, x_position, y_position
        )
        return shared, source_specific, target_specific, x_mask, y_mask

    @staticmethod
    def _last_masked(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("Native C2DSR target sequence must be nonempty")
        reverse_index = torch.flip(mask, dims=(1,)).float().argmax(dim=1)
        index = mask.size(1) - 1 - reverse_index
        return features[
            torch.arange(features.size(0), device=features.device), index
        ]

    def encode_user(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        shared, _, target_specific, _, target_mask = self._features(batch)
        return shared[:, -1] + self._last_masked(target_specific, target_mask)

    def candidate_representations(
        self, local_candidates: torch.Tensor
    ) -> torch.Tensor:
        indices = self._target_class_indices(local_candidates.to(self.device))
        return self.model.lin_Y.weight[indices]

    def candidate_bias(self, local_candidates: torch.Tensor) -> torch.Tensor:
        indices = self._target_class_indices(local_candidates.to(self.device))
        return self.model.lin_Y.bias[indices]

    def score_from_representations(
        self,
        user_state: torch.Tensor,
        local_candidates: torch.Tensor,
        candidate_representations: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.einsum("bd,bmd->bm", user_state, candidate_representations)
            + self.candidate_bias(local_candidates)
        )

    def score_candidates(
        self, batch: Dict[str, torch.Tensor], local_candidates: torch.Tensor
    ) -> torch.Tensor:
        # Native-backbone evaluation always follows the untouched official
        # output head. Factorized scoring is reserved for the bridge audit and
        # candidate-residual calculation.
        return self.direct_head_scores(batch, local_candidates)

    @staticmethod
    def score_candidate_delta(
        user_state: torch.Tensor,
        official_scores: torch.Tensor,
        candidate_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Add only the plugin's candidate-row residual to official scores."""
        return official_scores + torch.einsum(
            "bd,bmd->bm", user_state, candidate_delta
        )

    def direct_head_scores(
        self, batch: Dict[str, torch.Tensor], local_candidates: torch.Tensor
    ) -> torch.Tensor:
        user = self.encode_user(batch)
        return self.direct_scores_from_user(user, local_candidates)

    def direct_scores_from_user(
        self, user_state: torch.Tensor, local_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Apply the untouched official output head to an encoded user."""
        full_logits = self.model.lin_Y(user_state)
        indices = self._target_class_indices(local_candidates.to(self.device))
        return full_logits.gather(1, indices)

    def directional_training_loss(
        self,
        batch: Dict[str, torch.Tensor],
        native_lambda: float = 0.7,
    ) -> Tuple[torch.Tensor, dict]:
        """Target-direction branches of the official C2DSR objective.

        The official shared and target-specific output-head rows are retained.
        Their sampled softmax uses the common 1+K target candidates. Source-label
        losses are omitted because the common protocol admits target labels
        only. The official MI mechanism is applied only to examples with a
        visible source prefix.
        """
        shared, source_specific, target_specific, source_mask, target_mask = (
            self._features(batch)
        )
        target_last = self._last_masked(target_specific, target_mask)
        user = shared[:, -1] + target_last
        candidates = batch["candidates"].to(self.device)
        candidate_representations = self.candidate_representations(candidates)
        shared_scores = self.score_from_representations(
            shared[:, -1], candidates, candidate_representations
        )
        target_scores = self.score_from_representations(
            user, candidates, candidate_representations
        )
        labels = torch.zeros(
            candidates.size(0), dtype=torch.long, device=self.device
        )
        shared_loss = F.cross_entropy(shared_scores, labels)
        target_loss = F.cross_entropy(target_scores, labels)

        paired = source_mask.any(dim=1) & target_mask.any(dim=1)
        mi_loss = user.new_zeros(())
        if bool(paired.any()):
            source_weight = source_mask[paired].float()
            source_weight = source_weight / source_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            target_weight = target_mask[paired].float()
            target_weight = target_weight / target_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            r_x = torch.einsum(
                "bl,bld->bd", source_weight, source_specific[paired]
            )
            r_y = torch.einsum(
                "bl,bld->bd", target_weight, target_specific[paired]
            )
            real_x = torch.einsum("bl,bld->bd", source_weight, shared[paired])
            real_y = torch.einsum("bl,bld->bd", target_weight, shared[paired])
            # Batch permutation is the least invasive corruption that remains
            # well-defined for the fixed protocol and empty-source records.
            permutation = torch.roll(
                torch.arange(r_x.size(0), device=r_x.device), shifts=1
            )
            false_x = real_x[permutation]
            false_y = real_y[permutation]
            positive = torch.ones((r_x.size(0), 1), device=r_x.device)
            negative = torch.zeros_like(positive)
            mi_loss = (
                F.binary_cross_entropy_with_logits(
                    self.model.D_X(r_x, real_y), positive
                )
                + F.binary_cross_entropy_with_logits(
                    self.model.D_X(r_x, false_y), negative
                )
                + F.binary_cross_entropy_with_logits(
                    self.model.D_Y(r_y, real_x), positive
                )
                + F.binary_cross_entropy_with_logits(
                    self.model.D_Y(r_y, false_x), negative
                )
            )
        supervised = shared_loss + target_loss
        loss = float(native_lambda) * supervised + (1.0 - float(native_lambda)) * mi_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "shared_target_loss": float(shared_loss.detach().cpu()),
            "specific_target_loss": float(target_loss.detach().cpu()),
            "mi_loss": float(mi_loss.detach().cpu()),
            "paired_count": int(paired.sum().detach().cpu()),
        }

    def checkpoint_payload(self, extra_config: dict) -> dict:
        return {
            "model": self.model.state_dict(),
            "bridge_config": {
                "direction": self.spec.direction,
                "anum": self.anum,
                "bnum": self.bnum,
                "sequence_length": self.sequence_length,
                "opt": self.opt,
            },
            "run_config": extra_config,
            "official_provenance": self.provenance(),
        }

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        return payload

    def provenance(self) -> dict:
        files = {}
        for relative in self.OFFICIAL_FILES:
            path = self.official_src / relative
            files[relative] = _sha256(path)
        return {
            "backbone": "C2DSR",
            "official_root": str(self.official_root),
            "official_files_sha256": files,
            "official_model_modified": False,
            "bridge_source": str(Path(__file__).resolve()),
            "direction_mapping": {
                "X": self.spec.source_domain,
                "Y": self.spec.target_domain,
            },
            "scorer": "official lin_Y weight and bias",
        }
