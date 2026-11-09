"""Minimal bridges to the untouched RecBole SASRec and GRU4Rec classes.

The fixed-direction protocol stores visible histories with left padding.
RecBole sequential recommenders instead gather ``item_seq_len - 1`` and
therefore require right-padded inputs.  The conversion below is deliberately
kept in the bridge; no installed RecBole source is modified.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..protocol import parse_direction
from .base import NativeBackboneBridge

try:
    import recbole
    from recbole.model.sequential_recommender import GRU4Rec, SASRec
except Exception as exc:  # pragma: no cover - reported when bridge is selected.
    recbole = None
    GRU4Rec = SASRec = None
    RECBOLE_IMPORT_ERROR = exc
else:
    RECBOLE_IMPORT_ERROR = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RecBoleDataset:
    """Small dataset interface required by RecBole model constructors."""

    def __init__(self, item_count: int) -> None:
        self.item_count = int(item_count)

    def num(self, unused_field: str) -> int:
        return self.item_count


class NativeRecBoleSequentialBridge(NativeBackboneBridge):
    """Target-only sequential backbone with an audited RecBole scorer."""

    candidate_representation_name = "untouched RecBole tied item-embedding row"
    candidate_bias_preserved = True
    uses_full_catalog_training = True

    def __init__(
        self,
        model_name: str,
        direction: str,
        anum: int,
        bnum: int,
        device: str,
        sequence_length: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_heads: int = 2,
        inner_size: int = 256,
    ) -> None:
        if RECBOLE_IMPORT_ERROR is not None:
            raise ImportError(
                "RecBole is required for the SASRec/GRU4Rec bridges"
            ) from RECBOLE_IMPORT_ERROR
        name = str(model_name).lower()
        if name not in {"sasrec", "gru4rec"}:
            raise ValueError("model_name must be sasrec or gru4rec")
        self.model_name = name
        self.spec = parse_direction(direction)
        self.anum = int(anum)
        self.bnum = int(bnum)
        self.item_count = self.anum + self.bnum + 1
        self.native_item_count = (
            self.anum + 1
            if self.spec.target_domain == "A"
            else self.bnum + 1
        )
        self.device = torch.device(device)
        self.sequence_length = int(sequence_length)
        self.embedding_size = int(embedding_size)
        # NativeBackboneBridge.hidden_size denotes the candidate space used by
        # CERank, not GRU4Rec's internal recurrent width.
        self.hidden_size = self.embedding_size
        self.recurrent_hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.inner_size = int(inner_size)
        self.dropout = float(dropout)

        config = {
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "LIST_SUFFIX": "_list",
            "ITEM_LIST_LENGTH_FIELD": "item_length",
            "NEG_PREFIX": "neg_",
            "MAX_ITEM_LIST_LENGTH": self.sequence_length,
            "device": self.device,
            "loss_type": "CE",
            "embedding_size": self.embedding_size,
            "hidden_size": (
                self.embedding_size
                if self.model_name == "sasrec"
                else self.recurrent_hidden_size
            ),
            "num_layers": self.num_layers,
            "dropout_prob": self.dropout,
            "n_layers": self.num_layers,
            "n_heads": self.num_heads,
            "inner_size": self.inner_size,
            "hidden_dropout_prob": self.dropout,
            "attn_dropout_prob": self.dropout,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
        }
        model_class = SASRec if self.model_name == "sasrec" else GRU4Rec
        self.model = model_class(
            config, _RecBoleDataset(self.native_item_count)
        ).to(self.device)

    def _native_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        values = candidates.to(self.device).long()
        if self.spec.target_domain == "A":
            invalid = values.eq(0) | values.gt(self.anum)
            native = values
        else:
            invalid = values.le(self.anum) | values.ge(self.item_count)
            native = values - self.anum
        if bool(invalid.any()):
            raise ValueError(
                "%s received a non-target candidate" % self.__class__.__name__
            )
        return native

    @staticmethod
    def right_pad_visible(left_padded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compact a left-padded batch without changing visible item order."""
        values = left_padded.long()
        lengths = values.ne(0).sum(dim=1)
        if bool(lengths.eq(0).any()):
            raise ValueError("Target-only sequential history must be nonempty")
        width = values.size(1)
        positions = torch.arange(width, device=values.device).unsqueeze(0)
        source_positions = positions + (width - lengths).unsqueeze(1)
        compacted = values.gather(1, source_positions.clamp(max=width - 1))
        right_padded = torch.where(
            positions < lengths.unsqueeze(1),
            compacted,
            torch.zeros_like(compacted),
        )
        return right_padded, lengths

    def _native_sequence(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = batch["target_seq"].to(self.device).long()
        if target.size(1) != self.sequence_length:
            raise ValueError(
                "Protocol sequence length %d != RecBole length %d"
                % (target.size(1), self.sequence_length)
            )
        if self.spec.target_domain == "B":
            target = torch.where(
                target.ne(0), target - self.anum, torch.zeros_like(target)
            )
        return self.right_pad_visible(target)

    def encode_user(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        sequence, lengths = self._native_sequence(batch)
        return self.model.forward(sequence, lengths)

    def candidate_representations(
        self, local_candidates: torch.Tensor
    ) -> torch.Tensor:
        candidates = self._native_candidates(local_candidates)
        return self.model.item_embedding(candidates)

    def score_from_representations(
        self,
        user_state: torch.Tensor,
        local_candidates: torch.Tensor,
        candidate_representations: torch.Tensor,
    ) -> torch.Tensor:
        self._native_candidates(local_candidates)
        return torch.einsum(
            "bd,bmd->bm", user_state, candidate_representations
        )

    def direct_scores_from_user(
        self, user_state: torch.Tensor, local_candidates: torch.Tensor
    ) -> torch.Tensor:
        candidates = self._native_candidates(local_candidates)
        full_scores = user_state @ self.model.item_embedding.weight.t()
        return full_scores.gather(1, candidates)

    def direct_head_scores(
        self, batch: Dict[str, torch.Tensor], local_candidates: torch.Tensor
    ) -> torch.Tensor:
        candidates = self._native_candidates(local_candidates)
        sequence, lengths = self._native_sequence(batch)
        interaction = {
            self.model.ITEM_SEQ: sequence,
            self.model.ITEM_SEQ_LEN: lengths,
        }
        full_scores = self.model.full_sort_predict(interaction)
        return full_scores.gather(1, candidates)

    def score_candidates(
        self, batch: Dict[str, torch.Tensor], local_candidates: torch.Tensor
    ) -> torch.Tensor:
        # Candidate-row factorization is the native RecBole predict equation.
        user = self.encode_user(batch)
        representations = self.candidate_representations(local_candidates)
        return self.score_from_representations(
            user, local_candidates, representations
        )

    @staticmethod
    def score_candidate_delta(
        user_state: torch.Tensor,
        official_scores: torch.Tensor,
        candidate_delta: torch.Tensor,
    ) -> torch.Tensor:
        return official_scores + torch.einsum(
            "bd,bmd->bm", user_state, candidate_delta
        )

    def directional_training_loss(
        self, batch: Dict[str, torch.Tensor], unused_native_lambda: float = None
    ) -> Tuple[torch.Tensor, dict]:
        del unused_native_lambda
        user = self.encode_user(batch)
        target_weights = self.model.item_embedding.weight[1:]
        logits = user @ target_weights.t()
        labels = batch["label"].to(self.device).long()
        if self.spec.target_domain == "B":
            labels = labels - self.anum
        labels = labels - 1
        recommendation = F.cross_entropy(logits, labels)
        value = float(recommendation.detach().cpu())
        return recommendation, {
            "loss": value,
            "recommendation_loss": value,
        }

    def checkpoint_payload(self, extra_config: dict) -> dict:
        return {
            "model": self.model.state_dict(),
            "bridge_config": {
                "model_name": self.model_name,
                "direction": self.spec.direction,
                "anum": self.anum,
                "bnum": self.bnum,
                "native_item_count": self.native_item_count,
                "sequence_length": self.sequence_length,
                "embedding_size": self.embedding_size,
                "hidden_size": self.recurrent_hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "inner_size": self.inner_size,
                "dropout": self.dropout,
            },
            "run_config": extra_config,
            "official_provenance": self.provenance(),
        }

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        return payload

    def provenance(self) -> dict:
        model_file = Path(inspect.getfile(self.model.__class__)).resolve()
        package_root = Path(inspect.getfile(recbole)).resolve().parent
        abstract_file = (
            package_root / "model" / "abstract_recommender.py"
        ).resolve()
        return {
            "backbone": self.model_name.upper(),
            "implementation": "RecBole",
            "recbole_version": recbole.__version__,
            "installed_package_root": str(package_root),
            "official_files": {
                str(model_file): _sha256(model_file),
                str(abstract_file): _sha256(abstract_file),
            },
            "official_model_modified": False,
            "native_wrapper": str(Path(__file__).resolve()),
            "protocol_translation": (
                "fixed target-only left-padded prefixes are compacted to "
                "right-padded RecBole item_seq tensors and global target IDs "
                "are mapped to a target-only catalog; source histories are "
                "not exposed to the standard backbone"
            ),
            "scorer": (
                "untouched RecBole tied item embedding with inner-product "
                "ranking and target-domain full-softmax CE"
            ),
        }


class NativeSASRecBridge(NativeRecBoleSequentialBridge):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_name="sasrec", **kwargs)


class NativeGRU4RecBridge(NativeRecBoleSequentialBridge):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_name="gru4rec", **kwargs)
