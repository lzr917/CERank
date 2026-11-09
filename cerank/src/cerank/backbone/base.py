"""Contract that every minimally invasive native backbone bridge must satisfy."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class NativeBackboneBridge(ABC):
    """Thin protocol/scoring interface around an untouched official model."""

    model: nn.Module
    hidden_size: int

    @abstractmethod
    def encode_user(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def candidate_representations(
        self, local_candidates: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_from_representations(
        self,
        user_state: torch.Tensor,
        local_candidates: torch.Tensor,
        candidate_representations: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def score_candidates(
        self, batch: Dict[str, torch.Tensor], local_candidates: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def direct_head_scores(
        self, batch: Dict[str, torch.Tensor], local_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Score through the official output head, without factorization helpers."""
        raise NotImplementedError

    @abstractmethod
    def direct_scores_from_user(
        self, user_state: torch.Tensor, local_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Apply the untouched official head to an already encoded user."""
        raise NotImplementedError

    @abstractmethod
    def score_candidate_delta(
        self,
        user_state: torch.Tensor,
        official_scores: torch.Tensor,
        candidate_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Add an adapted-candidate residual to untouched official scores."""
        raise NotImplementedError

    @abstractmethod
    def provenance(self) -> dict:
        raise NotImplementedError

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def assert_score_identity(
        self,
        batch: Dict[str, torch.Tensor],
        local_candidates: torch.Tensor,
        absolute_tolerance: float = 5e-4,
        relative_tolerance: float = 1e-5,
    ) -> float:
        """Audit mathematical scorer factorization under float32 CUDA.

        The direct head is a single GEMM over every item, whereas the bridge
        gathers per-request rows before a batched dot product.  Those kernels
        can accumulate float32 products in a different order, so this check
        uses an explicit absolute-plus-relative numerical tolerance.
        """
        self.model.eval()
        with torch.no_grad():
            direct = self.direct_head_scores(batch, local_candidates)
            user = self.encode_user(batch)
            representations = self.candidate_representations(local_candidates)
            factorized = self.score_from_representations(
                user, local_candidates, representations
            )
            difference = (direct - factorized).abs()
            error = float(difference.max().cpu())
            scale = torch.maximum(direct.abs(), factorized.abs())
            allowed = absolute_tolerance + relative_tolerance * scale
            failed = difference > allowed
        if bool(failed.any()):
            worst_ratio = float(
                (difference / allowed.clamp_min(torch.finfo(allowed.dtype).tiny))
                .max()
                .cpu()
            )
            raise RuntimeError(
                "Native scorer identity failed: max_abs_error=%g, "
                "max_tolerance_ratio=%g (atol=%g, rtol=%g)"
                % (
                    error,
                    worst_ratio,
                    absolute_tolerance,
                    relative_tolerance,
                )
            )
        return error
