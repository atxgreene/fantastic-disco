"""Compression-Aware SDI — extends eternal-context's SDI with compression cost.

The original SDI scores entries by:
  importance (0.4) + recency (0.2) + uniqueness (0.2) + relevance (0.2)

This extension adds compression awareness: compressed entries cost fewer
tokens to include in context, so the SDI should prefer them when budgets
are tight. A 7x-compressed entry that's 80% as relevant as an uncompressed
one is often the better choice because it leaves room for more context.

Also integrates with the scoped memory hierarchy — higher scopes get
an authority bonus, and mixed-precision compression ratios factor into
the token budget calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .scoped import MemoryScope

logger = logging.getLogger("mnemosyne.memory.compressed_sdi")


@dataclass
class SDICandidate:
    """A candidate entry for SDI context selection."""
    entry_id: int
    content: str
    token_count: int                   # Actual tokens consumed if selected
    uncompressed_tokens: int           # Original token count before compression
    importance: float
    recency: float
    uniqueness: float
    relevance: float                   # Cosine similarity to query
    scope: MemoryScope = MemoryScope.SESSION
    compression_ratio: float = 1.0     # > 1.0 means compressed
    embedding: Optional[np.ndarray] = None


@dataclass
class SDIWeights:
    """Weights for SDI scoring factors."""
    importance: float = 0.35
    recency: float = 0.15
    uniqueness: float = 0.15
    relevance: float = 0.20
    efficiency: float = 0.10           # NEW: reward for compression efficiency
    authority: float = 0.05            # NEW: scope-based authority bonus

    def normalize(self) -> None:
        """Ensure weights sum to 1.0."""
        total = (
            self.importance + self.recency + self.uniqueness +
            self.relevance + self.efficiency + self.authority
        )
        if total > 0:
            self.importance /= total
            self.recency /= total
            self.uniqueness /= total
            self.relevance /= total
            self.efficiency /= total
            self.authority /= total


class CompressedSDI:
    """Semantic Density Index with compression and scope awareness.

    Extends the eternal-context SDI by adding:
    1. Efficiency score: compressed entries get a bonus per information-per-token
    2. Authority score: higher-scope entries are weighted as more authoritative
    3. Budget-aware selection: maximizes total information within token budget
    """

    def __init__(
        self,
        weights: Optional[SDIWeights] = None,
        context_budget_tokens: int = 3000,
    ):
        self.weights = weights or SDIWeights()
        self.weights.normalize()
        self.context_budget_tokens = context_budget_tokens

    def score(self, candidate: SDICandidate) -> float:
        """Compute the composite SDI score for a candidate entry."""
        w = self.weights

        # Standard factors (from eternal-context)
        s_importance = candidate.importance
        s_recency = candidate.recency
        s_uniqueness = candidate.uniqueness
        s_relevance = candidate.relevance

        # NEW: Efficiency — how much information per token?
        # A 7x compressed entry delivers 7x more original information per context token
        s_efficiency = min(1.0, candidate.compression_ratio / 10.0)

        # NEW: Authority — higher scopes are more authoritative
        s_authority = (candidate.scope.value - 1) / 3.0  # Normalized to [0, 1]

        score = (
            w.importance * s_importance +
            w.recency * s_recency +
            w.uniqueness * s_uniqueness +
            w.relevance * s_relevance +
            w.efficiency * s_efficiency +
            w.authority * s_authority
        )

        return score

    def select_context(
        self,
        candidates: list[SDICandidate],
        budget_tokens: Optional[int] = None,
    ) -> list[tuple[SDICandidate, float]]:
        """Select entries to fill context budget, maximizing total SDI score.

        Uses a greedy algorithm: sort by score/token ratio (bang per buck),
        then greedily fill the budget. This naturally favors compressed
        entries because they have higher score-per-token ratios.
        """
        budget = budget_tokens or self.context_budget_tokens

        # Score all candidates
        scored = [(c, self.score(c)) for c in candidates]

        # Sort by score-per-token (efficiency-weighted greedy)
        scored.sort(
            key=lambda x: x[1] / max(1, x[0].token_count),
            reverse=True,
        )

        selected: list[tuple[SDICandidate, float]] = []
        remaining_budget = budget

        for candidate, score in scored:
            if candidate.token_count <= remaining_budget:
                selected.append((candidate, score))
                remaining_budget -= candidate.token_count

            if remaining_budget <= 0:
                break

        # Re-sort selected by score (not efficiency) for output ordering
        selected.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"SDI selected {len(selected)}/{len(candidates)} entries, "
            f"using {budget - remaining_budget}/{budget} tokens"
        )
        return selected

    def adapt_weights(self, feedback_score: float) -> None:
        """Adapt weights based on feedback (from eternal-context pattern).

        If feedback is poor, shift weight toward relevance and efficiency.
        If feedback is good, maintain current balance.
        """
        if feedback_score < 0.5:
            # Poor quality: emphasize relevance and efficiency
            shift = 0.05 * (0.5 - feedback_score)
            self.weights.relevance += shift
            self.weights.efficiency += shift * 0.5
            self.weights.importance -= shift
            self.weights.recency -= shift * 0.5
            self.weights.normalize()
            logger.debug(f"SDI weights adapted (feedback={feedback_score:.2f})")

    def get_stats(self, candidates: list[SDICandidate]) -> dict:
        """Stats about the candidate pool."""
        if not candidates:
            return {"count": 0}

        compressions = [c.compression_ratio for c in candidates]
        return {
            "count": len(candidates),
            "avg_importance": sum(c.importance for c in candidates) / len(candidates),
            "avg_compression_ratio": sum(compressions) / len(compressions),
            "max_compression_ratio": max(compressions),
            "total_tokens_uncompressed": sum(c.uncompressed_tokens for c in candidates),
            "total_tokens_compressed": sum(c.token_count for c in candidates),
            "effective_compression": (
                sum(c.uncompressed_tokens for c in candidates) /
                max(1, sum(c.token_count for c in candidates))
            ),
            "scope_distribution": {
                scope.name: sum(1 for c in candidates if c.scope == scope)
                for scope in MemoryScope
            },
        }
