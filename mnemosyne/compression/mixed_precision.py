"""Mixed precision policy — K4/V2 inspired allocation.

TurboQuant research shows that routing information (keys / index data)
needs higher precision than content (values / payload). Applied to
Mnemosyne's memory system:

  - SDI index embeddings (routing): 4-bit compression → accurate retrieval
  - Memory content embeddings (payload): 2-bit compression → aggressive savings
  - High-importance entries: preserved at higher precision regardless

This maps directly to the eternal-context ICMS/SDI split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .turboquant import TurboQuantCompressor, CompressedVector

logger = logging.getLogger("mnemosyne.compression.mixed_precision")


class PrecisionTier(Enum):
    """Precision tiers inspired by TurboQuant K4/V2 finding."""
    HIGH = 4      # 4-bit: routing/index embeddings, high-importance content
    STANDARD = 3  # 3-bit: standard memory content
    LOW = 2       # 2-bit: bulk archival, low-importance content


@dataclass
class PrecisionPolicy:
    """Rules for assigning precision tiers."""
    importance_threshold_high: float = 0.8
    importance_threshold_low: float = 0.3
    is_index: bool = False  # SDI index embeddings always get HIGH


class MixedPrecisionPolicy:
    """Manages multiple TurboQuant compressors at different bit widths.

    Allocates precision based on the role and importance of each memory entry:
      - SDI index embeddings → 4-bit (accurate retrieval is critical)
      - High-importance content → 4-bit (preserve fidelity)
      - Standard content → 3-bit (good balance)
      - Low-importance / archival → 2-bit (maximum compression)

    Average across a typical memory store: ~3 bits (same budget as uniform 3-bit
    but allocated where it matters, matching TurboQuant K4/V2 insight).
    """

    def __init__(self, qjl_dim: int = 64, seed: int = 42):
        self.compressors = {
            PrecisionTier.HIGH: TurboQuantCompressor(bits=4, qjl_dim=qjl_dim, seed=seed),
            PrecisionTier.STANDARD: TurboQuantCompressor(bits=3, qjl_dim=qjl_dim, seed=seed + 1),
            PrecisionTier.LOW: TurboQuantCompressor(bits=2, qjl_dim=qjl_dim, seed=seed + 2),
        }
        self.policy = PrecisionPolicy()

        # Track allocation stats
        self._tier_counts = {t: 0 for t in PrecisionTier}

    def select_tier(self, importance: float, is_index: bool = False) -> PrecisionTier:
        """Select compression tier based on entry characteristics."""
        if is_index:
            return PrecisionTier.HIGH
        if importance >= self.policy.importance_threshold_high:
            return PrecisionTier.HIGH
        if importance <= self.policy.importance_threshold_low:
            return PrecisionTier.LOW
        return PrecisionTier.STANDARD

    def compress(
        self,
        vector: np.ndarray,
        importance: float = 0.5,
        is_index: bool = False,
    ) -> tuple[CompressedVector, PrecisionTier]:
        """Compress with importance-aware precision selection."""
        tier = self.select_tier(importance, is_index)
        compressor = self.compressors[tier]
        compressed = compressor.compress(vector)
        self._tier_counts[tier] += 1
        return compressed, tier

    def decompress(self, cv: CompressedVector) -> np.ndarray:
        """Decompress using the correct tier's compressor."""
        tier = PrecisionTier(cv.bits)
        return self.compressors[tier].decompress(cv)

    def inner_product(self, a: CompressedVector, b: CompressedVector) -> float:
        """Inner product between vectors potentially at different precisions.

        When precisions differ, use the higher-precision compressor
        to decompress both, then compute directly.
        """
        if a.bits == b.bits:
            tier = PrecisionTier(a.bits)
            return self.compressors[tier].inner_product(a, b)

        # Mixed precision: decompress both and compute directly
        vec_a = self.decompress(a)
        vec_b = self.decompress(b)
        return float(np.dot(vec_a, vec_b))

    def average_bits(self) -> float:
        """Average bits per coordinate across all compressed vectors."""
        total = sum(self._tier_counts.values())
        if total == 0:
            return 0.0
        weighted = sum(
            tier.value * count
            for tier, count in self._tier_counts.items()
        )
        return weighted / total

    def get_stats(self) -> dict:
        return {
            "tier_counts": {t.name: c for t, c in self._tier_counts.items()},
            "average_bits": self.average_bits(),
            "per_tier": {
                t.name: self.compressors[t].get_stats()
                for t in PrecisionTier
            },
        }
