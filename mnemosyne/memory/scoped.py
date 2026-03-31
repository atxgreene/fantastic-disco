"""Scoped Memory Hierarchy — layered context inspired by Claude Code's CLAUDE.md.

Claude Code uses a hierarchy: project → user → team memory.
Mnemosyne extends this to: session → project → user → collective.

Each scope has different:
  - Lifetime: session (ephemeral) → collective (permanent)
  - Visibility: session (private) → collective (shared)
  - Compression: session (uncompressed) → collective (heavily compressed)
  - Authority: session (tentative) → collective (canonical)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger("mnemosyne.memory.scoped")


class MemoryScope(IntEnum):
    """Memory scopes ordered by lifetime and breadth."""
    SESSION = 1     # Current conversation only — ephemeral, high-detail
    PROJECT = 2     # Persists across sessions within a project
    USER = 3        # Persists across all projects for this user
    COLLECTIVE = 4  # Shared across all users/agents — canonical knowledge


@dataclass
class ScopedEntry:
    """A memory entry with scope metadata."""
    content: str
    scope: MemoryScope
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None
    source_scope: Optional[MemoryScope] = None  # Where it was promoted from
    promotion_count: int = 0  # How many times promoted up
    project_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class ScopeConfig:
    """Configuration for a memory scope."""
    max_entries: int
    compression_bits: int  # TurboQuant bit width for this scope
    ttl_hours: Optional[float]  # Time-to-live (None = permanent)
    promotion_threshold: float  # Importance needed to promote to next scope


DEFAULT_SCOPE_CONFIGS = {
    MemoryScope.SESSION: ScopeConfig(
        max_entries=500,
        compression_bits=0,  # No compression — full precision
        ttl_hours=24.0,
        promotion_threshold=0.7,
    ),
    MemoryScope.PROJECT: ScopeConfig(
        max_entries=5000,
        compression_bits=4,  # High precision (K4 — routing fidelity)
        ttl_hours=None,  # Permanent
        promotion_threshold=0.8,
    ),
    MemoryScope.USER: ScopeConfig(
        max_entries=20000,
        compression_bits=3,  # Standard compression
        ttl_hours=None,
        promotion_threshold=0.9,
    ),
    MemoryScope.COLLECTIVE: ScopeConfig(
        max_entries=100000,
        compression_bits=2,  # Maximum compression (V2 — bulk storage)
        ttl_hours=None,
        promotion_threshold=1.0,  # Can't promote beyond collective
    ),
}


class ScopedMemoryHierarchy:
    """Manages memory across scopes with automatic promotion.

    Memories start in SESSION scope and get promoted up based on
    importance and access patterns. Higher scopes use more aggressive
    TurboQuant compression.
    """

    def __init__(
        self,
        configs: Optional[dict[MemoryScope, ScopeConfig]] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.configs = configs or DEFAULT_SCOPE_CONFIGS
        self.project_id = project_id
        self.user_id = user_id

        # Storage per scope
        self.stores: dict[MemoryScope, list[ScopedEntry]] = {
            scope: [] for scope in MemoryScope
        }

    def add(
        self,
        content: str,
        scope: MemoryScope = MemoryScope.SESSION,
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None,
    ) -> ScopedEntry:
        """Add a memory entry at the specified scope."""
        entry = ScopedEntry(
            content=content,
            scope=scope,
            importance=importance,
            embedding=embedding,
            project_id=self.project_id,
            user_id=self.user_id,
        )
        self.stores[scope].append(entry)

        # Check capacity
        config = self.configs[scope]
        if len(self.stores[scope]) > config.max_entries:
            self._evict_scope(scope)

        return entry

    def query(
        self,
        query_embedding: np.ndarray,
        scopes: Optional[list[MemoryScope]] = None,
        top_k: int = 10,
    ) -> list[tuple[ScopedEntry, float]]:
        """Query across scopes, returning entries with relevance scores.

        Searches all specified scopes (default: all) and merges results.
        Higher scopes get a slight authority bonus.
        """
        if scopes is None:
            scopes = list(MemoryScope)

        candidates: list[tuple[ScopedEntry, float]] = []

        for scope in scopes:
            for entry in self.stores[scope]:
                if entry.embedding is None:
                    continue

                # Cosine similarity
                norm_q = np.linalg.norm(query_embedding)
                norm_e = np.linalg.norm(entry.embedding)
                if norm_q < 1e-10 or norm_e < 1e-10:
                    continue
                sim = float(np.dot(query_embedding, entry.embedding) / (norm_q * norm_e))

                # Authority bonus: higher scopes are more authoritative
                authority_bonus = 0.05 * (scope.value - 1)
                score = sim + authority_bonus

                candidates.append((entry, score))

        # Sort by score, return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def promote(self, entry: ScopedEntry) -> Optional[ScopedEntry]:
        """Promote an entry to the next scope if it qualifies.

        Returns the promoted entry, or None if promotion conditions not met.
        """
        current_scope = entry.scope
        if current_scope == MemoryScope.COLLECTIVE:
            return None  # Already at top

        config = self.configs[current_scope]
        if entry.importance < config.promotion_threshold:
            return None

        next_scope = MemoryScope(current_scope.value + 1)

        promoted = ScopedEntry(
            content=entry.content,
            scope=next_scope,
            importance=entry.importance,
            embedding=entry.embedding.copy() if entry.embedding is not None else None,
            source_scope=current_scope,
            promotion_count=entry.promotion_count + 1,
            project_id=entry.project_id,
            user_id=entry.user_id,
        )

        self.stores[next_scope].append(promoted)
        logger.info(
            f"Promoted entry from {current_scope.name} → {next_scope.name} "
            f"(importance={entry.importance:.2f})"
        )
        return promoted

    def auto_promote(self) -> list[ScopedEntry]:
        """Scan all scopes and promote qualifying entries."""
        promoted = []
        for scope in [MemoryScope.SESSION, MemoryScope.PROJECT, MemoryScope.USER]:
            for entry in self.stores[scope]:
                result = self.promote(entry)
                if result:
                    promoted.append(result)
        return promoted

    def get_scope_stats(self) -> dict:
        """Stats per scope."""
        stats = {}
        for scope in MemoryScope:
            config = self.configs[scope]
            count = len(self.stores[scope])
            stats[scope.name] = {
                "count": count,
                "capacity": config.max_entries,
                "utilization": count / config.max_entries if config.max_entries > 0 else 0,
                "compression_bits": config.compression_bits,
                "ttl_hours": config.ttl_hours,
            }
        return stats

    def _evict_scope(self, scope: MemoryScope) -> None:
        """Evict lowest-importance entries from a scope."""
        config = self.configs[scope]
        entries = self.stores[scope]
        if len(entries) <= config.max_entries:
            return

        # Sort by importance, keep top max_entries
        entries.sort(key=lambda e: e.importance, reverse=True)

        # Try to promote evicted entries before discarding
        evicted = entries[config.max_entries:]
        for entry in evicted:
            self.promote(entry)

        self.stores[scope] = entries[:config.max_entries]
        logger.debug(f"Evicted {len(evicted)} entries from {scope.name}")
