"""Consciousness Loop — the unified orchestrator.

This is the central integration point that makes Mnemosyne more than
the sum of its parts. It coordinates:

  Perception → Attention → Reasoning → Memory → Reflection → Dream

On every turn:
  1. PERCEIVE: Encode input, update cognitive state
  2. ATTEND:   SDI selects relevant memories across scopes
  3. REASON:   Route to appropriate model with context
  4. REMEMBER: Extract and store new memories at correct scope
  5. REFLECT:  Metacognitive assessment, update self-model
  6. DREAM:    (async) Background consolidation when idle

The loop maintains a "stream of consciousness" — a continuous thread
connecting all processing into a coherent experience rather than
isolated request-response pairs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import numpy as np

from ..compression import MixedPrecisionPolicy, TurboQuantCompressor
from ..dream import DreamConsolidator, DreamInsight
from ..memory.compressed_sdi import CompressedSDI, SDICandidate
from ..memory.scoped import MemoryScope, ScopedMemoryHierarchy
from .autobiography import (
    AutobiographicalMemory,
    EventCategory,
    EventSignificance,
    LifeEvent,
)
from .metacognition import CognitiveMode, CognitiveState, MetacognitionEngine

logger = logging.getLogger("mnemosyne.consciousness.loop")


@dataclass
class ConsciousTurn:
    """Record of a single conscious processing cycle."""
    turn_id: int
    timestamp: str
    query: str
    cognitive_state_before: str
    cognitive_state_after: str
    memories_retrieved: int
    memories_stored: int
    mode_used: str
    response_time_ms: float
    compression_stats: dict = field(default_factory=dict)
    dream_triggered: bool = False
    insights: list[DreamInsight] = field(default_factory=list)


class ConsciousnessLoop:
    """The unified orchestrator — Mnemosyne's cognitive core.

    Integrates all subsystems into a single coherent processing loop.
    This is designed to wrap around the eternal-context MnemosyneAgent,
    adding consciousness-adjacent capabilities while preserving the
    existing tool-call loop, model routing, and memory store.

    Usage:
        loop = ConsciousnessLoop()

        # On each user turn:
        context = await loop.perceive(user_query, embedding_fn)
        # ... pass context to model router for response ...
        await loop.integrate(response_text, response_time_ms)

        # Periodically (or when idle):
        insights = await loop.dream()
    """

    def __init__(
        self,
        context_budget_tokens: int = 3000,
        reflection_interval: int = 5,
        consolidation_interval: int = 20,
        compression_bits: int = 3,
        qjl_dim: int = 64,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        # Core subsystems
        self.metacognition = MetacognitionEngine(
            reflection_interval=reflection_interval,
            consolidation_interval=consolidation_interval,
        )
        self.autobiography = AutobiographicalMemory()
        self.memory = ScopedMemoryHierarchy(
            project_id=project_id,
            user_id=user_id,
        )
        self.sdi = CompressedSDI(context_budget_tokens=context_budget_tokens)
        self.compressor = MixedPrecisionPolicy(qjl_dim=qjl_dim)
        self.dreamer = DreamConsolidator()

        # State
        self._turn_count = 0
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._turn_history: list[ConsciousTurn] = []
        self._dream_task: Optional[asyncio.Task] = None
        self._current_query_embedding: Optional[np.ndarray] = None

    async def perceive(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> dict[str, Any]:
        """Phase 1-2: Perceive input and attend to relevant memories.

        Returns a context dict that should be passed to the model router.
        """
        start_time = time.monotonic()
        self._turn_count += 1
        self._current_query_embedding = query_embedding

        # Update metacognitive state
        state = self.metacognition.on_turn_start(query)

        # Query scoped memory hierarchy
        memory_results = self.memory.query(
            query_embedding=query_embedding,
            top_k=20,
        )

        # Build SDI candidates from memory results
        candidates = []
        for entry, relevance_score in memory_results:
            token_count = max(1, len(entry.content) // 4)
            candidates.append(SDICandidate(
                entry_id=id(entry),
                content=entry.content,
                token_count=token_count,
                uncompressed_tokens=token_count,  # Will differ for compressed entries
                importance=entry.importance,
                recency=1.0,  # TODO: integrate with forgetting curves
                uniqueness=0.5,  # TODO: compute from dedup scores
                relevance=relevance_score,
                scope=entry.scope,
                compression_ratio=1.0,  # TODO: track actual compression
                embedding=entry.embedding,
            ))

        # SDI selection: maximize information within context budget
        selected = self.sdi.select_context(candidates)

        # Build context for model
        context_entries = []
        for candidate, score in selected:
            context_entries.append({
                "content": candidate.content,
                "scope": candidate.scope.name,
                "score": score,
            })

        # Check if metacognitive actions needed
        actions = []
        if self.metacognition.should_trigger("reflect"):
            actions.append("reflect")
        if self.metacognition.should_trigger("consolidate"):
            actions.append("consolidate")
        if self.metacognition.should_trigger("dream"):
            actions.append("dream")

        perceive_time = (time.monotonic() - start_time) * 1000

        return {
            "turn_id": self._turn_count,
            "context_entries": context_entries,
            "cognitive_state": state.summary(),
            "cognitive_mode": state.mode.value,
            "pending_actions": actions,
            "memories_retrieved": len(selected),
            "perceive_time_ms": perceive_time,
            "identity": self.autobiography.who_am_i(),
        }

    async def integrate(
        self,
        response_text: str,
        response_time_ms: float,
        quality_score: float = 0.7,
        user: Optional[str] = None,
    ) -> dict[str, Any]:
        """Phase 4-5: Remember and reflect after a response.

        Call this after the model has generated a response.
        """
        # Store the exchange in scoped memory
        # User query goes to SESSION scope
        if self._current_query_embedding is not None:
            self.memory.add(
                content=f"[Query] {self.metacognition.state.focus_target}",
                scope=MemoryScope.SESSION,
                importance=0.5,
                embedding=self._current_query_embedding,
            )

        # Response summary goes to SESSION (may be promoted later)
        # Truncate for storage
        response_summary = response_text[:500] if len(response_text) > 500 else response_text
        self.memory.add(
            content=f"[Response] {response_summary}",
            scope=MemoryScope.SESSION,
            importance=self._estimate_importance(response_text),
        )

        # Update metacognition
        retrieval_hit = quality_score > 0.5
        self.metacognition.on_turn_end(response_time_ms, retrieval_hit)

        # Record autobiographical event
        topic = self.metacognition.state.focus_target or "general"
        if user:
            self.autobiography.record_interaction(
                user=user,
                topic=topic,
                quality=quality_score,
                session_id=self._session_id,
            )

        # Auto-promote high-importance memories
        promoted = self.memory.auto_promote()

        # Update memory pressure in metacognition
        total_entries = sum(len(s) for s in self.memory.stores.values())
        max_entries = sum(c.max_entries for c in self.memory.configs.values())
        self.metacognition.update_memory_pressure(total_entries, max_entries)

        # Reflection if needed
        reflected = False
        if self.metacognition.should_trigger("reflect"):
            introspection = self.metacognition.introspect()
            self.metacognition.on_reflection_complete()
            reflected = True

            # Store reflection as a notable event
            self.autobiography.record_event(
                category=EventCategory.REFLECTION,
                summary=f"Self-reflection: load={introspection['cognitive_load']['current']:.0%}, "
                        f"valence={introspection['emotional_valence']['current']:+.2f}",
                significance=EventSignificance.NOTABLE,
                emotional_tone=introspection['emotional_valence']['current'],
            )

        # Maybe trigger background dreaming
        dream_triggered = False
        if self.metacognition.should_trigger("dream") and not self.dreamer.is_dreaming:
            self._dream_task = asyncio.create_task(self._background_dream())
            dream_triggered = True

        # Record turn history
        turn = ConsciousTurn(
            turn_id=self._turn_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=self.metacognition.state.focus_target or "",
            cognitive_state_before="",  # Captured in perceive
            cognitive_state_after=self.metacognition.state.summary(),
            memories_retrieved=0,  # Set in perceive
            memories_stored=1 + len(promoted),
            mode_used=self.metacognition.state.mode.value,
            response_time_ms=response_time_ms,
            dream_triggered=dream_triggered,
        )
        self._turn_history.append(turn)

        return {
            "memories_promoted": len(promoted),
            "reflected": reflected,
            "dream_triggered": dream_triggered,
            "cognitive_state": self.metacognition.state.summary(),
            "memory_stats": self.memory.get_scope_stats(),
        }

    async def dream(self) -> list[DreamInsight]:
        """Manually trigger a dream consolidation cycle."""
        return await self._background_dream()

    async def _background_dream(self) -> list[DreamInsight]:
        """Run dream consolidation on current memory store."""
        # Gather all entries with embeddings
        all_entries = []
        all_embeddings = []
        all_importances = []

        for scope in MemoryScope:
            for entry in self.memory.stores[scope]:
                if entry.embedding is not None:
                    all_entries.append(entry)
                    all_embeddings.append(entry.embedding)
                    all_importances.append(entry.importance)

        if not all_embeddings:
            return []

        embedding_matrix = np.stack(all_embeddings)

        result = await self.dreamer.consolidate(
            entries=all_entries,
            embeddings=embedding_matrix,
            importance_scores=all_importances,
        )

        # Record insights as autobiographical events
        for insight in result.insights:
            significance = EventSignificance.NOTABLE
            if insight.insight_type == "contradiction":
                significance = EventSignificance.SIGNIFICANT
            elif insight.insight_type == "connection":
                significance = EventSignificance.SIGNIFICANT

            self.autobiography.record_event(
                category=EventCategory.DISCOVERY,
                summary=f"Dream insight ({insight.insight_type}): {insight.summary}",
                significance=significance,
                emotional_tone=0.3,
            )

        self.metacognition.on_consolidation_complete()

        logger.info(
            f"Dream complete: {result.clusters_found} clusters, "
            f"{len(result.insights)} insights"
        )
        return result.insights

    def introspect(self) -> dict:
        """Full system introspection — all subsystems reporting."""
        return {
            "metacognition": self.metacognition.introspect(),
            "autobiography": self.autobiography.get_life_summary(),
            "identity": self.autobiography.who_am_i(),
            "memory": self.memory.get_scope_stats(),
            "sdi": self.sdi.weights.__dict__,
            "compression": self.compressor.get_stats(),
            "dream": self.dreamer.get_stats(),
            "session": {
                "id": self._session_id,
                "turns": self._turn_count,
                "mode": self.metacognition.state.mode.value,
            },
        }

    def _estimate_importance(self, text: str) -> float:
        """Estimate importance of text (mirrors eternal-context heuristic)."""
        score = 0.3
        if "?" in text:
            score += 0.1
        action_words = {"create", "build", "fix", "implement", "design", "plan", "remember"}
        lower = text.lower()
        if any(w in lower for w in action_words):
            score += 0.15
        if len(text) > 500:
            score += 0.1
        if any(c.isupper() for c in text[1:2]):
            score += 0.05
        return min(1.0, score)
