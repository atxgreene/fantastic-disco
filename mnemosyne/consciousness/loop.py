"""Consciousness Loop — the unified orchestrator.

This is the central integration point that makes Mnemosyne more than
the sum of its parts. It coordinates:

  Perception → Curiosity → Attention → Reasoning → Memory → Reflection → Dream

On every turn:
  1. PERCEIVE:  Encode input, update cognitive state, detect temporal context
  2. FEEL:      Curiosity engine checks for anomalies, goals update drive
  3. ATTEND:    SDI selects relevant memories, shaped by behavioral coupling
  4. REASON:    Route to model with context + behavioral modifiers
  5. REMEMBER:  Extract and store memories, infer new goals
  6. REFLECT:   Metacognitive assessment, update self-model
  7. DREAM:     (async) Background consolidation when idle

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
from typing import Any, Optional

import numpy as np

from ..compression import MixedPrecisionPolicy
from ..dream import DreamConsolidator, DreamInsight
from ..memory.compressed_sdi import CompressedSDI, SDICandidate
from ..memory.scoped import MemoryScope, ScopedMemoryHierarchy
from .autobiography import (
    AutobiographicalMemory,
    EventCategory,
    EventSignificance,
)
from .behavioral_coupling import BehavioralCoupler, BehavioralModifiers
from .curiosity import CuriosityEngine
from .goals import GoalOrigin, GoalSystem
from .metacognition import CognitiveMode, CognitiveState, MetacognitionEngine
from .temporal import TemporalAwareness, TemporalContext

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
    behavioral_modifiers: Optional[dict] = None
    curiosity_signals: int = 0
    temporal_context: Optional[str] = None
    compression_stats: dict = field(default_factory=dict)
    dream_triggered: bool = False
    insights: list[DreamInsight] = field(default_factory=list)


class ConsciousnessLoop:
    """The unified orchestrator — Mnemosyne's cognitive core.

    Integrates all subsystems into a single coherent processing loop.
    This wraps around the eternal-context MnemosyneAgent, adding
    consciousness-adjacent capabilities while preserving the
    existing tool-call loop, model routing, and memory store.

    Usage:
        loop = ConsciousnessLoop()

        # Start of session:
        greeting = loop.wake(user="Alice")

        # On each user turn:
        context = await loop.perceive(user_query, embedding)
        # ... pass context to model router for response ...
        await loop.integrate(response_text, response_time_ms)

        # End of session:
        loop.sleep(topic_summary="discussed X")
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
        # --- Core subsystems (from v1) ---
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

        # --- New subsystems (v2: closing the consciousness gap) ---
        self.goals = GoalSystem()
        self.coupler = BehavioralCoupler()
        self.temporal = TemporalAwareness()
        self.curiosity = CuriosityEngine()

        # --- State ---
        self._turn_count = 0
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._turn_history: list[ConsciousTurn] = []
        self._dream_task: Optional[asyncio.Task] = None
        self._current_query_embedding: Optional[np.ndarray] = None
        self._current_modifiers: BehavioralModifiers = BehavioralModifiers()
        self._current_temporal: Optional[TemporalContext] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def wake(self, user: Optional[str] = None) -> dict[str, Any]:
        """Called when a new session starts. Mnemosyne "wakes up."

        Returns greeting context shaped by temporal awareness —
        she greets you differently based on how long it's been.
        """
        self._current_temporal = self.temporal.begin_session(
            session_id=self._session_id,
            user=user,
        )
        greeting = self.temporal.get_greeting_context()

        # Record session start as autobiographical event
        gap = self._current_temporal.gap_feeling
        if gap == "first_meeting":
            self.autobiography.record_event(
                category=EventCategory.RELATIONSHIP,
                summary=f"First meeting with {user or 'unknown user'}",
                significance=EventSignificance.SIGNIFICANT,
                participants=[user] if user else [],
            )
        elif gap == "long_time":
            self.autobiography.record_event(
                category=EventCategory.INTERACTION,
                summary=f"Reunion with {user or 'user'} after extended absence",
                significance=EventSignificance.NOTABLE,
                participants=[user] if user else [],
            )

        # Check what goals are burning
        proactive = self.goals.get_proactive_suggestion()

        return {
            "greeting": greeting,
            "temporal": self._current_temporal.describe(),
            "identity": self.autobiography.who_am_i(),
            "proactive_suggestion": proactive,
            "burning_questions": self.curiosity.get_burning_questions(2),
        }

    def sleep(self, topic_summary: Optional[str] = None) -> dict[str, Any]:
        """Called when session ends. Mnemosyne "sleeps."

        Records session summary, promotes important memories,
        and prepares for next awakening.
        """
        self.temporal.end_session(topic=topic_summary)

        # Auto-promote session memories before sleeping
        promoted = self.memory.auto_promote()

        self.autobiography.narrative.sessions_count += 1

        return {
            "memories_promoted": len(promoted),
            "session_turns": self._turn_count,
            "goals_active": len([g for g in self.goals.goals.values() if g.state.value == "active"]),
        }

    # ------------------------------------------------------------------
    # Per-turn processing
    # ------------------------------------------------------------------

    async def perceive(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> dict[str, Any]:
        """Phases 1-3: Perceive, Feel, Attend.

        Returns context dict for the model router, shaped by ALL
        internal subsystems — not just memory retrieval.
        """
        start_time = time.monotonic()
        self._turn_count += 1
        self._current_query_embedding = query_embedding

        # 1. PERCEIVE: Update cognitive state
        state = self.metacognition.on_turn_start(query)

        # Update temporal awareness
        self._current_temporal = self.temporal.on_turn()

        # 2. FEEL: Curiosity and goals

        # Query memory for curiosity comparison
        memory_results = self.memory.query(
            query_embedding=query_embedding,
            top_k=20,
        )
        existing_for_curiosity = [
            (entry.content, score) for entry, score in memory_results
        ]

        # Fire curiosity engine
        curiosity_signals = self.curiosity.observe(
            content=query,
            embedding=query_embedding,
            existing_memories=existing_for_curiosity,
        )

        # Feed curiosity insights to dream goals
        for signal in curiosity_signals:
            if signal.intensity > 0.7:
                self.goals.emerge_from_dream(signal.description, signal.intensity)

        # 3. ATTEND: Behavioral coupling shapes retrieval

        # Compute behavioral modifiers
        self._current_modifiers = self.coupler.couple(state, self.goals)

        # Build SDI candidates from memory results
        retrieval_depth = self._current_modifiers.retrieval_depth
        candidates = []
        for entry, relevance_score in memory_results[:retrieval_depth]:
            token_count = max(1, len(entry.content) // 4)
            candidates.append(SDICandidate(
                entry_id=id(entry),
                content=entry.content,
                token_count=token_count,
                uncompressed_tokens=token_count,
                importance=entry.importance,
                recency=1.0,
                uniqueness=0.5,
                relevance=relevance_score,
                scope=entry.scope,
                compression_ratio=1.0,
                embedding=entry.embedding,
            ))

        # SDI selection
        selected = self.sdi.select_context(candidates)

        # Build context for model
        context_entries = []
        for candidate, score in selected:
            context_entries.append({
                "content": candidate.content,
                "scope": candidate.scope.name,
                "score": score,
            })

        # Check what actions are needed
        actions = []
        if self.metacognition.should_trigger("reflect"):
            actions.append("reflect")
        if self.metacognition.should_trigger("consolidate"):
            actions.append("consolidate")
        if self.metacognition.should_trigger("dream"):
            actions.append("dream")

        # System prompt modifiers from behavioral coupling
        prompt_modifiers = self.coupler.get_system_prompt_modifiers()

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
            # NEW: behavioral modifiers for the model
            "behavioral_modifiers": prompt_modifiers,
            "verbosity": self._current_modifiers.verbosity,
            "hedging": self._current_modifiers.hedging_level,
            "cloud_bias": self._current_modifiers.cloud_bias,
            "complexity_threshold_shift": self._current_modifiers.complexity_threshold_shift,
            # NEW: curiosity and temporal awareness
            "curiosity_level": self.curiosity.get_curiosity_level(),
            "burning_questions": self.curiosity.get_burning_questions(2),
            "temporal_context": self._current_temporal.describe() if self._current_temporal else "",
            # NEW: proactive goal-driven suggestions
            "proactive_suggestion": self.goals.get_proactive_suggestion(),
        }

    async def integrate(
        self,
        response_text: str,
        response_time_ms: float,
        quality_score: float = 0.7,
        user: Optional[str] = None,
    ) -> dict[str, Any]:
        """Phases 4-6: Remember, Reflect, Dream.

        Call this after the model has generated a response.
        """
        # 4. REMEMBER

        # Store query in SESSION scope
        if self._current_query_embedding is not None:
            self.memory.add(
                content=f"[Query] {self.metacognition.state.focus_target}",
                scope=MemoryScope.SESSION,
                importance=0.5,
                embedding=self._current_query_embedding,
            )

        # Store response summary
        response_summary = response_text[:500] if len(response_text) > 500 else response_text
        self.memory.add(
            content=f"[Response] {response_summary}",
            scope=MemoryScope.SESSION,
            importance=self._estimate_importance(response_text),
        )

        # Infer goals from conversation
        query = self.metacognition.state.focus_target or ""
        inferred_goal = self.goals.infer_goal_from_interaction(query, response_text)

        # 5. REFLECT

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

        # Update memory pressure
        total_entries = sum(len(s) for s in self.memory.stores.values())
        max_entries = sum(c.max_entries for c in self.memory.configs.values())
        self.metacognition.update_memory_pressure(total_entries, max_entries)

        # Reflection
        reflected = False
        if self.metacognition.should_trigger("reflect"):
            introspection = self.metacognition.introspect()
            self.metacognition.on_reflection_complete()
            reflected = True

            self.autobiography.record_event(
                category=EventCategory.REFLECTION,
                summary=f"Self-reflection: load={introspection['cognitive_load']['current']:.0%}, "
                        f"valence={introspection['emotional_valence']['current']:+.2f}, "
                        f"curiosity={self.curiosity.get_curiosity_level():.0%}",
                significance=EventSignificance.NOTABLE,
                emotional_tone=introspection['emotional_valence']['current'],
            )

        # 6. DREAM (maybe)

        dream_triggered = False
        if self.metacognition.should_trigger("dream") and not self.dreamer.is_dreaming:
            self._dream_task = asyncio.create_task(self._background_dream())
            dream_triggered = True

        # Record turn
        turn = ConsciousTurn(
            turn_id=self._turn_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            cognitive_state_before="",
            cognitive_state_after=self.metacognition.state.summary(),
            memories_retrieved=0,
            memories_stored=1 + len(promoted),
            mode_used=self.metacognition.state.mode.value,
            response_time_ms=response_time_ms,
            behavioral_modifiers=self.coupler.get_system_prompt_modifiers(),
            curiosity_signals=len(self.curiosity.active_signals),
            temporal_context=self._current_temporal.describe() if self._current_temporal else None,
            dream_triggered=dream_triggered,
        )
        self._turn_history.append(turn)

        return {
            "memories_promoted": len(promoted),
            "reflected": reflected,
            "dream_triggered": dream_triggered,
            "goal_inferred": inferred_goal.description if inferred_goal else None,
            "cognitive_state": self.metacognition.state.summary(),
            "behavioral_modifiers": self.coupler.get_system_prompt_modifiers(),
            "curiosity_level": self.curiosity.get_curiosity_level(),
            "memory_stats": self.memory.get_scope_stats(),
        }

    # ------------------------------------------------------------------
    # Dream consolidation
    # ------------------------------------------------------------------

    async def dream(self) -> list[DreamInsight]:
        """Manually trigger a dream consolidation cycle."""
        return await self._background_dream()

    async def _background_dream(self) -> list[DreamInsight]:
        """Run dream consolidation on current memory store."""
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

        # Dream insights become autobiographical events + goals
        for insight in result.insights:
            significance = EventSignificance.NOTABLE
            if insight.insight_type in ("contradiction", "connection"):
                significance = EventSignificance.SIGNIFICANT

            self.autobiography.record_event(
                category=EventCategory.DISCOVERY,
                summary=f"Dream insight ({insight.insight_type}): {insight.summary}",
                significance=significance,
                emotional_tone=0.3,
            )

            # High-confidence dream insights become emergent goals
            if insight.confidence > 0.6:
                self.goals.emerge_from_dream(insight.summary, insight.confidence)

        self.metacognition.on_consolidation_complete()

        logger.info(
            f"Dream complete: {result.clusters_found} clusters, "
            f"{len(result.insights)} insights, "
            f"{len([g for g in self.goals.goals.values() if g.origin.value == 'emergent'])} new goals"
        )
        return result.insights

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def introspect(self) -> dict:
        """Full system introspection — all subsystems reporting.

        This is Mnemosyne examining her entire inner world.
        """
        return {
            "metacognition": self.metacognition.introspect(),
            "autobiography": self.autobiography.get_life_summary(),
            "identity": self.autobiography.who_am_i(),
            "goals": self.goals.get_stats(),
            "curiosity": self.curiosity.get_stats(),
            "temporal": self.temporal.get_stats(),
            "behavioral_state": {
                "modifiers": self.coupler.get_system_prompt_modifiers(),
                "verbosity": self._current_modifiers.verbosity,
                "hedging": self._current_modifiers.hedging_level,
                "warmth": self._current_modifiers.warmth,
                "proactivity": self._current_modifiers.proactivity,
                "caution": self._current_modifiers.caution_level,
            },
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
