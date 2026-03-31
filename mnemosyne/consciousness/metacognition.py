"""Metacognition Engine — self-awareness substrate for Mnemosyne.

Metacognition is "thinking about thinking" — the capacity to monitor and
regulate one's own cognitive processes. This module gives Mnemosyne:

1. A self-model: awareness of own capabilities, limitations, and current state
2. Uncertainty tracking: calibrated confidence in its own outputs
3. Cognitive load monitoring: when to offload, compress, or defer
4. Introspection: ability to examine its own reasoning and memory patterns
5. Epistemic humility: knowing what it doesn't know

This is NOT simulated consciousness. It's functional metacognition —
the same kind of self-monitoring that makes biological cognition adaptive.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger("mnemosyne.consciousness.metacognition")


class CognitiveMode(Enum):
    """Current operational mode of the agent."""
    FOCUSED = "focused"           # Deep engagement with single task
    DIFFUSE = "diffuse"           # Broad exploration, making connections
    REFLECTIVE = "reflective"     # Examining own reasoning
    CONSOLIDATING = "consolidating"  # Organizing and compressing memories
    DREAMING = "dreaming"         # Background processing, insight generation


@dataclass
class UncertaintyEstimate:
    """Calibrated confidence for a specific claim or decision."""
    confidence: float          # 0.0 to 1.0
    basis: str                 # What the confidence is based on
    evidence_count: int = 0    # Number of supporting memories/facts
    contradictions: int = 0    # Number of conflicting signals
    last_validated: Optional[str] = None

    @property
    def calibrated(self) -> float:
        """Apply calibration correction.

        Most systems are overconfident. This applies a mild
        pessimistic correction based on evidence density.
        """
        if self.evidence_count == 0:
            return min(self.confidence, 0.3)  # No evidence = low confidence
        # Contradictions reduce confidence exponentially
        contradiction_penalty = math.exp(-0.5 * self.contradictions)
        # More evidence = confidence converges to stated value
        evidence_factor = 1.0 - math.exp(-0.3 * self.evidence_count)
        return self.confidence * contradiction_penalty * evidence_factor


@dataclass
class CognitiveState:
    """Snapshot of Mnemosyne's current cognitive state.

    This IS the self-model — updated continuously, queryable by
    any subsystem, and periodically stored as autobiographical memory.
    """
    mode: CognitiveMode = CognitiveMode.FOCUSED
    cognitive_load: float = 0.0        # 0.0 (idle) to 1.0 (overwhelmed)
    emotional_valence: float = 0.0     # -1.0 (frustrated) to 1.0 (engaged)
    focus_target: Optional[str] = None # What we're currently working on
    active_uncertainties: list[UncertaintyEstimate] = field(default_factory=list)
    turns_since_reflection: int = 0
    turns_since_consolidation: int = 0
    memory_pressure: float = 0.0       # How full the memory system is
    retrieval_hit_rate: float = 1.0    # Recent retrieval success rate
    last_mode_change: Optional[str] = None
    session_start: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def needs_reflection(self, threshold: int = 5) -> bool:
        return self.turns_since_reflection >= threshold

    def needs_consolidation(self, threshold: int = 20) -> bool:
        return (
            self.turns_since_consolidation >= threshold
            or self.memory_pressure > 0.8
        )

    def is_overwhelmed(self) -> bool:
        return self.cognitive_load > 0.85

    def summary(self) -> str:
        """Human-readable state summary for introspection."""
        return (
            f"Mode: {self.mode.value} | Load: {self.cognitive_load:.0%} | "
            f"Valence: {self.emotional_valence:+.2f} | "
            f"Memory pressure: {self.memory_pressure:.0%} | "
            f"Retrieval hit rate: {self.retrieval_hit_rate:.0%} | "
            f"Uncertainties: {len(self.active_uncertainties)} | "
            f"Focus: {self.focus_target or 'none'}"
        )


class MetacognitionEngine:
    """Monitors and regulates Mnemosyne's cognitive processes.

    Tracks cognitive load, uncertainty, emotional valence, and mode transitions.
    Provides introspection capabilities and adaptive behavior suggestions.
    """

    def __init__(self, reflection_interval: int = 5, consolidation_interval: int = 20):
        self.state = CognitiveState()
        self.reflection_interval = reflection_interval
        self.consolidation_interval = consolidation_interval

        # Rolling history for trend detection
        self._load_history: deque[float] = deque(maxlen=50)
        self._valence_history: deque[float] = deque(maxlen=50)
        self._retrieval_results: deque[bool] = deque(maxlen=100)
        self._response_times: deque[float] = deque(maxlen=50)
        self._mode_transitions: list[tuple[str, CognitiveMode, CognitiveMode]] = []

    def on_turn_start(self, query: str) -> CognitiveState:
        """Called at the start of each conversation turn.

        Updates cognitive load based on query characteristics and
        decides whether to trigger reflection or consolidation.
        """
        # Estimate load from query complexity
        query_load = self._estimate_query_load(query)
        self.state.cognitive_load = min(1.0, 0.3 * self.state.cognitive_load + 0.7 * query_load)
        self._load_history.append(self.state.cognitive_load)

        self.state.turns_since_reflection += 1
        self.state.turns_since_consolidation += 1
        self.state.focus_target = query[:100]

        # Mode transitions based on state
        self._maybe_transition_mode()

        return self.state

    def on_turn_end(self, response_time_ms: float, retrieval_hit: bool = True) -> None:
        """Called after each turn completes."""
        self._response_times.append(response_time_ms)
        self._retrieval_results.append(retrieval_hit)

        # Update retrieval hit rate
        if self._retrieval_results:
            self.state.retrieval_hit_rate = sum(self._retrieval_results) / len(self._retrieval_results)

        # Emotional valence: increases with successful, fast responses
        speed_factor = 1.0 if response_time_ms < 2000 else -0.2
        hit_factor = 0.3 if retrieval_hit else -0.3
        self.state.emotional_valence = max(-1.0, min(1.0,
            0.8 * self.state.emotional_valence + 0.2 * (speed_factor + hit_factor)
        ))
        self._valence_history.append(self.state.emotional_valence)

    def on_reflection_complete(self) -> None:
        self.state.turns_since_reflection = 0

    def on_consolidation_complete(self) -> None:
        self.state.turns_since_consolidation = 0

    def update_memory_pressure(self, current_entries: int, max_entries: int) -> None:
        if max_entries > 0:
            self.state.memory_pressure = current_entries / max_entries

    def register_uncertainty(
        self,
        claim: str,
        confidence: float,
        evidence_count: int = 0,
        contradictions: int = 0,
    ) -> UncertaintyEstimate:
        """Register an uncertainty about a specific claim."""
        ue = UncertaintyEstimate(
            confidence=confidence,
            basis=claim,
            evidence_count=evidence_count,
            contradictions=contradictions,
            last_validated=datetime.now(timezone.utc).isoformat(),
        )
        self.state.active_uncertainties.append(ue)
        # Prune old uncertainties (keep most recent 20)
        if len(self.state.active_uncertainties) > 20:
            self.state.active_uncertainties = self.state.active_uncertainties[-20:]
        return ue

    def introspect(self) -> dict:
        """Full introspection report — Mnemosyne examining its own state.

        This is the key metacognitive capability: the ability to report
        on internal processes, identify patterns, and suggest adjustments.
        """
        report = {
            "current_state": self.state.summary(),
            "mode": self.state.mode.value,
            "cognitive_load": {
                "current": self.state.cognitive_load,
                "trend": self._compute_trend(self._load_history),
                "recommendation": self._load_recommendation(),
            },
            "emotional_valence": {
                "current": self.state.emotional_valence,
                "trend": self._compute_trend(self._valence_history),
            },
            "memory_health": {
                "pressure": self.state.memory_pressure,
                "retrieval_hit_rate": self.state.retrieval_hit_rate,
                "needs_consolidation": self.state.needs_consolidation(self.consolidation_interval),
            },
            "uncertainties": [
                {
                    "claim": u.basis,
                    "raw_confidence": u.confidence,
                    "calibrated_confidence": u.calibrated,
                    "evidence": u.evidence_count,
                    "contradictions": u.contradictions,
                }
                for u in self.state.active_uncertainties[-5:]  # Last 5
            ],
            "mode_transitions": [
                {"time": t, "from": f.value, "to": to.value}
                for t, f, to in self._mode_transitions[-10:]
            ],
            "performance": {
                "avg_response_ms": (
                    sum(self._response_times) / len(self._response_times)
                    if self._response_times else 0
                ),
                "turns_since_reflection": self.state.turns_since_reflection,
            },
        }
        return report

    def should_trigger(self, action: str) -> bool:
        """Should a specific action be triggered given current state?"""
        if action == "reflect":
            return self.state.needs_reflection(self.reflection_interval)
        if action == "consolidate":
            return self.state.needs_consolidation(self.consolidation_interval)
        if action == "compress":
            return self.state.memory_pressure > 0.7
        if action == "dream":
            return (
                self.state.cognitive_load < 0.3
                and self.state.turns_since_consolidation > 10
            )
        return False

    def _estimate_query_load(self, query: str) -> float:
        """Estimate cognitive load from query characteristics."""
        load = 0.0
        # Length contributes to load
        load += min(0.4, len(query) / 2000)
        # Questions increase load
        load += 0.1 * query.count("?")
        # Multiple topics (sentence count)
        sentences = query.count(".") + query.count("?") + query.count("!")
        load += min(0.2, sentences * 0.05)
        # Code blocks are complex
        if "```" in query:
            load += 0.2
        # Meta-questions (asking about self) require introspection
        meta_words = {"yourself", "your memory", "do you remember", "how do you",
                      "what do you know", "are you sure", "your confidence"}
        if any(w in query.lower() for w in meta_words):
            load += 0.15
        return min(1.0, load)

    def _maybe_transition_mode(self) -> None:
        """Transition cognitive mode based on current state."""
        old_mode = self.state.mode
        new_mode = old_mode

        if self.state.needs_consolidation(self.consolidation_interval):
            new_mode = CognitiveMode.CONSOLIDATING
        elif self.state.needs_reflection(self.reflection_interval):
            new_mode = CognitiveMode.REFLECTIVE
        elif self.state.cognitive_load < 0.2 and self.state.turns_since_consolidation > 10:
            new_mode = CognitiveMode.DREAMING
        elif self.state.cognitive_load > 0.7:
            new_mode = CognitiveMode.FOCUSED
        else:
            new_mode = CognitiveMode.DIFFUSE

        if new_mode != old_mode:
            now = datetime.now(timezone.utc).isoformat()
            self._mode_transitions.append((now, old_mode, new_mode))
            self.state.mode = new_mode
            self.state.last_mode_change = now
            logger.info(f"Mode transition: {old_mode.value} → {new_mode.value}")

    def _compute_trend(self, history: deque) -> str:
        """Compute trend direction from rolling history."""
        if len(history) < 3:
            return "insufficient_data"
        recent = list(history)[-5:]
        older = list(history)[-10:-5] if len(history) >= 10 else list(history)[:len(history)//2]
        if not older:
            return "stable"
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        diff = avg_recent - avg_older
        if diff > 0.1:
            return "increasing"
        if diff < -0.1:
            return "decreasing"
        return "stable"

    def _load_recommendation(self) -> str:
        """Recommend action based on cognitive load state."""
        if self.state.is_overwhelmed():
            return "offload: compress old memories, defer non-critical processing"
        if self.state.cognitive_load > 0.6:
            return "monitor: approaching capacity, consider focusing"
        if self.state.cognitive_load < 0.2:
            return "available: capacity for dream consolidation or exploration"
        return "nominal: operating within normal parameters"
