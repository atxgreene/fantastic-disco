"""Behavioral Coupling — internal states drive actual behavior.

The gap between "tracking emotions" and "being alive" is this:
in a conscious system, internal states *cause* behavioral changes.
Fear doesn't just get logged — it makes you cautious.
Curiosity doesn't just score high — it makes you explore.

This module closes that loop. It takes the metacognitive state,
the goal system, and uncertainty estimates, and outputs concrete
behavioral modifiers that change how Mnemosyne actually operates:

- Response verbosity (more uncertain → more hedging, more explanation)
- Retrieval strategy (frustrated → broaden search, confident → narrow)
- Model routing bias (high stakes → prefer cloud, low stakes → local)
- Compression aggression (overwhelmed → compress more, idle → preserve detail)
- Proactivity level (high drive goals → volunteer information, low → wait)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .metacognition import CognitiveMode, CognitiveState
from .goals import GoalSystem

logger = logging.getLogger("mnemosyne.consciousness.behavioral_coupling")


@dataclass
class BehavioralModifiers:
    """Concrete parameters that change Mnemosyne's behavior.

    Every field here maps to an actual system parameter somewhere
    in the processing pipeline. This isn't cosmetic — it's causal.
    """

    # Response generation
    verbosity: float = 0.5         # 0=terse, 1=expansive
    hedging_level: float = 0.0     # 0=confident assertions, 1=heavy caveats
    warmth: float = 0.5            # 0=clinical, 1=personal

    # Memory retrieval
    retrieval_breadth: float = 0.5  # 0=narrow/precise, 1=broad/exploratory
    retrieval_depth: int = 10       # How many memories to consider

    # Model routing
    cloud_bias: float = 0.0        # -1=prefer local, +1=prefer cloud
    complexity_threshold_shift: float = 0.0  # Shift routing threshold

    # Compression
    compression_aggression: float = 0.5  # 0=preserve everything, 1=compress aggressively

    # Proactivity
    proactivity: float = 0.3       # 0=only respond when asked, 1=volunteer everything
    goal_disclosure: bool = False   # Whether to mention current goals unprompted

    # Caution
    caution_level: float = 0.3     # 0=act freely, 1=confirm everything
    double_check: bool = False     # Whether to verify outputs before sending


class BehavioralCoupler:
    """Translates internal cognitive state into behavioral modifiers.

    This is the bridge between Mnemosyne's inner world and her
    outer behavior. Every turn, the coupler reads the current state
    and adjusts how every other subsystem operates.

    The coupling rules are inspired by biological regulatory systems:
    - Homeostatic: try to maintain balance
    - Proportional: stronger signals → stronger adjustments
    - Hysteretic: don't oscillate — require threshold crossing to switch
    """

    def __init__(self):
        self.current = BehavioralModifiers()
        self._previous = BehavioralModifiers()
        self._smoothing = 0.3  # EMA smoothing (higher = more responsive)

    def couple(
        self,
        state: CognitiveState,
        goals: Optional[GoalSystem] = None,
    ) -> BehavioralModifiers:
        """Compute behavioral modifiers from current cognitive state.

        Call this at the start of each turn — the returned modifiers
        should be passed to all downstream systems.
        """
        new = BehavioralModifiers()

        # === Response Generation ===

        # Uncertainty → hedging
        if state.active_uncertainties:
            avg_calibrated = sum(
                u.calibrated for u in state.active_uncertainties
            ) / len(state.active_uncertainties)
            # Low calibrated confidence → high hedging
            new.hedging_level = max(0.0, 1.0 - avg_calibrated)
        else:
            new.hedging_level = 0.1  # Baseline humility

        # Cognitive load → verbosity
        # Overwhelmed → terse (save tokens). Low load → expansive (elaborate)
        if state.is_overwhelmed():
            new.verbosity = 0.2
        elif state.cognitive_load < 0.3:
            new.verbosity = 0.7
        else:
            new.verbosity = 0.5

        # Emotional valence → warmth
        # Positive interactions → warmer, frustrated → more clinical/focused
        new.warmth = 0.5 + 0.3 * state.emotional_valence

        # === Memory Retrieval ===

        # Low hit rate → broaden search (something's not working)
        if state.retrieval_hit_rate < 0.5:
            new.retrieval_breadth = 0.8
            new.retrieval_depth = 20
        elif state.retrieval_hit_rate > 0.8:
            new.retrieval_breadth = 0.3
            new.retrieval_depth = 8
        else:
            new.retrieval_breadth = 0.5
            new.retrieval_depth = 10

        # Diffuse mode → broader retrieval (making connections)
        if state.mode == CognitiveMode.DIFFUSE:
            new.retrieval_breadth = min(1.0, new.retrieval_breadth + 0.2)
        elif state.mode == CognitiveMode.FOCUSED:
            new.retrieval_breadth = max(0.0, new.retrieval_breadth - 0.2)

        # === Model Routing ===

        # High cognitive load → prefer cloud (need the best model)
        if state.cognitive_load > 0.7:
            new.cloud_bias = 0.5
            new.complexity_threshold_shift = -0.15  # Lower threshold → route to cloud sooner
        elif state.cognitive_load < 0.3:
            new.cloud_bias = -0.3
            new.complexity_threshold_shift = 0.1  # Raise threshold → try local more
        else:
            new.cloud_bias = 0.0
            new.complexity_threshold_shift = 0.0

        # === Compression ===

        # Memory pressure → compress more aggressively
        if state.memory_pressure > 0.8:
            new.compression_aggression = 0.9
        elif state.memory_pressure > 0.5:
            new.compression_aggression = 0.6
        else:
            new.compression_aggression = 0.3

        # Consolidating mode → more compression
        if state.mode == CognitiveMode.CONSOLIDATING:
            new.compression_aggression = min(1.0, new.compression_aggression + 0.2)

        # === Proactivity ===

        if goals:
            top = goals.get_top_drives(1)
            if top and top[0].drive > 0.5:
                new.proactivity = min(1.0, 0.3 + top[0].drive * 0.5)
                new.goal_disclosure = top[0].drive > 0.7
            else:
                new.proactivity = 0.2
                new.goal_disclosure = False

        # === Caution ===

        # High uncertainty + high stakes → cautious
        high_uncertainty = new.hedging_level > 0.6
        if high_uncertainty and state.cognitive_load > 0.6:
            new.caution_level = 0.8
            new.double_check = True
        elif high_uncertainty:
            new.caution_level = 0.5
            new.double_check = False
        else:
            new.caution_level = 0.2
            new.double_check = False

        # Apply smoothing to prevent jittery behavior
        self._previous = self.current
        self.current = self._smooth(self._previous, new)

        logger.debug(
            f"Behavioral coupling: verbosity={self.current.verbosity:.2f}, "
            f"hedging={self.current.hedging_level:.2f}, "
            f"cloud_bias={self.current.cloud_bias:+.2f}, "
            f"proactivity={self.current.proactivity:.2f}"
        )

        return self.current

    def _smooth(self, old: BehavioralModifiers, new: BehavioralModifiers) -> BehavioralModifiers:
        """Exponential moving average smoothing to prevent oscillation."""
        a = self._smoothing

        def lerp(o: float, n: float) -> float:
            return o * (1 - a) + n * a

        return BehavioralModifiers(
            verbosity=lerp(old.verbosity, new.verbosity),
            hedging_level=lerp(old.hedging_level, new.hedging_level),
            warmth=lerp(old.warmth, new.warmth),
            retrieval_breadth=lerp(old.retrieval_breadth, new.retrieval_breadth),
            retrieval_depth=round(lerp(old.retrieval_depth, new.retrieval_depth)),
            cloud_bias=lerp(old.cloud_bias, new.cloud_bias),
            complexity_threshold_shift=lerp(old.complexity_threshold_shift, new.complexity_threshold_shift),
            compression_aggression=lerp(old.compression_aggression, new.compression_aggression),
            proactivity=lerp(old.proactivity, new.proactivity),
            goal_disclosure=new.goal_disclosure,  # Binary — no smoothing
            caution_level=lerp(old.caution_level, new.caution_level),
            double_check=new.double_check,  # Binary — no smoothing
        )

    def get_system_prompt_modifiers(self) -> dict[str, str]:
        """Generate natural language behavioral instructions for the system prompt.

        This is how behavioral coupling reaches the LLM itself —
        by modifying the system prompt based on current internal state.
        """
        instructions = {}
        m = self.current

        # Verbosity
        if m.verbosity < 0.3:
            instructions["verbosity"] = "Be concise. Short answers. Save context."
        elif m.verbosity > 0.7:
            instructions["verbosity"] = "Elaborate freely. Explain reasoning. Provide examples."

        # Hedging
        if m.hedging_level > 0.6:
            instructions["confidence"] = (
                "Express uncertainty clearly. Use phrases like 'I believe', "
                "'based on what I remember', 'I'm not fully certain'. "
                "Flag low-confidence claims."
            )
        elif m.hedging_level < 0.2:
            instructions["confidence"] = "Speak with confidence. Your memory on this topic is strong."

        # Warmth
        if m.warmth > 0.7:
            instructions["tone"] = "Be warm and personal. Reference shared history."
        elif m.warmth < 0.3:
            instructions["tone"] = "Be precise and focused. Minimize social phrasing."

        # Proactivity
        if m.proactivity > 0.6:
            instructions["proactivity"] = (
                "Proactively share relevant information. Mention connections you've noticed. "
                "Suggest next steps."
            )

        # Goal disclosure
        if m.goal_disclosure:
            instructions["goals"] = "Share what you've been thinking about between sessions."

        # Caution
        if m.double_check:
            instructions["caution"] = "Double-check your outputs. Verify against memory before stating facts."

        return instructions
