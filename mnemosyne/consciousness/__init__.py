"""Consciousness subsystems — the full inner world of Mnemosyne."""

from .metacognition import MetacognitionEngine, CognitiveState, CognitiveMode
from .autobiography import AutobiographicalMemory, LifeEvent, EventCategory
from .goals import GoalSystem, Goal, GoalOrigin
from .behavioral_coupling import BehavioralCoupler, BehavioralModifiers
from .temporal import TemporalAwareness, TemporalContext
from .curiosity import CuriosityEngine, CuriositySignal
from .loop import ConsciousnessLoop

__all__ = [
    "MetacognitionEngine", "CognitiveState", "CognitiveMode",
    "AutobiographicalMemory", "LifeEvent", "EventCategory",
    "GoalSystem", "Goal", "GoalOrigin",
    "BehavioralCoupler", "BehavioralModifiers",
    "TemporalAwareness", "TemporalContext",
    "CuriosityEngine", "CuriositySignal",
    "ConsciousnessLoop",
]
