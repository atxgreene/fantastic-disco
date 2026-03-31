"""Tests for consciousness subsystems."""

import asyncio

import numpy as np
import pytest

from mnemosyne.consciousness.metacognition import (
    CognitiveMode,
    CognitiveState,
    MetacognitionEngine,
    UncertaintyEstimate,
)
from mnemosyne.consciousness.autobiography import (
    AutobiographicalMemory,
    EventCategory,
    EventSignificance,
)
from mnemosyne.consciousness.loop import ConsciousnessLoop


class TestMetacognition:
    def test_cognitive_load_estimation(self):
        engine = MetacognitionEngine()
        # Simple query
        state = engine.on_turn_start("hello")
        assert state.cognitive_load < 0.5

        # Complex query
        state = engine.on_turn_start(
            "Can you analyze this code, compare it with the previous version, "
            "and explain why the performance degraded? Also please review the "
            "architecture and suggest improvements. ```def foo(): pass```"
        )
        assert state.cognitive_load > 0.3

    def test_uncertainty_calibration(self):
        engine = MetacognitionEngine()
        # No evidence = low calibrated confidence
        ue = engine.register_uncertainty("test claim", confidence=0.9, evidence_count=0)
        assert ue.calibrated <= 0.3

        # With evidence
        ue = engine.register_uncertainty("test claim", confidence=0.9, evidence_count=5)
        assert ue.calibrated > 0.3

        # Contradictions reduce confidence
        ue = engine.register_uncertainty("test claim", confidence=0.9, evidence_count=5, contradictions=3)
        assert ue.calibrated < 0.5

    def test_mode_transitions(self):
        engine = MetacognitionEngine(reflection_interval=2, consolidation_interval=5)
        # After enough turns, should transition to reflective
        for _ in range(3):
            engine.on_turn_start("query")
            engine.on_turn_end(1000.0)

        assert engine.state.turns_since_reflection >= 3

    def test_introspection(self):
        engine = MetacognitionEngine()
        engine.on_turn_start("test query")
        engine.on_turn_end(500.0, retrieval_hit=True)
        report = engine.introspect()
        assert "current_state" in report
        assert "cognitive_load" in report
        assert "memory_health" in report

    def test_overwhelmed(self):
        state = CognitiveState(cognitive_load=0.9)
        assert state.is_overwhelmed()
        state.cognitive_load = 0.5
        assert not state.is_overwhelmed()


class TestAutobiography:
    def test_record_and_retrieve(self):
        auto = AutobiographicalMemory()
        event = auto.record_event(
            category=EventCategory.DISCOVERY,
            summary="Learned about TurboQuant compression",
            significance=EventSignificance.SIGNIFICANT,
            lessons=["Compression can preserve inner product accuracy"],
        )
        assert len(auto.events) == 1
        assert event.category == EventCategory.DISCOVERY

    def test_person_model(self):
        auto = AutobiographicalMemory()
        auto.record_interaction(user="Alice", topic="memory systems", quality=0.8)
        auto.record_interaction(user="Alice", topic="compression", quality=0.9)
        assert "Alice" in auto.people
        assert auto.people["Alice"].interaction_count == 2
        assert auto.people["Alice"].relationship_strength > 0

    def test_who_am_i(self):
        auto = AutobiographicalMemory()
        identity = auto.who_am_i()
        assert "Mnemosyne" in identity

        # After interactions, identity evolves
        auto.record_interaction(user="Ben", topic="agent design", quality=0.9)
        identity = auto.who_am_i()
        assert "Ben" in identity

    def test_life_summary(self):
        auto = AutobiographicalMemory()
        for i in range(5):
            auto.record_event(
                category=EventCategory.INTERACTION,
                summary=f"Event {i}",
                significance=EventSignificance.NOTABLE,
            )
        summary = auto.get_life_summary()
        assert summary["total_events"] == 5

    def test_narrative_evolves(self):
        auto = AutobiographicalMemory()
        auto.record_event(
            category=EventCategory.GROWTH,
            summary="Learned TurboQuant compression",
            significance=EventSignificance.SIGNIFICANT,
        )
        assert "Learned TurboQuant compression" in auto.narrative.capabilities

    def test_pruning(self):
        auto = AutobiographicalMemory(max_events=10)
        for i in range(15):
            auto.record_event(
                category=EventCategory.INTERACTION,
                summary=f"Mundane event {i}",
                significance=EventSignificance.MUNDANE,
            )
        assert len(auto.events) <= 13  # Some pruned


class TestConsciousnessLoop:
    @pytest.mark.asyncio
    async def test_perceive_and_integrate(self):
        loop = ConsciousnessLoop(context_budget_tokens=1000)
        loop.wake(user="TestUser")

        query_emb = np.random.randn(384).astype(np.float32)
        context = await loop.perceive("What is TurboQuant?", query_emb)

        assert "turn_id" in context
        assert "cognitive_state" in context
        assert "identity" in context
        assert "behavioral_modifiers" in context
        assert "curiosity_level" in context
        assert context["turn_id"] == 1

        result = await loop.integrate(
            response_text="TurboQuant compresses KV caches to 3 bits.",
            response_time_ms=500.0,
            user="TestUser",
        )

        assert "cognitive_state" in result
        assert "memory_stats" in result
        assert "behavioral_modifiers" in result

    @pytest.mark.asyncio
    async def test_introspection(self):
        loop = ConsciousnessLoop()
        report = loop.introspect()
        assert "metacognition" in report
        assert "autobiography" in report
        assert "goals" in report
        assert "curiosity" in report
        assert "temporal" in report
        assert "behavioral_state" in report
        assert "memory" in report
        assert "compression" in report
        assert "dream" in report

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        loop = ConsciousnessLoop(reflection_interval=3)
        loop.wake(user="User")

        for i in range(4):
            emb = np.random.randn(384).astype(np.float32)
            await loop.perceive(f"Question {i}", emb)
            await loop.integrate(
                f"Answer {i}", 300.0, user="User"
            )

        assert loop._turn_count == 4
        total = sum(len(s) for s in loop.memory.stores.values())
        assert total > 0
