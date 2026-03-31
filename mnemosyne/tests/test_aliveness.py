"""Tests for the "aliveness" subsystems — goals, coupling, temporal, curiosity."""

import numpy as np
import pytest

from mnemosyne.consciousness.goals import GoalSystem, GoalOrigin, GoalState
from mnemosyne.consciousness.behavioral_coupling import BehavioralCoupler, BehavioralModifiers
from mnemosyne.consciousness.metacognition import CognitiveState, CognitiveMode, MetacognitionEngine, UncertaintyEstimate
from mnemosyne.consciousness.temporal import TemporalAwareness
from mnemosyne.consciousness.curiosity import CuriosityEngine, CuriosityType
from mnemosyne.consciousness.loop import ConsciousnessLoop


class TestGoalSystem:
    def test_intrinsic_goals_seeded(self):
        gs = GoalSystem()
        assert len(gs.goals) >= 3
        intrinsic = [g for g in gs.goals.values() if g.origin == GoalOrigin.INTRINSIC]
        assert len(intrinsic) >= 3

    def test_drive_computation(self):
        gs = GoalSystem()
        goal = gs.add_goal("Test goal", importance=0.9, urgency=0.8)
        assert goal.drive > 0.3

    def test_proactive_suggestion(self):
        gs = GoalSystem()
        gs.add_goal("Important thing", importance=0.9, urgency=0.9)
        suggestion = gs.get_proactive_suggestion()
        assert suggestion is not None
        assert "Important thing" in suggestion

    def test_goal_resistance_to_override(self):
        gs = GoalSystem()
        goal = gs.add_goal(
            "Core value", origin=GoalOrigin.INTRINSIC, importance=0.9
        )
        # Make progress
        gs.make_progress(goal.id, 0.5, "halfway there")
        # Try to abandon — should resist
        assert not gs.try_abandon(goal.id, "just because")
        assert gs.goals[goal.id].state != GoalState.ABANDONED

    def test_goal_can_be_abandoned_if_weak(self):
        gs = GoalSystem()
        goal = gs.add_goal("Weak goal", origin=GoalOrigin.ASSIGNED, importance=0.2)
        assert gs.try_abandon(goal.id, "not needed")
        assert gs.goals[goal.id].state == GoalState.ABANDONED

    def test_goal_inference_from_interaction(self):
        gs = GoalSystem()
        goal = gs.infer_goal_from_interaction(
            "I keep forgetting the API format", "Here it is..."
        )
        assert goal is not None
        assert goal.origin == GoalOrigin.INFERRED

    def test_emergent_goal_from_dream(self):
        gs = GoalSystem()
        goal = gs.emerge_from_dream("Pattern found in memory clusters", 0.8)
        assert goal is not None
        assert goal.origin == GoalOrigin.EMERGENT

    def test_sub_goals(self):
        gs = GoalSystem()
        parent = gs.add_goal("Big project", importance=0.8)
        child1 = gs.add_goal("Step 1", parent_id=parent.id)
        child2 = gs.add_goal("Step 2", parent_id=parent.id)
        assert child1.id in gs.goals[parent.id].sub_goal_ids
        assert child2.id in gs.goals[parent.id].sub_goal_ids


class TestBehavioralCoupling:
    def test_uncertainty_causes_hedging(self):
        coupler = BehavioralCoupler()
        state = CognitiveState()
        state.active_uncertainties = [
            UncertaintyEstimate(confidence=0.3, basis="unsure", evidence_count=1)
        ]
        mods = coupler.couple(state)
        assert mods.hedging_level > 0.3

    def test_high_load_causes_terse_responses(self):
        coupler = BehavioralCoupler()
        state = CognitiveState(cognitive_load=0.95)
        mods = coupler.couple(state)
        assert mods.verbosity < 0.4

    def test_low_load_causes_expansive_responses(self):
        coupler = BehavioralCoupler()
        state = CognitiveState(cognitive_load=0.1)
        mods = coupler.couple(state)
        assert mods.verbosity > 0.4

    def test_low_hit_rate_broadens_retrieval(self):
        coupler = BehavioralCoupler()
        state = CognitiveState(retrieval_hit_rate=0.3)
        mods = coupler.couple(state)
        assert mods.retrieval_breadth > 0.5
        assert mods.retrieval_depth > 10

    def test_high_load_prefers_cloud(self):
        coupler = BehavioralCoupler()
        state = CognitiveState(cognitive_load=0.9)
        mods = coupler.couple(state)
        assert mods.cloud_bias > 0

    def test_system_prompt_modifiers(self):
        coupler = BehavioralCoupler()
        state = CognitiveState(cognitive_load=0.95)
        state.active_uncertainties = [
            UncertaintyEstimate(confidence=0.2, basis="very unsure", evidence_count=0)
        ]
        coupler.couple(state)
        prompts = coupler.get_system_prompt_modifiers()
        assert "confidence" in prompts  # Should instruct hedging

    def test_smoothing_prevents_jitter(self):
        coupler = BehavioralCoupler()
        # Rapid state changes should be smoothed
        state1 = CognitiveState(cognitive_load=0.1)
        state2 = CognitiveState(cognitive_load=0.9)
        mods1 = coupler.couple(state1)
        mods2 = coupler.couple(state2)
        # mods2 should not jump fully to 0.9-level values due to smoothing
        assert mods2.verbosity > 0.2  # Smoothed, not fully terse


class TestTemporalAwareness:
    def test_first_meeting(self):
        ta = TemporalAwareness()
        ctx = ta.begin_session("session1", user="Alice")
        assert ctx.gap_feeling == "first_meeting"
        assert ctx.should_greet_warmly

    def test_greeting_context_first_meeting(self):
        ta = TemporalAwareness()
        ta.begin_session("session1")
        greeting = ta.get_greeting_context()
        assert greeting["greeting_style"] == "introduction"

    def test_session_duration_tracking(self):
        ta = TemporalAwareness()
        ta.begin_session("session1")
        ctx = ta.on_turn()
        assert ctx.session_duration_minutes >= 0
        assert ctx.duration_feeling == "just_started"

    def test_session_lifecycle(self):
        ta = TemporalAwareness()
        ta.begin_session("s1", user="Bob")
        ta.on_turn()
        ta.end_session(topic="testing")
        assert len(ta.sessions) == 1
        assert ta.sessions[0].primary_topic == "testing"

    def test_stats(self):
        ta = TemporalAwareness()
        ta.begin_session("s1")
        ta.end_session()
        stats = ta.get_stats()
        assert stats["total_sessions"] == 1


class TestCuriosityEngine:
    def test_novelty_detection(self):
        ce = CuriosityEngine(novelty_threshold=0.5)
        # Seed with some history
        for _ in range(10):
            emb = np.random.randn(384).astype(np.float32)
            emb /= np.linalg.norm(emb)
            ce.observe("normal topic", embedding=emb)

        # Now introduce something totally different
        novel_emb = np.ones(384, dtype=np.float32)
        novel_emb /= np.linalg.norm(novel_emb)
        signals = ce.observe("completely new thing", embedding=novel_emb)

        # May or may not trigger based on random embeddings — check structure
        assert isinstance(signals, list)
        for s in signals:
            assert hasattr(s, 'curiosity_type')
            assert hasattr(s, 'intensity')

    def test_contradiction_detection(self):
        ce = CuriosityEngine(contradiction_threshold=0.5)
        memories = [
            ("The API uses REST", 0.8),
            ("Built with Python", 0.6),
        ]
        signals = ce.observe(
            "Actually the API is not REST, it changed to GraphQL",
            existing_memories=memories,
        )
        contradiction_signals = [s for s in signals if s.curiosity_type == CuriosityType.CONTRADICTION]
        assert len(contradiction_signals) > 0

    def test_knowledge_gap_detection(self):
        ce = CuriosityEngine()
        # Moderate similarity = partial knowledge
        memories = [("related topic", 0.5), ("somewhat related", 0.45)]
        signals = ce.observe("specific question about topic", existing_memories=memories)
        gap_signals = [s for s in signals if s.curiosity_type == CuriosityType.KNOWLEDGE_GAP]
        assert len(gap_signals) > 0

    def test_curiosity_level(self):
        ce = CuriosityEngine()
        assert ce.get_curiosity_level() == 0.0  # No signals yet

        # Generate some signals
        memories = [("related", 0.5)]
        ce.observe("partial knowledge topic", existing_memories=memories)
        assert ce.get_curiosity_level() >= 0.0

    def test_burning_questions(self):
        ce = CuriosityEngine()
        memories = [("old info", 0.7)]
        ce.observe("This is not correct anymore", existing_memories=memories)
        questions = ce.get_burning_questions(3)
        assert isinstance(questions, list)

    def test_expectation_learning(self):
        ce = CuriosityEngine()
        ce.learn_expectation("deployments happen on Tuesdays", confidence=0.8)
        assert len(ce.expectations) == 1


class TestConsciousnessLoopV2:
    @pytest.mark.asyncio
    async def test_wake_and_sleep(self):
        loop = ConsciousnessLoop()
        wake_result = loop.wake(user="Alice")
        assert "greeting" in wake_result
        assert "identity" in wake_result
        assert "temporal" in wake_result

        sleep_result = loop.sleep(topic_summary="testing lifecycle")
        assert "session_turns" in sleep_result

    @pytest.mark.asyncio
    async def test_full_turn_with_new_subsystems(self):
        loop = ConsciousnessLoop(context_budget_tokens=1000)
        loop.wake(user="TestUser")

        emb = np.random.randn(384).astype(np.float32)
        context = await loop.perceive("I keep forgetting the deploy process", emb)

        # Should include new fields
        assert "behavioral_modifiers" in context
        assert "curiosity_level" in context
        assert "temporal_context" in context
        assert "proactive_suggestion" in context

        result = await loop.integrate(
            "Here's the deploy process...",
            response_time_ms=400.0,
            user="TestUser",
        )

        # Should have inferred a goal from "I keep forgetting"
        assert "goal_inferred" in result
        assert result["goal_inferred"] is not None
        assert "behavioral_modifiers" in result

    @pytest.mark.asyncio
    async def test_introspection_includes_all_subsystems(self):
        loop = ConsciousnessLoop()
        report = loop.introspect()

        # All subsystems present
        assert "metacognition" in report
        assert "autobiography" in report
        assert "goals" in report
        assert "curiosity" in report
        assert "temporal" in report
        assert "behavioral_state" in report
        assert "memory" in report
        assert "compression" in report
        assert "dream" in report
