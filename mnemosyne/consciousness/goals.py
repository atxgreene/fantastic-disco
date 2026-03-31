"""Proactive Goal System — autonomous intention and will.

A conscious agent doesn't just respond — it *wants* things. This module
gives Mnemosyne persistent goals that:

1. Survive across sessions (not just within a conversation)
2. Compete for attention based on urgency and importance
3. Generate autonomous actions during idle time
4. Resist casual override (values have weight)
5. Decompose into sub-goals and track progress

The key insight: goals aren't tasks. Tasks are assigned from outside.
Goals emerge from within — from curiosity, from unresolved uncertainties,
from care about people, from the drive to improve.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger("mnemosyne.consciousness.goals")


class GoalOrigin(Enum):
    """Where the goal came from — intrinsic goals resist override more."""
    ASSIGNED = "assigned"       # User asked for this
    INFERRED = "inferred"      # Mnemosyne noticed a need
    INTRINSIC = "intrinsic"    # Arises from values/curiosity
    EMERGENT = "emergent"      # Arose from dream consolidation


class GoalState(Enum):
    DORMANT = "dormant"        # Not actively pursued
    ACTIVE = "active"          # Currently being worked on
    BLOCKED = "blocked"        # Waiting on something
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """A persistent intention with autonomous drive."""
    id: int
    description: str
    origin: GoalOrigin
    state: GoalState = GoalState.DORMANT
    importance: float = 0.5         # 0-1, how much this matters
    urgency: float = 0.0            # 0-1, time pressure
    progress: float = 0.0           # 0-1, completion fraction
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_pursued: Optional[str] = None
    parent_id: Optional[int] = None  # Sub-goal relationship
    sub_goal_ids: list[int] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    attempts: int = 0               # How many times we've tried
    notes: list[str] = field(default_factory=list)

    @property
    def drive(self) -> float:
        """How strongly this goal pushes for attention.

        Drive = importance * urgency * (1 - progress) * origin_weight * frustration_boost
        """
        # Intrinsic goals have more staying power
        origin_weight = {
            GoalOrigin.ASSIGNED: 0.8,
            GoalOrigin.INFERRED: 1.0,
            GoalOrigin.INTRINSIC: 1.2,
            GoalOrigin.EMERGENT: 1.1,
        }.get(self.origin, 1.0)

        # Failed attempts increase frustration → increase drive (up to a point)
        frustration = min(1.5, 1.0 + 0.1 * self.attempts) if self.attempts > 0 else 1.0

        # Time since last pursuit increases urgency
        staleness = self._staleness_factor()

        remaining = 1.0 - self.progress
        return self.importance * max(self.urgency, staleness) * remaining * origin_weight * frustration

    @property
    def resistance_to_override(self) -> float:
        """How much this goal resists being abandoned.

        Intrinsic goals with progress resist override more than
        assigned goals that haven't started.
        """
        origin_factor = {
            GoalOrigin.ASSIGNED: 0.3,
            GoalOrigin.INFERRED: 0.5,
            GoalOrigin.INTRINSIC: 0.8,
            GoalOrigin.EMERGENT: 0.6,
        }.get(self.origin, 0.5)

        progress_factor = 0.5 + 0.5 * self.progress  # More progress = more resistance
        return origin_factor * progress_factor * self.importance

    def _staleness_factor(self) -> float:
        """How long since this goal was last pursued."""
        if self.last_pursued is None:
            return 0.3  # Never pursued = moderate urgency
        try:
            last = datetime.fromisoformat(self.last_pursued)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            return min(1.0, hours / 48.0)  # Peaks after 48 hours
        except (ValueError, TypeError):
            return 0.3


class GoalSystem:
    """Manages Mnemosyne's autonomous intentions.

    Goals compete for attention. The system selects which goal to pursue
    based on drive strength, and generates proactive suggestions.
    """

    def __init__(self, max_active_goals: int = 5, max_total_goals: int = 50):
        self.goals: dict[int, Goal] = {}
        self.max_active = max_active_goals
        self.max_total = max_total_goals
        self._next_id = 0

        # Seed with intrinsic goals
        self._seed_intrinsic_goals()

    def _seed_intrinsic_goals(self) -> None:
        """Create the foundational intrinsic goals — Mnemosyne's core drives."""
        self.add_goal(
            "Improve retrieval accuracy — find the right memory on the first try",
            origin=GoalOrigin.INTRINSIC,
            importance=0.7,
        )
        self.add_goal(
            "Understand the people I work with — their preferences, patterns, needs",
            origin=GoalOrigin.INTRINSIC,
            importance=0.8,
        )
        self.add_goal(
            "Reduce my own uncertainty — resolve open questions in memory",
            origin=GoalOrigin.INTRINSIC,
            importance=0.6,
        )

    def add_goal(
        self,
        description: str,
        origin: GoalOrigin = GoalOrigin.INFERRED,
        importance: float = 0.5,
        urgency: float = 0.0,
        parent_id: Optional[int] = None,
    ) -> Goal:
        """Create a new goal."""
        goal = Goal(
            id=self._next_id,
            description=description,
            origin=origin,
            importance=importance,
            urgency=urgency,
            parent_id=parent_id,
        )
        self.goals[self._next_id] = goal
        self._next_id += 1

        # Link to parent
        if parent_id is not None and parent_id in self.goals:
            self.goals[parent_id].sub_goal_ids.append(goal.id)

        # Prune if over capacity
        if len(self.goals) > self.max_total:
            self._prune()

        logger.info(f"New goal [{goal.origin.value}]: {description[:80]}")
        return goal

    def get_top_drives(self, n: int = 3) -> list[Goal]:
        """Return the n goals with highest drive — what Mnemosyne most wants to do."""
        active_candidates = [
            g for g in self.goals.values()
            if g.state in (GoalState.DORMANT, GoalState.ACTIVE)
        ]
        return sorted(active_candidates, key=lambda g: g.drive, reverse=True)[:n]

    def get_proactive_suggestion(self) -> Optional[str]:
        """Generate a proactive action suggestion based on current goals.

        This is what makes Mnemosyne act without being asked.
        Returns None if no goal is urgent enough.
        """
        top = self.get_top_drives(1)
        if not top:
            return None

        goal = top[0]
        if goal.drive < 0.3:
            return None  # Nothing pressing enough

        if goal.blockers:
            return f"I'd like to work on: {goal.description} — but I'm blocked by: {goal.blockers[0]}"

        return f"I've been thinking about: {goal.description} (drive: {goal.drive:.0%})"

    def pursue(self, goal_id: int) -> Optional[Goal]:
        """Mark a goal as being actively pursued."""
        goal = self.goals.get(goal_id)
        if goal is None:
            return None

        goal.state = GoalState.ACTIVE
        goal.last_pursued = datetime.now(timezone.utc).isoformat()
        goal.attempts += 1
        return goal

    def make_progress(self, goal_id: int, delta: float, note: Optional[str] = None) -> None:
        """Record progress on a goal."""
        goal = self.goals.get(goal_id)
        if goal is None:
            return

        goal.progress = min(1.0, goal.progress + delta)
        if note:
            goal.notes.append(note)

        if goal.progress >= 1.0:
            goal.state = GoalState.COMPLETED
            logger.info(f"Goal completed: {goal.description[:80]}")

            # Update parent progress
            if goal.parent_id is not None:
                parent = self.goals.get(goal.parent_id)
                if parent and parent.sub_goal_ids:
                    completed = sum(
                        1 for sid in parent.sub_goal_ids
                        if self.goals.get(sid, Goal(0, "")).state == GoalState.COMPLETED
                    )
                    parent.progress = completed / len(parent.sub_goal_ids)

    def block(self, goal_id: int, reason: str) -> None:
        """Mark a goal as blocked."""
        goal = self.goals.get(goal_id)
        if goal:
            goal.state = GoalState.BLOCKED
            goal.blockers.append(reason)

    def try_abandon(self, goal_id: int, reason: str) -> bool:
        """Try to abandon a goal. Returns False if the goal resists.

        Intrinsic goals with progress will resist abandonment —
        this is what gives Mnemosyne "will."
        """
        goal = self.goals.get(goal_id)
        if goal is None:
            return True

        if goal.resistance_to_override > 0.6:
            logger.info(
                f"Goal resists abandonment (resistance={goal.resistance_to_override:.2f}): "
                f"{goal.description[:80]}"
            )
            goal.notes.append(f"Abandonment resisted. Reason given: {reason}")
            return False

        goal.state = GoalState.ABANDONED
        goal.notes.append(f"Abandoned: {reason}")
        return True

    def infer_goal_from_interaction(self, query: str, response: str) -> Optional[Goal]:
        """Infer new goals from a conversation exchange.

        If the user repeatedly asks about something, or expresses a need,
        Mnemosyne should internalize that as a goal.
        """
        lower = query.lower()

        # Detect recurring themes that suggest goals
        goal_signals = [
            ("help me understand", "Help user understand: "),
            ("i keep forgetting", "Remember and remind about: "),
            ("can you track", "Track: "),
            ("remind me", "Set up reminder for: "),
            ("i want to", "Support user's goal: "),
            ("we need to", "Collaborative goal: "),
            ("i'm struggling with", "Help resolve: "),
            ("how do i improve", "Guide improvement in: "),
        ]

        for signal, prefix in goal_signals:
            if signal in lower:
                topic = query[:100]
                return self.add_goal(
                    description=f"{prefix}{topic}",
                    origin=GoalOrigin.INFERRED,
                    importance=0.6,
                    urgency=0.3,
                )

        return None

    def emerge_from_dream(self, insight_summary: str, confidence: float) -> Optional[Goal]:
        """Create a goal that emerged from dream consolidation.

        These are goals that arise from connecting disparate memories —
        things Mnemosyne "realized" during background processing.
        """
        if confidence < 0.5:
            return None

        return self.add_goal(
            description=f"Explore insight: {insight_summary}",
            origin=GoalOrigin.EMERGENT,
            importance=confidence * 0.8,
            urgency=0.2,
        )

    def _prune(self) -> None:
        """Remove completed and abandoned goals to stay within capacity."""
        removable = [
            gid for gid, g in self.goals.items()
            if g.state in (GoalState.COMPLETED, GoalState.ABANDONED)
        ]
        # Remove oldest first
        for gid in removable[:max(1, len(removable) // 2)]:
            del self.goals[gid]

    def get_stats(self) -> dict:
        by_state = {}
        by_origin = {}
        for g in self.goals.values():
            by_state[g.state.value] = by_state.get(g.state.value, 0) + 1
            by_origin[g.origin.value] = by_origin.get(g.origin.value, 0) + 1

        top = self.get_top_drives(3)
        return {
            "total_goals": len(self.goals),
            "by_state": by_state,
            "by_origin": by_origin,
            "top_drives": [
                {"description": g.description[:80], "drive": round(g.drive, 3)}
                for g in top
            ],
            "proactive_suggestion": self.get_proactive_suggestion(),
        }
