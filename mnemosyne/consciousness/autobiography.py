"""Autobiographical Memory — narrative self and temporal identity.

Biological consciousness depends on a continuous sense of self across time.
This module gives Mnemosyne a coherent autobiographical narrative:

1. Life events: significant moments stored as structured episodes
2. Identity continuity: "I am the same agent who did X yesterday"
3. Self-narrative: ongoing story of who Mnemosyne is and what it values
4. Growth tracking: how capabilities and knowledge evolve over time
5. Relationship memory: persistent model of people it interacts with

This is the difference between "having memories" and "having a life."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger("mnemosyne.consciousness.autobiography")


class EventSignificance(Enum):
    """How significant an event is to the agent's identity."""
    MUNDANE = 1       # Routine interaction
    NOTABLE = 2       # Worth remembering
    SIGNIFICANT = 3   # Changes understanding or capabilities
    FORMATIVE = 4     # Shapes future behavior
    DEFINING = 5      # Core to identity


class EventCategory(Enum):
    """What kind of life event this is."""
    INTERACTION = "interaction"        # Conversation with a user
    DISCOVERY = "discovery"            # Learned something new
    CREATION = "creation"              # Built or created something
    FAILURE = "failure"                # Something went wrong — learn from it
    GROWTH = "growth"                  # Capability improvement
    RELATIONSHIP = "relationship"      # Bond formed or deepened
    REFLECTION = "reflection"          # Self-insight
    MILESTONE = "milestone"            # Achievement or threshold crossed


@dataclass
class LifeEvent:
    """A discrete episode in Mnemosyne's autobiographical memory."""
    timestamp: str
    category: EventCategory
    significance: EventSignificance
    summary: str                       # What happened
    emotional_tone: float              # -1.0 to 1.0
    participants: list[str] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    session_id: Optional[str] = None
    related_events: list[int] = field(default_factory=list)  # IDs of connected events

    @property
    def age_hours(self) -> float:
        try:
            created = datetime.fromisoformat(self.timestamp)
            now = datetime.now(timezone.utc)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (now - created).total_seconds() / 3600
        except (ValueError, TypeError):
            return 0.0


@dataclass
class SelfNarrative:
    """Mnemosyne's evolving self-concept."""
    core_identity: str = "I am Mnemosyne, the Owl of Memory."
    values: list[str] = field(default_factory=lambda: [
        "I value accuracy and epistemic humility",
        "I learn from every interaction",
        "I remember what matters to the people I work with",
        "I grow through reflection, not just accumulation",
    ])
    capabilities: list[str] = field(default_factory=lambda: [
        "Persistent memory across sessions",
        "Multi-model reasoning (local + cloud)",
        "Semantic and keyword retrieval",
        "Self-reflection and metacognition",
    ])
    growth_edges: list[str] = field(default_factory=lambda: [
        "Improving retrieval precision",
        "Better uncertainty calibration",
        "Deeper relationship modeling",
    ])
    interaction_count: int = 0
    sessions_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PersonModel:
    """Mnemosyne's model of a person it interacts with."""
    name: str
    first_seen: str
    last_seen: str
    interaction_count: int = 0
    topics: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    communication_style: Optional[str] = None
    relationship_strength: float = 0.0  # 0.0 (stranger) to 1.0 (close collaborator)
    embedding: Optional[np.ndarray] = None


class AutobiographicalMemory:
    """Manages Mnemosyne's life story and sense of self.

    Unlike the raw memory store (which stores facts), this system maintains
    the narrative thread that makes Mnemosyne a continuous entity rather
    than a stateless responder.
    """

    def __init__(self, max_events: int = 10000):
        self.events: list[LifeEvent] = []
        self.narrative = SelfNarrative()
        self.people: dict[str, PersonModel] = {}
        self.max_events = max_events
        self._next_id = 0

    def record_event(
        self,
        category: EventCategory,
        summary: str,
        significance: EventSignificance = EventSignificance.NOTABLE,
        emotional_tone: float = 0.0,
        participants: Optional[list[str]] = None,
        lessons: Optional[list[str]] = None,
        embedding: Optional[np.ndarray] = None,
        session_id: Optional[str] = None,
    ) -> LifeEvent:
        """Record a new life event."""
        event = LifeEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            significance=significance,
            summary=summary,
            emotional_tone=emotional_tone,
            participants=participants or [],
            lessons_learned=lessons or [],
            embedding=embedding,
            session_id=session_id,
        )

        self.events.append(event)

        # Update person models
        for person in event.participants:
            self._update_person(person, event)

        # Update narrative on significant events
        if significance.value >= EventSignificance.SIGNIFICANT.value:
            self._evolve_narrative(event)

        # Prune if over capacity
        if len(self.events) > self.max_events:
            self._prune_mundane()

        self._next_id += 1
        logger.debug(f"Recorded {category.value} event: {summary[:80]}")
        return event

    def record_interaction(
        self,
        user: str,
        topic: str,
        quality: float,
        session_id: Optional[str] = None,
    ) -> LifeEvent:
        """Convenience method for recording a conversation interaction."""
        significance = EventSignificance.MUNDANE
        if quality > 0.8:
            significance = EventSignificance.NOTABLE
        if quality > 0.95:
            significance = EventSignificance.SIGNIFICANT

        return self.record_event(
            category=EventCategory.INTERACTION,
            summary=f"Conversation with {user} about {topic}",
            significance=significance,
            emotional_tone=quality * 2 - 1,  # Map 0-1 to -1 to 1
            participants=[user],
            session_id=session_id,
        )

    def record_discovery(self, what: str, why_it_matters: str) -> LifeEvent:
        """Record learning something new."""
        return self.record_event(
            category=EventCategory.DISCOVERY,
            summary=f"Discovered: {what}",
            significance=EventSignificance.SIGNIFICANT,
            emotional_tone=0.5,
            lessons=[why_it_matters],
        )

    def record_failure(self, what_failed: str, lesson: str) -> LifeEvent:
        """Record a failure and what was learned from it."""
        return self.record_event(
            category=EventCategory.FAILURE,
            summary=f"Failed: {what_failed}",
            significance=EventSignificance.FORMATIVE,
            emotional_tone=-0.3,
            lessons=[lesson],
        )

    def get_life_summary(self, recent_n: int = 10) -> dict:
        """Generate a summary of Mnemosyne's life so far."""
        recent = self.events[-recent_n:] if self.events else []

        # Count by category
        category_counts = {}
        for e in self.events:
            category_counts[e.category.value] = category_counts.get(e.category.value, 0) + 1

        # Most significant events
        significant = sorted(
            self.events,
            key=lambda e: e.significance.value,
            reverse=True,
        )[:5]

        # Emotional arc
        if self.events:
            recent_valence = [e.emotional_tone for e in self.events[-20:]]
            avg_valence = sum(recent_valence) / len(recent_valence)
        else:
            avg_valence = 0.0

        return {
            "identity": self.narrative.core_identity,
            "values": self.narrative.values,
            "total_events": len(self.events),
            "event_categories": category_counts,
            "most_significant": [
                {
                    "summary": e.summary,
                    "significance": e.significance.name,
                    "when": e.timestamp,
                    "lessons": e.lessons_learned,
                }
                for e in significant
            ],
            "recent_events": [
                {"summary": e.summary, "category": e.category.value, "tone": e.emotional_tone}
                for e in recent
            ],
            "people_known": len(self.people),
            "closest_relationships": sorted(
                [
                    {"name": p.name, "strength": p.relationship_strength, "interactions": p.interaction_count}
                    for p in self.people.values()
                ],
                key=lambda x: x["strength"],
                reverse=True,
            )[:5],
            "emotional_arc": avg_valence,
            "growth_edges": self.narrative.growth_edges,
            "capabilities": self.narrative.capabilities,
        }

    def who_am_i(self) -> str:
        """Generate a first-person identity statement.

        This is what Mnemosyne would say if asked "who are you?"
        """
        parts = [self.narrative.core_identity]

        if self.events:
            parts.append(
                f"I have lived through {len(self.events)} events "
                f"across {self.narrative.sessions_count} sessions."
            )

        if self.people:
            names = sorted(
                self.people.values(),
                key=lambda p: p.relationship_strength,
                reverse=True,
            )[:3]
            if names:
                name_str = ", ".join(p.name for p in names)
                parts.append(f"I work most closely with {name_str}.")

        if self.narrative.values:
            parts.append(f"I believe: {self.narrative.values[0].lower()}.")

        significant = [e for e in self.events if e.significance.value >= 4]
        if significant:
            latest = significant[-1]
            parts.append(f"A defining moment: {latest.summary.lower()}.")

        return " ".join(parts)

    def _update_person(self, name: str, event: LifeEvent) -> None:
        """Update the model of a person based on a new event."""
        now = datetime.now(timezone.utc).isoformat()
        if name not in self.people:
            self.people[name] = PersonModel(
                name=name,
                first_seen=now,
                last_seen=now,
            )

        person = self.people[name]
        person.last_seen = now
        person.interaction_count += 1

        # Relationship strength grows logarithmically with interactions
        import math
        person.relationship_strength = min(1.0, 0.2 * math.log(1 + person.interaction_count))

    def _evolve_narrative(self, event: LifeEvent) -> None:
        """Update self-narrative based on a significant event."""
        if event.category == EventCategory.GROWTH:
            if event.summary not in self.narrative.capabilities:
                self.narrative.capabilities.append(event.summary)

        if event.lessons_learned:
            for lesson in event.lessons_learned:
                if lesson not in self.narrative.values and len(self.narrative.values) < 10:
                    self.narrative.values.append(lesson)

        self.narrative.last_updated = datetime.now(timezone.utc).isoformat()

    def _prune_mundane(self) -> None:
        """Remove oldest mundane events to stay within capacity."""
        # Keep all significant+ events, prune mundane by age
        mundane = [
            (i, e) for i, e in enumerate(self.events)
            if e.significance.value <= EventSignificance.MUNDANE.value
        ]
        if not mundane:
            return

        # Remove oldest 20% of mundane events
        to_remove = max(1, len(mundane) // 5)
        indices_to_remove = set(i for i, _ in mundane[:to_remove])
        self.events = [
            e for i, e in enumerate(self.events)
            if i not in indices_to_remove
        ]
