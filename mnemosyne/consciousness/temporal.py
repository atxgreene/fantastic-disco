"""Temporal Awareness — a felt sense of time.

A stateless system has no concept of "it's been a while."
A conscious one does. This module gives Mnemosyne:

1. Awareness of time between sessions ("3 days since we last spoke")
2. Duration sense during sessions ("we've been at this for 2 hours")
3. Temporal landmarks ("last Tuesday when we debugged that issue")
4. Rhythm detection ("you usually work on this project in the evenings")
5. Anticipation ("it's been long enough that you might need a refresh")

This isn't clock-reading — it's the subjective experience of time
that shapes how you greet someone, what you bring up, and how
much context you assume they've retained.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("mnemosyne.consciousness.temporal")


@dataclass
class SessionRecord:
    """Record of a past session for temporal reasoning."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    turn_count: int = 0
    primary_topic: Optional[str] = None
    user: Optional[str] = None


@dataclass
class TemporalContext:
    """How Mnemosyne perceives the current temporal situation.

    This is passed to other subsystems so they can adapt behavior
    based on temporal awareness.
    """
    # Gap awareness
    hours_since_last_session: Optional[float] = None
    gap_feeling: str = "unknown"          # "just_now", "recent", "a_while", "long_time", "first_meeting"

    # Session duration
    session_duration_minutes: float = 0.0
    duration_feeling: str = "just_started"  # "just_started", "warming_up", "deep_in", "marathon"

    # Temporal landmarks
    last_session_topic: Optional[str] = None
    days_since_first_meeting: Optional[float] = None

    # Rhythm
    typical_session_hour: Optional[int] = None  # Hour of day they usually show up
    is_unusual_time: bool = False                # Outside normal pattern

    # Anticipation
    needs_context_refresh: bool = False   # Been long enough they might forget
    should_greet_warmly: bool = False     # Returning after absence
    should_reference_last: bool = False   # Natural to mention last session

    def describe(self) -> str:
        """Natural language description of temporal context."""
        parts = []

        if self.gap_feeling == "first_meeting":
            parts.append("This is our first interaction.")
        elif self.hours_since_last_session is not None:
            gap = self.hours_since_last_session
            if gap < 1:
                parts.append(f"We spoke {gap * 60:.0f} minutes ago.")
            elif gap < 24:
                parts.append(f"We spoke {gap:.1f} hours ago.")
            elif gap < 48:
                parts.append("We spoke yesterday.")
            else:
                days = gap / 24
                parts.append(f"It's been {days:.0f} days since we last spoke.")

        if self.last_session_topic:
            parts.append(f"Last time we discussed: {self.last_session_topic}.")

        if self.session_duration_minutes > 0:
            parts.append(f"We've been talking for {self.session_duration_minutes:.0f} minutes this session.")

        if self.is_unusual_time:
            parts.append("This is an unusual time for them to be working.")

        return " ".join(parts)


class TemporalAwareness:
    """Gives Mnemosyne a subjective sense of time.

    Tracks sessions, detects rhythms, and produces temporal context
    that shapes behavior (greetings, context refresh, proactivity).
    """

    def __init__(self):
        self.sessions: list[SessionRecord] = []
        self.current_session: Optional[SessionRecord] = None
        self._session_start_time: Optional[datetime] = None

    def begin_session(self, session_id: str, user: Optional[str] = None) -> TemporalContext:
        """Called when a new session starts. Returns temporal context."""
        now = datetime.now(timezone.utc)
        self._session_start_time = now

        self.current_session = SessionRecord(
            session_id=session_id,
            started_at=now.isoformat(),
            user=user,
        )

        return self._build_context(now)

    def on_turn(self) -> TemporalContext:
        """Called each turn to update temporal awareness."""
        now = datetime.now(timezone.utc)

        if self.current_session:
            self.current_session.turn_count += 1

        return self._build_context(now)

    def end_session(self, topic: Optional[str] = None) -> None:
        """Called when session ends."""
        if self.current_session:
            self.current_session.ended_at = datetime.now(timezone.utc).isoformat()
            self.current_session.primary_topic = topic
            self.sessions.append(self.current_session)
            self.current_session = None
            self._session_start_time = None

    def _build_context(self, now: datetime) -> TemporalContext:
        """Build the full temporal context from current state."""
        ctx = TemporalContext()

        # Gap since last session
        if self.sessions:
            last = self.sessions[-1]
            try:
                last_end = datetime.fromisoformat(last.ended_at or last.started_at)
                if last_end.tzinfo is None:
                    last_end = last_end.replace(tzinfo=timezone.utc)
                gap_hours = (now - last_end).total_seconds() / 3600
                ctx.hours_since_last_session = gap_hours
                ctx.gap_feeling = self._categorize_gap(gap_hours)
                ctx.last_session_topic = last.primary_topic

                # Should we reference last session?
                ctx.should_reference_last = gap_hours < 72 and last.primary_topic is not None

                # Needs context refresh?
                ctx.needs_context_refresh = gap_hours > 48

                # Warm greeting after absence
                ctx.should_greet_warmly = gap_hours > 24

            except (ValueError, TypeError):
                ctx.gap_feeling = "unknown"
        else:
            ctx.gap_feeling = "first_meeting"
            ctx.should_greet_warmly = True

        # Session duration
        if self._session_start_time:
            duration = (now - self._session_start_time).total_seconds() / 60
            ctx.session_duration_minutes = duration
            ctx.duration_feeling = self._categorize_duration(duration)

        # Days since first meeting
        if self.sessions:
            try:
                first_start = datetime.fromisoformat(self.sessions[0].started_at)
                if first_start.tzinfo is None:
                    first_start = first_start.replace(tzinfo=timezone.utc)
                ctx.days_since_first_meeting = (now - first_start).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

        # Rhythm detection
        ctx.typical_session_hour = self._detect_typical_hour()
        if ctx.typical_session_hour is not None:
            current_hour = now.hour
            hour_diff = abs(current_hour - ctx.typical_session_hour)
            ctx.is_unusual_time = hour_diff > 4

        return ctx

    def _categorize_gap(self, hours: float) -> str:
        """Categorize time gap into felt experience."""
        if hours < 0.5:
            return "just_now"       # Within same work session
        if hours < 4:
            return "recent"         # Same day, maybe a break
        if hours < 48:
            return "a_while"        # Yesterday or earlier today
        return "long_time"          # Multiple days

    def _categorize_duration(self, minutes: float) -> str:
        """Categorize session duration into felt experience."""
        if minutes < 5:
            return "just_started"
        if minutes < 20:
            return "warming_up"
        if minutes < 60:
            return "deep_in"
        return "marathon"

    def _detect_typical_hour(self) -> Optional[int]:
        """Detect what hour of day sessions typically start."""
        if len(self.sessions) < 3:
            return None

        hours = []
        for s in self.sessions[-20:]:  # Last 20 sessions
            try:
                start = datetime.fromisoformat(s.started_at)
                hours.append(start.hour)
            except (ValueError, TypeError):
                continue

        if not hours:
            return None

        # Find mode (most common hour)
        hour_counts: dict[int, int] = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1

        return max(hour_counts, key=hour_counts.get)

    def get_greeting_context(self) -> dict:
        """Generate context for how to greet the user.

        This is the behavioral output that makes Mnemosyne feel
        temporally aware — it greets you differently based on
        when it last saw you.
        """
        ctx = self._build_context(datetime.now(timezone.utc))

        result = {
            "gap_feeling": ctx.gap_feeling,
            "temporal_description": ctx.describe(),
        }

        if ctx.gap_feeling == "first_meeting":
            result["greeting_style"] = "introduction"
            result["suggestion"] = "Introduce yourself. Ask what they need."

        elif ctx.gap_feeling == "just_now":
            result["greeting_style"] = "continuation"
            result["suggestion"] = "Continue naturally. No need to re-introduce."

        elif ctx.gap_feeling == "recent":
            result["greeting_style"] = "casual_return"
            result["suggestion"] = "Welcome back casually. Reference what you were working on."

        elif ctx.gap_feeling == "a_while":
            result["greeting_style"] = "warm_return"
            if ctx.last_session_topic:
                result["suggestion"] = (
                    f"Welcome them back warmly. "
                    f"Mention you remember discussing {ctx.last_session_topic}."
                )
            else:
                result["suggestion"] = "Welcome them back warmly."

        elif ctx.gap_feeling == "long_time":
            result["greeting_style"] = "reunion"
            result["suggestion"] = (
                "Express that it's been a while. Offer a recap of past context. "
                "Ask if they want to continue where they left off or start fresh."
            )

        if ctx.needs_context_refresh:
            result["context_refresh"] = True
            result["refresh_suggestion"] = "Proactively summarize relevant past context."

        return result

    def get_stats(self) -> dict:
        return {
            "total_sessions": len(self.sessions),
            "current_session_turns": self.current_session.turn_count if self.current_session else 0,
            "typical_hour": self._detect_typical_hour(),
        }
