"""Data models for Progressive Memory Consolidation (PMC).

Three memory types inspired by cognitive science:
- Episodic: raw interaction traces with natural decay
- Semantic: cross-episode factual knowledge
- Procedural: reusable action strategies ("when X, do Y")
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class EpisodicMemory:
    """A single interaction episode — what happened, in what context, with what result."""

    id: str = field(default_factory=_new_id)
    session_key: str = ""
    # The user query / task description
    query: str = ""
    # A concise summary of the agent's actions and outcome
    summary: str = ""
    # Whether the episode was successful / neutral / failed
    outcome: str = "neutral"  # success | neutral | failure
    # Optional domain tags for retrieval (e.g. ["python", "django"])
    tags: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    confidence: float = 0.5  # starts low, single observation

    # Has this episode been absorbed into semantic memory?
    consolidated: bool = False

    def strength(self) -> float:
        """Ebbinghaus-inspired memory strength with spacing effect."""
        hours = max(((_now() - self.last_accessed).total_seconds()) / 3600, 0.01)
        # Spacing effect: more accesses → slower decay
        stability = 1.0 + math.log1p(self.access_count)
        decay = math.exp(-0.3 * hours / stability)
        return self.confidence * decay

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1


@dataclass
class SemanticMemory:
    """A distilled fact or knowledge extracted from multiple episodes."""

    id: str = field(default_factory=_new_id)
    # The knowledge statement
    content: str = ""
    # Which episodes contributed to this knowledge
    source_episode_ids: list[str] = field(default_factory=list)
    # Domain tags
    tags: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    # Starts higher than episodic — requires multiple episodes to form
    confidence: float = 0.6
    # How many times this has been reinforced by new episodes
    reinforcement_count: int = 1

    def strength(self) -> float:
        hours = max(((_now() - self.last_accessed).total_seconds()) / 3600, 0.01)
        # Semantic memories are more stable (slower decay)
        stability = 2.0 + math.log1p(self.access_count + self.reinforcement_count)
        decay = math.exp(-0.1 * hours / stability)
        return self.confidence * decay

    def reinforce(self, episode_id: str) -> None:
        """A new episode confirms this knowledge."""
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        if episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
        self.touch()

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1


@dataclass
class ProceduralMemory:
    """A reusable strategy: 'when <trigger>, do <action>'."""

    id: str = field(default_factory=_new_id)
    # When should this strategy be applied?
    trigger: str = ""
    # What to do
    action: str = ""
    # Which semantic memories contributed
    source_semantic_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    confidence: float = 0.7  # highest initial — requires most evidence
    reinforcement_count: int = 1

    def strength(self) -> float:
        hours = max(((_now() - self.last_accessed).total_seconds()) / 3600, 0.01)
        # Procedural memories are most stable
        stability = 3.0 + math.log1p(self.access_count + self.reinforcement_count)
        decay = math.exp(-0.05 * hours / stability)
        return self.confidence * decay

    def reinforce(self) -> None:
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.03)
        self.touch()

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1
