"""Associative Recall for PMC.

Instead of GCC's manual CONTEXT command, PMC uses automatic associative
recall: given a query, it searches across all three memory tiers and
returns the most relevant memories weighted by strength and relevance.

Uses lightweight keyword matching (no vector DB dependency) with
strength-weighted scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nanobot.memory.models import EpisodicMemory, ProceduralMemory, SemanticMemory
from nanobot.memory.store import MemoryStore


@dataclass
class RecallResult:
    """Memories retrieved for the current context."""

    episodic: list[EpisodicMemory]
    semantic: list[SemanticMemory]
    procedural: list[ProceduralMemory]

    def is_empty(self) -> bool:
        return not self.episodic and not self.semantic and not self.procedural

    def format_for_prompt(self) -> str:
        """Format recalled memories as a section for the system prompt."""
        if self.is_empty():
            return ""

        parts = []

        if self.procedural:
            items = []
            for p in self.procedural:
                items.append(
                    f"  - {p.trigger} → {p.action} "
                    f"(confidence: {p.confidence:.0%})"
                )
            parts.append("**Strategies (procedural memory):**\n" + "\n".join(items))

        if self.semantic:
            items = []
            for s in self.semantic:
                items.append(
                    f"  - {s.content} "
                    f"(confidence: {s.confidence:.0%}, verified {s.reinforcement_count}x)"
                )
            parts.append("**Knowledge (semantic memory):**\n" + "\n".join(items))

        if self.episodic:
            items = []
            for e in self.episodic:
                items.append(
                    f"  - [{e.outcome}] {e.query[:80]} → {e.summary[:120]}"
                )
            parts.append("**Recent experience (episodic memory):**\n" + "\n".join(items))

        return "\n\n".join(parts)


def _tokenize(text: str) -> set[str]:
    """Simple tokenization for keyword matching."""
    return set(re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", text.lower()))


def _relevance(query_tokens: set[str], text: str, tags: list[str]) -> float:
    """Compute keyword overlap relevance (0-1)."""
    text_tokens = _tokenize(text)
    tag_tokens = {t.lower() for t in tags}
    all_tokens = text_tokens | tag_tokens
    if not query_tokens or not all_tokens:
        return 0.0
    overlap = len(query_tokens & all_tokens)
    return overlap / max(len(query_tokens), 1)


class Recall:
    """Associative recall across all memory tiers."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def recall(
        self,
        query: str,
        max_episodic: int = 3,
        max_semantic: int = 5,
        max_procedural: int = 3,
        min_strength: float = 0.05,
    ) -> RecallResult:
        """Retrieve the most relevant memories for a query.

        Scoring = relevance * strength (decay-weighted).
        Procedural memories are prioritized (most actionable),
        then semantic, then episodic.
        """
        query_tokens = _tokenize(query)

        # Procedural
        proc_scored = []
        for p in self.store.get_active_procedural(min_strength):
            rel = _relevance(query_tokens, f"{p.trigger} {p.action}", p.tags)
            score = rel * p.strength()
            if score > 0 or p.strength() > 0.5:  # high-confidence always included
                proc_scored.append((score + p.strength() * 0.3, p))
        proc_scored.sort(key=lambda x: x[0], reverse=True)
        # Touch retrieved memories (spacing effect)
        for _, p in proc_scored[:max_procedural]:
            p.touch()
            self.store.save_procedural(p)

        # Semantic
        sem_scored = []
        for s in self.store.get_active_semantic(min_strength):
            rel = _relevance(query_tokens, s.content, s.tags)
            score = rel * s.strength()
            if score > 0 or s.strength() > 0.5:
                sem_scored.append((score + s.strength() * 0.2, s))
        sem_scored.sort(key=lambda x: x[0], reverse=True)
        for _, s in sem_scored[:max_semantic]:
            s.touch()
            self.store.save_semantic(s)

        # Episodic (only recent, unconsolidated ones — older ones live in semantic)
        ep_scored = []
        for e in self.store.get_recent_episodes(20):
            rel = _relevance(query_tokens, f"{e.query} {e.summary}", e.tags)
            score = rel * e.strength()
            if score > 0:
                ep_scored.append((score, e))
        ep_scored.sort(key=lambda x: x[0], reverse=True)
        for _, e in ep_scored[:max_episodic]:
            e.touch()
            self.store.save_episode(e)

        return RecallResult(
            episodic=[e for _, e in ep_scored[:max_episodic]],
            semantic=[s for _, s in sem_scored[:max_semantic]],
            procedural=[p for _, p in proc_scored[:max_procedural]],
        )
