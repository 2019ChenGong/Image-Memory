"""Progressive Memory Consolidation engine.

Mirrors the cognitive process: episodic memories are periodically
consolidated into semantic knowledge, and semantic knowledge is
further distilled into procedural strategies.

The LLM itself performs the consolidation — acting as the "sleeping brain"
that replays and abstracts experiences.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from nanobot.memory.models import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
)
from nanobot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

CONSOLIDATE_EPISODIC_PROMPT = """\
You are a memory consolidation system. Given a batch of recent interaction episodes,
extract reusable **factual knowledge** (semantic memories) that generalize across episodes.

Episodes:
{episodes}

Existing semantic memories (avoid duplicates, reinforce if overlapping):
{existing_semantic}

Instructions:
- Extract 1-5 factual knowledge statements that generalize from these episodes.
- Each statement should be a standalone fact or insight, NOT a summary of a single episode.
- If an episode merely confirms existing semantic memory, note it for reinforcement.
- Include relevant domain tags.

Respond in JSON (no markdown fences):
{{
  "new_semantic": [
    {{"content": "...", "tags": ["tag1", "tag2"], "source_episode_ids": ["id1", "id2"]}}
  ],
  "reinforce_semantic_ids": ["existing_semantic_id_to_reinforce", ...],
  "reinforcing_episode_ids": ["episode_id_confirming_existing", ...]
}}

If no meaningful knowledge can be extracted, return {{"new_semantic": [], "reinforce_semantic_ids": [], "reinforcing_episode_ids": []}}
"""

CONSOLIDATE_SEMANTIC_PROMPT = """\
You are a strategy extraction system. Given a set of factual knowledge (semantic memories),
extract reusable **action strategies** (procedural memories) in the form "when <trigger>, do <action>".

Semantic memories:
{semantic_memories}

Existing procedural memories (avoid duplicates):
{existing_procedural}

Instructions:
- Extract 1-3 actionable strategies that the agent can apply in future tasks.
- Each strategy must have a clear trigger condition and action.
- Strategies should be general enough to transfer across tasks.
- Only extract if there's enough evidence (multiple semantic memories supporting it).

Respond in JSON (no markdown fences):
{{
  "new_procedural": [
    {{
      "trigger": "When encountering ...",
      "action": "Do ...",
      "tags": ["tag1"],
      "source_semantic_ids": ["id1", "id2"]
    }}
  ],
  "reinforce_procedural_ids": ["existing_procedural_id", ...]
}}

If no strategies can be extracted, return {{"new_procedural": [], "reinforce_procedural_ids": []}}
"""


class LLMCallable(Protocol):
    """Any async callable that takes messages and returns text."""

    async def __call__(self, messages: list[dict[str, Any]]) -> str: ...


class Consolidator:
    """Runs the progressive consolidation pipeline.

    Trigger policy:
    - Episodic → Semantic: every `episode_batch_size` unconsolidated episodes
    - Semantic → Procedural: every `semantic_threshold` semantic memories
    """

    def __init__(
        self,
        store: MemoryStore,
        llm_call: LLMCallable,
        episode_batch_size: int = 5,
        semantic_threshold: int = 6,
    ):
        self.store = store
        self.llm_call = llm_call
        self.episode_batch_size = episode_batch_size
        self.semantic_threshold = semantic_threshold

    async def maybe_consolidate(self) -> dict[str, int]:
        """Check if consolidation is needed, run if so. Returns counts of new memories."""
        result = {"new_semantic": 0, "new_procedural": 0, "reinforced": 0}

        # Phase 1: Episodic → Semantic
        n_uncons = self.store.count_unconsolidated()
        if n_uncons >= self.episode_batch_size:
            r1 = await self._consolidate_episodes()
            result["new_semantic"] = r1.get("new_semantic", 0)
            result["reinforced"] += r1.get("reinforced", 0)

        # Phase 2: Semantic → Procedural
        n_semantic = len(self.store.get_active_semantic())
        if n_semantic >= self.semantic_threshold:
            r2 = await self._consolidate_semantic()
            result["new_procedural"] = r2.get("new_procedural", 0)
            result["reinforced"] += r2.get("reinforced", 0)

        return result

    async def _consolidate_episodes(self) -> dict[str, int]:
        """Consolidate unconsolidated episodes into semantic memories."""
        episodes = self.store.get_unconsolidated_episodes(limit=self.episode_batch_size * 2)
        if not episodes:
            return {}

        existing = self.store.get_all_semantic()

        episodes_text = "\n".join(
            f"- [{ep.id}] (outcome={ep.outcome}, tags={ep.tags}) "
            f"Query: {ep.query[:200]} | Summary: {ep.summary[:300]}"
            for ep in episodes
        )
        existing_text = "\n".join(
            f"- [{m.id}] (confidence={m.confidence:.2f}, reinforced={m.reinforcement_count}x) "
            f"{m.content[:200]}"
            for m in existing
        ) or "(none yet)"

        prompt = CONSOLIDATE_EPISODIC_PROMPT.format(
            episodes=episodes_text, existing_semantic=existing_text
        )

        try:
            raw = await self.llm_call([
                {"role": "system", "content": "You are a memory consolidation engine. Respond only in valid JSON."},
                {"role": "user", "content": prompt},
            ])
            data = _parse_json(raw)
        except Exception as e:
            logger.warning(f"Episodic consolidation failed: {e}")
            return {}

        counts = {"new_semantic": 0, "reinforced": 0}

        # Create new semantic memories
        for item in data.get("new_semantic", []):
            mem = SemanticMemory(
                content=item.get("content", ""),
                source_episode_ids=item.get("source_episode_ids", []),
                tags=item.get("tags", []),
            )
            if mem.content:
                self.store.save_semantic(mem)
                counts["new_semantic"] += 1

        # Reinforce existing
        existing_map = {m.id: m for m in existing}
        for sid in data.get("reinforce_semantic_ids", []):
            if sid in existing_map:
                m = existing_map[sid]
                # find a relevant episode id
                eids = data.get("reinforcing_episode_ids", [])
                m.reinforce(eids[0] if eids else "")
                self.store.save_semantic(m)
                counts["reinforced"] += 1

        # Mark episodes as consolidated
        self.store.mark_consolidated([ep.id for ep in episodes])

        return counts

    async def _consolidate_semantic(self) -> dict[str, int]:
        """Consolidate semantic memories into procedural strategies."""
        semantic = self.store.get_active_semantic()
        if len(semantic) < self.semantic_threshold:
            return {}

        existing_proc = self.store.get_all_procedural()

        semantic_text = "\n".join(
            f"- [{m.id}] (confidence={m.confidence:.2f}, tags={m.tags}) {m.content[:300]}"
            for m in semantic
        )
        proc_text = "\n".join(
            f"- [{p.id}] When: {p.trigger[:150]} → Do: {p.action[:150]}"
            for p in existing_proc
        ) or "(none yet)"

        prompt = CONSOLIDATE_SEMANTIC_PROMPT.format(
            semantic_memories=semantic_text, existing_procedural=proc_text
        )

        try:
            raw = await self.llm_call([
                {"role": "system", "content": "You are a strategy extraction engine. Respond only in valid JSON."},
                {"role": "user", "content": prompt},
            ])
            data = _parse_json(raw)
        except Exception as e:
            logger.warning(f"Semantic consolidation failed: {e}")
            return {}

        counts = {"new_procedural": 0, "reinforced": 0}

        for item in data.get("new_procedural", []):
            mem = ProceduralMemory(
                trigger=item.get("trigger", ""),
                action=item.get("action", ""),
                source_semantic_ids=item.get("source_semantic_ids", []),
                tags=item.get("tags", []),
            )
            if mem.trigger and mem.action:
                self.store.save_procedural(mem)
                counts["new_procedural"] += 1

        # Reinforce
        proc_map = {p.id: p for p in existing_proc}
        for pid in data.get("reinforce_procedural_ids", []):
            if pid in proc_map:
                proc_map[pid].reinforce()
                self.store.save_procedural(proc_map[pid])
                counts["reinforced"] += 1

        return counts


def _parse_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output (handles markdown fences)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)
