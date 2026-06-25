"""Memory consolidation ("dreaming") — extractive by default, LLM-pluggable.

The blueprint's dreaming phase wanted a "lightweight LLM" to rewrite redundant
facts. Two modes:

  * default (no model): forgetting (edge decay), deduplication (collapse near-
    duplicate nodes by token-Jaccard within an entity cluster), pruning.
  * optional `llm` callable: abstractive consolidation — rewrite a cluster of
    related/duplicate memories into one clean atomic fact. This is the ideal job
    for a small *local* on-device model (e.g. OpenBMB MiniCPM-1B): it runs
    ASYNC/offline, is not latency-critical, and extraction/summarization is well
    within a 1B model's reach. Per-query retrieval stays model-free.

`llm` is any callable `str -> str` (prompt -> completion); plug llama.cpp /
transformers / an API as you like.
"""
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

from .text import tokenize

logger = logging.getLogger("nebulamem.consolidation")


class MemoryConsolidator:
    def __init__(self, mem, dedup_threshold: float = 0.82,
                 llm: Optional[Callable[[str], str]] = None):
        self.mem = mem
        self.dedup_threshold = dedup_threshold
        self.llm = llm  # optional local model (e.g. MiniCPM-1B) for async rewrite

    def decay(self, idle_seconds: float, decay_rate: float = 0.005) -> None:
        self.mem.graph.decay_weights(idle_seconds, decay_rate)

    def abstract_cluster(self, node_ids: List[str]) -> Optional[str]:
        """Rewrite a set of related memories into one clean atomic fact using the
        pluggable local LLM. Returns None if no `llm` is configured."""
        if self.llm is None:
            return None
        facts = [self.mem.store.get(n) or "" for n in node_ids]
        facts = [f for f in facts if f.strip()]
        if not facts:
            return None
        joined = "\n".join(f"- {f}" for f in facts)
        prompt = (
            "Merge these related memory facts into ONE concise, factual sentence. "
            "Keep only the most up-to-date information; drop duplicates and "
            "contradictions in favor of the newest. Output only the sentence.\n\n"
            f"{joined}\n\nMerged fact:"
        )
        try:
            return self.llm(prompt).strip()
        except Exception as e:  # never let dreaming crash the host
            logger.warning("LLM consolidation failed, keeping extractive: %s", e)
            return None

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def deduplicate(self) -> Dict[str, int]:
        """Collapse near-duplicate nodes. Returns {'merged': n, 'pruned_edges': m}."""
        graph = self.mem.graph
        # candidate pairs only within shared-entity clusters (cheap, no O(N^2))
        seen_pairs: Set[Tuple[str, str]] = set()
        token_cache: Dict[str, Set[str]] = {}
        merged = 0
        retired: Set[str] = set()

        def toks(nid: str) -> Set[str]:
            if nid not in token_cache:
                token_cache[nid] = set(tokenize(self.mem.store.get(nid) or ""))
            return token_cache[nid]

        for ent, members in graph.entity_index.items():
            members = [m for m in members if m not in retired]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    key = (a, b) if a < b else (b, a)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    if a in retired or b in retired:
                        continue
                    if self._jaccard(toks(a), toks(b)) >= self.dedup_threshold:
                        keep, drop = self._pick_survivor(a, b)
                        self._merge(keep, drop)
                        retired.add(drop)
                        merged += 1
        return {"merged": merged, "retired_nodes": len(retired)}

    def _pick_survivor(self, a: str, b: str) -> Tuple[str, str]:
        # keep the more-recently-activated / higher-degree node
        ma, mb = self.mem.graph.node_meta(a), self.mem.graph.node_meta(b)
        da, db = len(self.mem.graph.out.get(a, {})), len(self.mem.graph.out.get(b, {}))
        if (mb["last_activated"], db) > (ma["last_activated"], da):
            return b, a
        return a, b

    def _merge(self, keep: str, drop: str) -> None:
        graph = self.mem.graph
        # rewire drop's outgoing edges onto keep
        for tgt, edge in graph.out.get(drop, {}).items():
            if tgt != keep:
                graph.add_edge(keep, tgt, edge.weight, edge.edge_type)
        # rewire edges pointing at drop
        for src, dsts in graph.out.items():
            if drop in dsts and src != keep:
                e = dsts.pop(drop)
                graph.add_edge(src, keep, e.weight, e.edge_type)
        graph.out.pop(drop, None)
        graph.meta.pop(drop, None)
        for ent in graph.node_entities.get(drop, set()):
            graph.entity_index.get(ent, set()).discard(drop)
        graph.node_entities.pop(drop, None)
        self.mem.lexical.remove(drop)
