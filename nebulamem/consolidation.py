"""Memory consolidation ("dreaming") — extractive, no LLM.

The blueprint's dreaming phase wanted a "lightweight LLM" to rewrite redundant
facts. With no model available we keep the useful half — keeping the graph clean
— using purely set-based algorithms:

  * forgetting: decay non-inhibitory edge weights toward the floor;
  * deduplication: collapse near-duplicate nodes (high token-Jaccard within an
    entity cluster) into a single representative, rewiring their edges;
  * pruning: drop edges that have decayed below the minimum weight.

Abstractive rewriting (true summarization) is intentionally out of scope without
a model; this preserves "记忆图谱极度清爽" without hallucinating new text.
"""
import logging
from typing import Dict, List, Set, Tuple

from .text import tokenize

logger = logging.getLogger("nebulamem.consolidation")


class MemoryConsolidator:
    def __init__(self, mem, dedup_threshold: float = 0.82):
        self.mem = mem
        self.dedup_threshold = dedup_threshold

    def decay(self, idle_seconds: float, decay_rate: float = 0.005) -> None:
        self.mem.graph.decay_weights(idle_seconds, decay_rate)

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
