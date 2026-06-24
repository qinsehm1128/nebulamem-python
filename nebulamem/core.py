"""NebulaMem core orchestrator (model-free).

Pipeline, all classical/self-developed algorithms, no embedding model, no LLM:

  1. lexical BM25 (+ optional self-developed Random-Indexing rescoring) locates
     1..k seed star-points;
  2. spreading activation diffuses energy across the associative graph, with
     lateral inhibition suppressing superseded facts;
  3. fired nodes are compiled into a token-budgeted Markdown constellation so a
     large recall can never blow up the LLM context window.
"""
import logging
import time
from typing import Dict, List, Optional

from .graph import GraphStore
from .lexical import BM25Index
from .semantic import RandomIndex
from .store import NodeStore
from .text import extract_entities, normalize_phrase
from .types import (ActivatedNode, MemoryEdgeType, MemoryNodeType,
                    SpreadingActivationConfig)

logger = logging.getLogger("nebulamem.core")


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish estimate of LLM tokens (no tokenizer dependency)."""
    words = len(text.split())
    return max(words, len(text) // 4)


class NebulaMem:
    def __init__(self, db_dir: Optional[str] = None, use_semantic: bool = False,
                 semantic_dim: int = 256):
        # db_dir=None => fully in-memory (fast). A path => sqlite content store.
        content_db = f"{db_dir}/content.sqlite" if db_dir else None
        self.graph = GraphStore()
        self.lexical = BM25Index()
        self.store = NodeStore(content_db)
        self.semantic = RandomIndex(dim=semantic_dim) if use_semantic else None
        self._registered = 0

    # ---- ingestion -----------------------------------------------------------
    def register_memory(self, node_id: str, content: str,
                        node_type: MemoryNodeType = MemoryNodeType.FACT,
                        extra_entities: Optional[List[str]] = None,
                        auto_link: bool = True) -> None:
        ents = extract_entities(content)
        if extra_entities:
            ents |= {normalize_phrase(e) for e in extra_entities if e}
        self.graph.add_node(node_id, node_type, ents)
        self.store.put(node_id, content)
        self.lexical.add(node_id, content)
        if self.semantic is not None:
            self.semantic.add(content)
        if auto_link:
            self.graph.auto_associate(node_id)
        self._registered += 1

    def associate(self, source_id: str, target_id: str, weight: float = 0.8) -> None:
        self.graph.add_edge(source_id, target_id, weight, MemoryEdgeType.ASSOCIATION)
        self.graph.add_edge(target_id, source_id, weight * 0.8, MemoryEdgeType.ASSOCIATION)

    def update_state_with_suppression(self, old_node_id: str, new_node_id: str) -> None:
        self.graph.apply_temporal_override(old_node_id, new_node_id)

    # ---- retrieval -----------------------------------------------------------
    def _seeds(self, query: str, cfg: SpreadingActivationConfig) -> Dict[str, float]:
        ranked = self.lexical.search(query, limit=max(cfg.seed_limit * 3, cfg.seed_limit))
        if not ranked:
            return {}
        # blend self-developed semantic score if enabled
        if self.semantic is not None and cfg.semantic_weight > 0:
            qv = self.semantic.vector(query)
            blended = []
            max_bm = max(s for _, s in ranked) or 1.0
            for doc_id, bm in ranked:
                body = self.store.get(doc_id) or ""
                cos = self.semantic.cosine(qv, self.semantic.vector(body))
                score = (1 - cfg.semantic_weight) * (bm / max_bm) + cfg.semantic_weight * cos
                blended.append((doc_id, score))
            ranked = sorted(blended, key=lambda kv: kv[1], reverse=True)
            norm = ranked[0][1] or 1.0
        else:
            norm = ranked[0][1] or 1.0
        seeds: Dict[str, float] = {}
        for doc_id, score in ranked[:cfg.seed_limit]:
            e = score / norm
            if e >= cfg.seed_min_score:
                seeds[doc_id] = e
        return seeds

    def retrieve(self, query: str,
                 config: Optional[SpreadingActivationConfig] = None) -> List[ActivatedNode]:
        cfg = config or SpreadingActivationConfig()
        activations = self._seeds(query, cfg)
        hop_of: Dict[str, int] = {nid: 0 for nid in activations}

        # spreading activation with lateral inhibition
        for step in range(cfg.steps):
            delta: Dict[str, float] = {}
            for node, energy in activations.items():
                if energy <= 0:
                    continue
                for edge in self.graph.edges_from(node):
                    if edge.edge_type == MemoryEdgeType.INHIBITORY:
                        delta[edge.target] = delta.get(edge.target, 0.0) - energy * edge.weight * cfg.inhibition
                    else:
                        spread = energy * edge.weight * cfg.decay
                        delta[edge.target] = delta.get(edge.target, 0.0) + spread
                        if edge.target not in hop_of:
                            hop_of[edge.target] = hop_of.get(node, 0) + 1
            if not delta:
                break
            for nid, d in delta.items():
                v = activations.get(nid, 0.0) + d
                activations[nid] = max(-1.0, min(1.0, v))

        # fire threshold — rank on energy FIRST, then materialize content only
        # for the survivors (true on-demand loading: cost ~ max_results, not corpus).
        candidates = [(nid, e) for nid, e in activations.items() if e >= cfg.fire_threshold]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        candidates = candidates[:cfg.max_results]
        fired: List[ActivatedNode] = []
        for nid, energy in candidates:
            meta = self.graph.node_meta(nid)
            if not meta:
                continue
            content = self.store.get(nid) or ""  # on-demand load
            fired.append(ActivatedNode(
                id=nid, content=content, node_type=meta["type"],
                activation_energy=round(energy, 4), hops=hop_of.get(nid, 99),
                created_at=meta["created_at"], last_activated=meta["last_activated"],
            ))

        # token-budget guard: never overflow the context window
        if cfg.token_budget is not None:
            kept, used = [], 0
            for n in fired:
                t = estimate_tokens(n.content)
                if used + t > cfg.token_budget and kept:
                    break
                kept.append(n)
                used += t
            fired = kept
        return fired

    # ---- compilation ---------------------------------------------------------
    def compile_to_markdown(self, nodes: List[ActivatedNode]) -> str:
        if not nodes:
            return "## Active Memory\n(no relevant long-term context)"
        out = ["## NebulaMem — Active Constellation"]
        buckets = {
            MemoryNodeType.STATE: "STATE",
            MemoryNodeType.FACT: "FACT",
            MemoryNodeType.ENTITY: "ENTITY",
        }
        for ntype, label in buckets.items():
            items = [n for n in nodes if n.node_type == ntype]
            if not items:
                continue
            out.append(f"\n### {label}")
            for n in items:
                out.append(f"- ({n.activation_energy:+.2f}, hop {n.hops}) {n.content}")
        return "\n".join(out)
