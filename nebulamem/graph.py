"""In-process associative graph: synapses, plasticity, decay, inhibition.

Pure Python — no Cypher engine, no external graph DB. Holds only lightweight
node metadata (id, type, entities, timestamps) and weighted edges. Node *content*
lives in the NodeStore and is fetched on demand, so the hot graph stays small
even when the corpus is large.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .types import MemoryEdge, MemoryEdgeType, MemoryNodeType


class GraphStore:
    REINFORCE_STEP = 0.1     # synaptic potentiation per co-activation
    MIN_WEIGHT = 0.1         # prune/decay floor

    def __init__(self):
        self.meta: Dict[str, dict] = {}                       # id -> {type, created_at, last_activated}
        self.out: Dict[str, Dict[str, MemoryEdge]] = defaultdict(dict)  # src -> {dst: edge}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)       # entity -> {node_ids}
        self.node_entities: Dict[str, Set[str]] = {}

    # ---- nodes ---------------------------------------------------------------
    def add_node(self, node_id: str, node_type: MemoryNodeType, entities: Set[str]) -> None:
        now = time.time()
        self.meta[node_id] = {"type": node_type, "created_at": now, "last_activated": now}
        self.node_entities[node_id] = set(entities)
        for e in entities:
            self.entity_index[e].add(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.meta

    def node_meta(self, node_id: str) -> Optional[dict]:
        return self.meta.get(node_id)

    def all_node_ids(self) -> List[str]:
        return list(self.meta.keys())

    # ---- edges ---------------------------------------------------------------
    def add_edge(self, source: str, target: str, weight: float,
                 edge_type: MemoryEdgeType = MemoryEdgeType.ASSOCIATION) -> None:
        if source == target:
            return
        now = time.time()
        existing = self.out[source].get(target)
        if existing and existing.edge_type == edge_type and edge_type != MemoryEdgeType.INHIBITORY:
            existing.weight = min(1.0, existing.weight + self.REINFORCE_STEP)  # plasticity
            existing.updated_at = now
        else:
            self.out[source][target] = MemoryEdge(source, target, weight, edge_type, now)

    def auto_associate(self, node_id: str, max_links: int = 32,
                       max_anchor_df: int = 200) -> int:
        """Wire a new node to existing nodes sharing entity anchors.

        Weight scales with the number of shared anchors. This is the self-
        organizing substrate that produces multi-hop bridges with no model:
        two memories that mention the same proper noun become neighbors.

        Anchors that already link to more than `max_anchor_df` nodes are skipped
        as non-discriminative hubs (document-frequency pruning) — this keeps
        ingestion near-linear and stops generic terms from creating giant,
        meaningless stars at scale.
        """
        my_ents = self.node_entities.get(node_id, set())
        if not my_ents:
            return 0
        shared: Dict[str, int] = defaultdict(int)
        for e in my_ents:
            bucket = self.entity_index.get(e, ())
            if len(bucket) > max_anchor_df:
                continue
            for other in bucket:
                if other != node_id:
                    shared[other] += 1
        links = 0
        for other, overlap in sorted(shared.items(), key=lambda kv: kv[1], reverse=True)[:max_links]:
            w = min(1.0, 0.45 + 0.18 * overlap)
            self.add_edge(node_id, other, w, MemoryEdgeType.ASSOCIATION)
            self.add_edge(other, node_id, w, MemoryEdgeType.ASSOCIATION)
            links += 1
        return links

    def apply_temporal_override(self, old_id: str, new_id: str) -> None:
        """new supersedes old: progression edge + back inhibitory edge."""
        self.add_edge(old_id, new_id, 0.8, MemoryEdgeType.TEMPORAL_SEQUENCE)
        self.add_edge(new_id, old_id, 1.0, MemoryEdgeType.INHIBITORY)

    def edges_from(self, source: str) -> List[MemoryEdge]:
        return list(self.out.get(source, {}).values())

    def decay_weights(self, idle_seconds: float, decay_rate: float = 0.005) -> int:
        """Biological forgetting: non-inhibitory edges decay toward the floor."""
        pruned = 0
        for src, dsts in list(self.out.items()):
            for dst, edge in list(dsts.items()):
                if edge.edge_type == MemoryEdgeType.INHIBITORY:
                    continue
                edge.weight = max(self.MIN_WEIGHT, edge.weight - idle_seconds * decay_rate)
        return pruned
