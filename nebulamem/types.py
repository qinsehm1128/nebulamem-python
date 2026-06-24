"""Core data structures for NebulaMem (model-free edition).

No embedding model, no LLM. Every type here is plain data describing the
star-field: atomic nodes, weighted synaptic edges, and the configuration of
the spreading-activation retrieval pass.
"""
import enum
from dataclasses import dataclass, field
from typing import List, Optional, Set


class MemoryNodeType(str, enum.Enum):
    FACT = "fact"
    ENTITY = "entity"
    STATE = "state"


class MemoryEdgeType(str, enum.Enum):
    ASSOCIATION = "association"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    INHIBITORY = "inhibitory"


@dataclass
class MemoryNode:
    id: str
    content: str
    node_type: MemoryNodeType = MemoryNodeType.FACT
    # Surface entities mined from `content` (proper nouns / titles). These drive
    # automatic associative wiring and the multi-hop bridge edges. No model used.
    entities: Set[str] = field(default_factory=set)
    created_at: float = 0.0
    last_activated: float = 0.0


@dataclass
class MemoryEdge:
    source: str
    target: str
    weight: float  # synaptic strength 0..1 (or up to 1 for inhibitory magnitude)
    edge_type: MemoryEdgeType = MemoryEdgeType.ASSOCIATION
    updated_at: float = 0.0


@dataclass
class ActivatedNode:
    id: str
    content: str
    node_type: MemoryNodeType
    activation_energy: float
    hops: int  # graph distance from the nearest lexical seed (0 = seed itself)
    created_at: float = 0.0
    last_activated: float = 0.0


@dataclass
class SpreadingActivationConfig:
    # seed selection
    seed_limit: int = 8           # max lexical/semantic seeds
    seed_min_score: float = 0.0   # min normalized seed score to ignite

    # spreading
    steps: int = 3                # N iterations of energy diffusion
    decay: float = 0.6            # lambda: per-hop diffusion damping
    inhibition: float = 0.95      # beta: lateral-inhibition strength
    fire_threshold: float = 0.15  # tau: energy needed to be "fired"

    # output control
    max_results: int = 10         # hard cap on returned nodes
    token_budget: Optional[int] = None  # cap compiled-context size (approx tokens)
    per_cluster_cap: Optional[int] = None  # max nodes kept per concept-cluster (precision)
    energy_gap_ratio: Optional[float] = None  # drop nodes below ratio*top_energy (precision)

    # fusion of lexical vs self-developed semantic (random-indexing) seed scores
    semantic_weight: float = 0.0  # 0 = pure lexical; >0 blends RI cosine

    # dense hybrid channel (requires an embedder on the NebulaMem instance)
    hybrid_weight: float = 0.0    # 0 = no dense; 0.5 = equal BM25/dense fusion
    rerank_pool: int = 50         # candidates re-scored by the dense channel
