import enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

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
    embedding: List[float] = field(default_factory=list)
    node_type: MemoryNodeType = MemoryNodeType.FACT
    created_at: float = 0.0
    last_activated: float = 0.0

@dataclass
class MemoryEdge:
    source: str
    target: str
    weight: float  # Synaptic strength: 0.0 to 1.0
    edge_type: MemoryEdgeType = MemoryEdgeType.ASSOCIATION
    updated_at: float = 0.0

@dataclass
class ActivatedNode:
    id: str
    content: str
    node_type: MemoryNodeType
    activation_energy: float
    created_at: float
    last_activated: float

@dataclass
class SpreadingActivationConfig:
    steps: int = 3
    decay: float = 0.6
    fire_threshold: float = 0.15
    max_results: int = 10
