from .types import (
    MemoryNode, MemoryEdge, MemoryNodeType, MemoryEdgeType, 
    ActivatedNode, SpreadingActivationConfig
)
from .core import NebulaMem
from .consolidation import MemoryConsolidator

__version__ = "2.0.0"
__all__ = [
    "MemoryNode",
    "MemoryEdge",
    "MemoryNodeType",
    "MemoryEdgeType",
    "ActivatedNode",
    "SpreadingActivationConfig",
    "NebulaMem",
    "MemoryConsolidator"
]
