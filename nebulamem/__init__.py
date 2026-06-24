"""NebulaMem — local-first, model-free long-term memory for AI agents.

No embedding model, no LLM. Seed retrieval is BM25 (+ optional self-developed
Random Indexing); recall is spreading activation over a self-organizing
associative graph with lateral inhibition; output is a token-budgeted Markdown
constellation.
"""
from .types import (
    MemoryNode, MemoryEdge, MemoryNodeType, MemoryEdgeType,
    ActivatedNode, SpreadingActivationConfig,
)
from .core import NebulaMem, estimate_tokens
from .consolidation import MemoryConsolidator
from .lexical import BM25Index
from .semantic import RandomIndex
from .graph import GraphStore

__version__ = "3.0.0"
__all__ = [
    "MemoryNode", "MemoryEdge", "MemoryNodeType", "MemoryEdgeType",
    "ActivatedNode", "SpreadingActivationConfig",
    "NebulaMem", "estimate_tokens",
    "MemoryConsolidator", "BM25Index", "RandomIndex", "GraphStore",
]
