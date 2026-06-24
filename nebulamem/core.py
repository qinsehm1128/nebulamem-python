import logging
from typing import List, Dict, Any, Optional
from .types import (
    MemoryNode, MemoryEdge, MemoryNodeType, MemoryEdgeType, 
    ActivatedNode, SpreadingActivationConfig
)
from .vector import ZvecVectorStore
from .graph import LadybugGraphStore
from .kv import LmdbKVStore

logger = logging.getLogger("nebulamem.core")

class NebulaMem:
    """
    NebulaMem Core Orchestrator.
    Fuses Alibaba Zvec (vector similarity), LadybugDB (property graph relations & lateral inhibition),
    and LMDB (fast session state key-value registry) into a seamless, local-first long-term memory engine.
    """
    def __init__(self, db_dir: str = "./nebulamem_store", dimension: int = 64):
        self.db_dir = db_dir
        self.dimension = dimension
        
        # Initialize stores
        self.vector_store = ZvecVectorStore(db_path=f"{db_dir}/zvec", dimension=dimension)
        self.graph_store = LadybugGraphStore(db_path=f"{db_dir}/ladybug.lbug")
        self.kv_store = LmdbKVStore(db_path=f"{db_dir}/lmdb")
        
        logger.info("NebulaMem initialized successfully.")

    def register_memory(self, node_id: str, content: str, embedding: List[float], node_type: MemoryNodeType = MemoryNodeType.FACT) -> None:
        """
        Saves an atomic memory fact.
        1. Embeds and indexes metadata in Alibaba Zvec.
        2. Structurizes facts in LadybugDB graph Node.
        """
        self.vector_store.insert(node_id, embedding, content, node_type.value)
        self.graph_store.add_node(node_id, content, node_type)
        logger.debug(f"Memory registered: [{node_id}] of type '{node_type.value}'")

    def associate(self, source_id: str, target_id: str, weight: float = 1.0, edge_type: MemoryEdgeType = MemoryEdgeType.ASSOCIATION) -> None:
        """
        Links two memory concepts with a synapses edge in LadybugDB.
        """
        self.graph_store.add_edge(source_id, target_id, weight, edge_type)
        # Bidirectional linking for basic association to map cognitive paths
        if edge_type == MemoryEdgeType.ASSOCIATION:
            self.graph_store.add_edge(target_id, source_id, weight * 0.8, edge_type)
        logger.debug(f"Connected [{source_id}] --[{edge_type.value}(w={weight})]--> [{target_id}]")

    def update_state_with_suppression(self, old_node_id: str, new_node_id: str) -> None:
        """
        Transition state where the new state actively suppresses the old outdated facts
        using lateral inhibition.
        """
        self.graph_store.apply_temporal_override(old_node_id, new_node_id)
        logger.info(f"State updated: Suppressing [{old_node_id}] in favor of [{new_node_id}]")

    def retrieve(self, query_embedding: List[float], config: Optional[SpreadingActivationConfig] = None) -> List[ActivatedNode]:
        """
        Orchestrates the complete hybrid cognitive retrieval:
        1. Query Zvec vector index to find closest semantic seed nodes.
        2. Propagate energy through LadybugDB graph paths.
        3. Apply inhibitory dampeners on contradictory paths.
        4. Return nodes exceeding activation threshold.
        """
        cfg = config or SpreadingActivationConfig()
        
        # 1. Search Zvec for seeds
        seeds = self.vector_store.search(query_embedding, limit=3)
        activations: Dict[str, float] = {}
        for seed in seeds:
            if seed["similarity"] > 0.25: # minimum similarity to trigger seed activation
                activations[seed["id"]] = seed["similarity"]

        # 2. Run Spreading Activation algorithm
        for step in range(cfg.steps):
            next_activations = dict(activations)
            for current_node, energy in activations.items():
                edges = self.graph_store.get_edges_from(current_node)
                for edge in edges:
                    target = edge.target
                    weight = edge.weight
                    
                    if edge.edge_type == MemoryEdgeType.INHIBITORY:
                        # Active node dampens contradictory node energy (suppresses contradiction)
                        suppression = energy * weight * 0.95
                        next_activations[target] = next_activations.get(target, 0.0) - suppression
                    else:
                        # Positive energy propagation
                        spread = energy * weight * cfg.decay
                        next_activations[target] = next_activations.get(target, 0.0) + spread
            
            # Apply biological ceiling bounding (-1.0 to 1.0)
            for node_id in next_activations:
                next_activations[node_id] = max(-1.0, min(1.0, next_activations[node_id]))
                
            activations = next_activations

        # 3. Filter activated nodes exceeding fire threshold
        fired_nodes: List[ActivatedNode] = []
        for node_id, energy in activations.items():
            if energy >= cfg.fire_threshold:
                node = self.graph_store.get_node(node_id)
                if node:
                    fired_nodes.append(
                        ActivatedNode(
                            id=node.id,
                            content=node.content,
                            node_type=node.node_type,
                            activation_energy=round(energy, 4),
                            created_at=node.created_at,
                            last_activated=node.last_activated
                        )
                    )

        # Sort by final activation weight descending
        fired_nodes.sort(key=lambda x: x.activation_energy, reverse=True)
        return fired_nodes[:cfg.max_results]

    def compile_to_markdown(self, activated_nodes: List[ActivatedNode]) -> str:
        """
        Compiles the active thought constellation into a structured prompt block
        aligned with the LLM Wiki specification.
        """
        if not activated_nodes:
            return "## 🌌 Active Memory Space\nNo active long-term context retrieved."

        md = "## 🌌 NebulaMem v2: Active Memory Constellation (Zvec + LadybugDB)\n"
        md += "The following relevant contexts are currently high-lighted in your local cognitive star-field:\n\n"
        
        # Segment by node type
        categories = {
            "state": [n for n in activated_nodes if n.node_type == MemoryNodeType.STATE],
            "fact": [n for n in activated_nodes if n.node_type == MemoryNodeType.FACT],
            "entity": [n for n in activated_nodes if n.node_type == MemoryNodeType.ENTITY]
        }

        for cat, items in categories.items():
            if items:
                md += f"### ✦ {cat.upper()} MEMORIES\n"
                for item in items:
                    md += f"- **[{item.id}]** (Activation: {item.activation_energy})\n"
                    md += f"  - *Context*: {item.content}\n"
                md += "\n"
                
        return md
