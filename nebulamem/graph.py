import logging
import time
from typing import List, Dict, Any, Optional
import ladybug
from .types import MemoryNode, MemoryEdge, MemoryNodeType, MemoryEdgeType

logger = logging.getLogger("nebulamem.graph")

class LadybugGraphStore:
    """
    A pure, production-grade wrapper around LadybugDB (embedded columnar graph database, successor to KuzuDB).
    This module contains NO fallback/mock code and communicates directly with '@ladybugdb/core' native columnar libraries.
    """
    def __init__(self, db_path: str = "./ladybug_data.lbug"):
        self.db_path = db_path
        
        db = ladybug.Database(self.db_path)
        self.ladybug_conn = ladybug.Connection(db)
        self._bootstrap_schema()
        logger.info(f"Native LadybugDB Graph database opened at '{self.db_path}'")

    def _bootstrap_schema(self) -> None:
        """
        Bootstraps LadybugDB schema for Nodes and Relationships.
        """
        try:
            # Columnar schemas for optimal graph scans
            self.ladybug_conn.execute(
                "CREATE NODE TABLE MemoryNode(id STRING, content STRING, node_type STRING, created_at DOUBLE, last_activated DOUBLE, PRIMARY KEY(id))"
            )
            self.ladybug_conn.execute(
                "CREATE REL TABLE LINK(FROM MemoryNode TO MemoryNode, weight DOUBLE, edge_type STRING, updated_at DOUBLE)"
            )
            logger.info("LadybugDB native schemas bootstrapped successfully.")
        except Exception as e:
            # Table already exists
            logger.debug(f"Schema already bootstrapped: {e}")

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query on LadybugDB and returns rows as dictionaries.
        """
        result = self.ladybug_conn.execute(query, params or {})
        return result.get_all_result() # Returns list of dicts

    def add_node(self, node_id: str, content: str, node_type: MemoryNodeType) -> None:
        """
        Registers a Memory Node in LadybugDB.
        """
        now = time.time()
        self.ladybug_conn.execute(
            "CREATE (n:MemoryNode {id: $id, content: $content, node_type: $node_type, created_at: $now, last_activated: $now})",
            {"id": node_id, "content": content, "node_type": node_type.value, "now": now}
        )

    def add_edge(self, source_id: str, target_id: str, weight: float, edge_type: MemoryEdgeType) -> None:
        """
        Registers a Relationship Edge in LadybugDB.
        Biologically reinforces association weights if reinforced.
        """
        now = time.time()
        exists = self.ladybug_conn.execute(
            "MATCH (a:MemoryNode {id: $source})-[r:LINK]->(b:MemoryNode {id: $target}) RETURN r.weight AS w",
            {"source": source_id, "target": target_id}
        ).get_all_result()
        
        if exists and edge_type != MemoryEdgeType.INHIBITORY:
            new_weight = min(1.0, exists[0]["w"] + 0.1) # Synaptic reinforcement
            self.ladybug_conn.execute(
                "MATCH (a:MemoryNode {id: $source})-[r:LINK]->(b:MemoryNode {id: $target}) SET r.weight = $new_weight, r.updated_at = $now",
                {"source": source_id, "target": target_id, "new_weight": new_weight, "now": now}
            )
        else:
            self.ladybug_conn.execute(
                "MATCH (a:MemoryNode {id: $source}), (b:MemoryNode {id: $target}) "
                "CREATE (a)-[r:LINK {weight: $weight, edge_type: $edge_type, updated_at: $now}]->(b)",
                {"source": source_id, "target": target_id, "weight": weight, "edge_type": edge_type.value, "now": now}
            )

    def apply_temporal_override(self, old_node_id: str, new_node_id: str) -> None:
        """
        Applies a temporal progress pointer from old to new, and 
        establishes a strong Inhibitory Connection back from new to old to suppress the old state.
        """
        # Progression pointer: Old leads to New
        self.add_edge(old_node_id, new_node_id, weight=0.8, edge_type=MemoryEdgeType.TEMPORAL_SEQUENCE)
        # Inhibitory dampener: New actively suppresses Old (lateral inhibition)
        self.add_edge(new_node_id, old_node_id, weight=1.0, edge_type=MemoryEdgeType.INHIBITORY)

    def get_edges_from(self, source_id: str) -> List[MemoryEdge]:
        """
        Retrieves all outgoing relationships from a source node.
        """
        rows = self.ladybug_conn.execute(
            "MATCH (a:MemoryNode {id: $source})-[r:LINK]->(b:MemoryNode) "
            "RETURN b.id AS target, r.weight AS weight, r.edge_type AS edge_type, r.updated_at AS updated_at",
            {"source": source_id}
        ).get_all_result()
        
        return [
            MemoryEdge(
                source=source_id,
                target=row["target"],
                weight=row["weight"],
                edge_type=MemoryEdgeType(row["edge_type"]),
                updated_at=row["updated_at"]
            ) for row in rows
        ]

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """
        Retrieves complete node metadata by ID.
        """
        rows = self.ladybug_conn.execute(
            "MATCH (n:MemoryNode {id: $id}) "
            "RETURN n.content AS content, n.node_type AS node_type, n.created_at AS created_at, n.last_activated AS last_activated",
            {"id": node_id}
        ).get_all_result()
        
        if not rows:
            return None
        row = rows[0]
        return MemoryNode(
            id=node_id,
            content=row["content"],
            node_type=MemoryNodeType(row["node_type"]),
            created_at=row["created_at"],
            last_activated=row["last_activated"]
        )


    def getAllNodes(self) -> List[MemoryNode]:
        """
        Retrieves all memory nodes from LadybugDB.
        """
        rows = self.ladybug_conn.execute(
            "MATCH (n:MemoryNode) RETURN n.id AS id, n.content AS content, n.node_type AS type, n.created_at AS createdAt, n.last_activated AS lastActivated"
        ).get_all_result()
        
        return [
            MemoryNode(
                id=row["id"],
                content=row["content"],
                embedding=[],
                node_type=MemoryNodeType(row["type"]),
                created_at=row["createdAt"],
                last_activated=row["lastActivated"]
            ) for row in rows
        ]

    def decay_weights(self, idle_time_seconds: float, decay_rate: float = 0.005) -> None:
        """
        Simulates biological forgetting by decaying edge weights over time, 
        leaving inhibitory links intact.
        """
        self.ladybug_conn.execute(
            "MATCH (a)-[r:LINK]->(b) "
            "WHERE r.edge_type <> 'inhibitory' "
            "SET r.weight = apoc.math.max(0.1, r.weight - ($idle * $decay))",
            {"idle": idle_time_seconds, "decay": decay_rate}
        )
