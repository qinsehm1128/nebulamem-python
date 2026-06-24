import unittest
import numpy as np
import os
import shutil
import time
import sys
from unittest.mock import MagicMock

# =====================================================================
# Sandbox Test-Time Mock Setup (Isolating the Environment)
# =====================================================================

# 1. Mock zvec
mock_zvec = MagicMock()
mock_zvec.DataType.STRING = "string"
mock_zvec.DataType.FLOAT_VECTOR = "float_vector"
mock_zvec.MetricType.COSINE = "cosine"

# Virtual Zvec Collection implementing exact vector search in the test environment
class MockZvecCollection:
    def __init__(self):
        self.vectors = {}

    def insert(self, data):
        self.vectors[data["id"]] = {
            "embedding": data["embedding"],
            "content": data["content"],
            "node_type": data["node_type"]
        }

    def delete(self, node_id):
        if node_id in self.vectors:
            del self.vectors[node_id]

    def search(self, vector, limit=10):
        results = []
        q_vec = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
            
        for k, v in self.vectors.items():
            n_vec = np.array(v["embedding"], dtype=np.float32)
            n_norm = np.linalg.norm(n_vec)
            if n_norm == 0:
                continue
            sim = np.dot(q_vec, n_vec) / (q_norm * n_norm)
            results.append({"id": k, "score": float(sim)})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

z_collection = MockZvecCollection()
mock_zvec.ZVecOpen.return_value = z_collection
sys.modules['zvec'] = mock_zvec

# 2. Mock ladybug
mock_ladybug = MagicMock()

class MockLadybugResult:
    def __init__(self, rows):
        self.rows = rows
    def get_all_result(self):
        return self.rows

class MockLadybugConnection:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def execute(self, query, params=None):
        params = params or {}
        query_upper = query.upper()
        
        # Emulate CREATE (n:MemoryNode)
        if "CREATE (N:MEMORYNODE" in query_upper:
            node_id = params["id"]
            self.nodes[node_id] = {
                "id": node_id,
                "content": params["content"],
                "node_type": params["node_type"],
                "created_at": params["now"],
                "last_activated": params["now"]
            }
            return MockLadybugResult([])
            
        # Emulate MATCH a, b CREATE a-[r:LINK]->b
        elif "MATCH (A:MEMORYNODE" in query_upper and "CREATE (A)-[R:LINK" in query_upper:
            source = params["source"]
            target = params["target"]
            # Synaptic reinforcement or new edge
            existing = None
            for e in self.edges:
                if e["source"] == source and e["target"] == target:
                    existing = e
                    break
            if existing and params["edge_type"] != "inhibitory":
                existing["weight"] = min(1.0, existing["weight"] + 0.1)
                existing["updated_at"] = params["now"]
            else:
                self.edges.append({
                    "source": source,
                    "target": target,
                    "weight": params["weight"],
                    "edge_type": params["edge_type"],
                    "updated_at": params["now"]
                })
            return MockLadybugResult([])
            
        # Emulate SET r.weight = $new_weight (synaptic reinforcement update)
        elif "SET R.WEIGHT = $NEW_WEIGHT" in query_upper:
            source = params["source"]
            target = params["target"]
            new_weight = params["new_weight"]
            for e in self.edges:
                if e["source"] == source and e["target"] == target:
                    e["weight"] = new_weight
                    e["updated_at"] = params["now"]
            return MockLadybugResult([])

        # Emulate MATCH exists checking
        elif "MATCH (A:MEMORYNODE" in query_upper and "RETURN R.WEIGHT" in query_upper:
            source = params["source"]
            target = params["target"]
            for e in self.edges:
                if e["source"] == source and e["target"] == target:
                    return MockLadybugResult([{"w": e["weight"]}])
            return MockLadybugResult([])
            
        # Emulate Temporal Suppression MATCH MATCH CREATE CREATE
        elif "CREATE (OLD)-[R1:LINK" in query_upper and "CREATE (NEW)-[R2:LINK" in query_upper:
            old_id = params["oldId"]
            new_id = params["newId"]
            now = params["now"]
            self.edges.append({"source": old_id, "target": new_id, "weight": 0.8, "edge_type": "temporal_sequence", "updated_at": now})
            self.edges.append({"source": new_id, "target": old_id, "weight": 1.0, "edge_type": "inhibitory", "updated_at": now})
            return MockLadybugResult([])
            
        # Emulate get_edges_from MATCH (a)-[r:LINK]->(b)
        elif "MATCH (A:MEMORYNODE" in query_upper and "RETURN B.ID" in query_upper:
            source = params["source"]
            matching_edges = []
            for e in self.edges:
                if e["source"] == source:
                    matching_edges.append({
                        "target": e["target"],
                        "weight": e["weight"],
                        "edge_type": e["edge_type"],
                        "updated_at": e["updated_at"]
                    })
            return MockLadybugResult(matching_edges)
            
        # Emulate get_node MATCH (n:MemoryNode {id: $id})
        elif "MATCH (N:MEMORYNODE {ID: $ID})" in query_upper:
            node_id = params["id"]
            node = self.nodes.get(node_id)
            if not node:
                return MockLadybugResult([])
            return MockLadybugResult([{
                "content": node["content"],
                "node_type": node["node_type"],
                "created_at": node["created_at"],
                "last_activated": node["last_activated"]
            }])
            
        # Emulate getAllNodes MATCH (n:MemoryNode)
        elif "MATCH (N:MEMORYNODE)" in query_upper:
            all_nodes = []
            for k, node in self.nodes.items():
                all_nodes.append({
                    "id": node["id"],
                    "content": node["content"],
                    "type": node["node_type"],
                    "createdAt": node["created_at"],
                    "lastActivated": node["last_activated"]
                })
            return MockLadybugResult(all_nodes)
            
        # Emulate weight decay SET r.weight = apoc.math.max
        elif "SET R.WEIGHT = APOC.MATH.MAX" in query_upper:
            idle = params["idle"]
            decay = params["decay"]
            for e in self.edges:
                if e["edge_type"] != "inhibitory":
                    e["weight"] = max(0.1, e["weight"] - (idle * decay))
            return MockLadybugResult([])
            
        return MockLadybugResult([])

ladybug_conn = MockLadybugConnection()
mock_ladybug.Database.return_value = None
mock_ladybug.Connection.return_value = ladybug_conn
sys.modules['ladybug'] = mock_ladybug

# 3. Mock lmdb
mock_lmdb = MagicMock()

class MockLmdbTxn:
    def __init__(self, db):
        self.db = db
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def put(self, key, value):
        self.db[key] = value
    def get(self, key):
        return self.db.get(key, None)
    def delete(self, key):
        if key in self.db:
            del self.db[key]

class MockLmdbEnv:
    def __init__(self):
        self.db = {}
    def begin(self, write=False):
        return MockLmdbTxn(self.db)

lmdb_env = MockLmdbEnv()
mock_lmdb.open.return_value = lmdb_env
sys.modules['lmdb'] = mock_lmdb


# =====================================================================
# Now import production classes (they are completely clean and native!)
# =====================================================================
from nebulamem import (
    NebulaMem, MemoryNodeType, MemoryEdgeType, SpreadingActivationConfig, MemoryConsolidator
)

class TestNebulaMem(unittest.TestCase):
    def setUp(self):
        # Create a fresh store directory for each test
        self.store_dir = "./test_nebulamem_store"
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        os.makedirs(self.store_dir)
        self.mem = NebulaMem(db_dir=self.store_dir, dimension=8)

    def tearDown(self):
        # Clean up
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_registration_and_simple_retrieval(self):
        # Simple vectors
        v_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.mem.register_memory("node_a", "Alpha concept", v_a, MemoryNodeType.FACT)
        self.mem.register_memory("node_b", "Beta concept", v_b, MemoryNodeType.FACT)
        
        # Link them
        self.mem.associate("node_a", "node_b", weight=0.9, edge_type=MemoryEdgeType.ASSOCIATION)
        
        # Search close to A
        query = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        activated = self.mem.retrieve(query, SpreadingActivationConfig(steps=2, decay=0.6, fire_threshold=0.15))
        
        # Both node_a (vector closest) and node_b (via graph propagation) should be active
        active_ids = [n.id for n in activated]
        self.assertIn("node_a", active_ids)
        self.assertIn("node_b", active_ids)

    def test_lateral_inhibition_suppression(self):
        # We model the user's contradiction case (Python vs Rust)
        # Seed embeddings
        v_user = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v_py = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v_rust = [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0] # semantically close to py
        
        self.mem.register_memory("user", "AI Developer in Yiwu", v_user, MemoryNodeType.ENTITY)
        self.mem.register_memory("py_state", "Using Python legacy database sync", v_py, MemoryNodeType.STATE)
        self.mem.register_memory("rust_state", "Migrated to Rust LadybugDB and Zvec", v_rust, MemoryNodeType.STATE)
        
        # Associations
        self.mem.associate("user", "rust_state", weight=0.9)
        self.mem.associate("user", "py_state", weight=0.7)
        
        # Apply temporal override (Rust actively inhibits Python)
        self.mem.update_state_with_suppression("py_state", "rust_state")
        
        # Query near rust_state (What's our technical database sync state?)
        query = [0.0, 0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]
        activated = self.mem.retrieve(query, SpreadingActivationConfig(steps=3, decay=0.6, fire_threshold=0.15))
        
        active_ids = [n.id for n in activated]
        
        # rust_state should fire
        self.assertIn("rust_state", active_ids)
        # py_state should be actively suppressed and NOT fire!
        self.assertNotIn("py_state", active_ids)

    def test_synaptic_reinforcement(self):
        v_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mem.register_memory("a", "Concept A", v_a)
        self.mem.register_memory("b", "Concept B", v_b)
        
        # Link once
        self.mem.associate("a", "b", weight=0.5)
        edges = self.mem.graph_store.get_edges_from("a")
        weight_1 = [e.weight for e in edges if e.target == "b"][0]
        self.assertEqual(weight_1, 0.5)
        
        # Link again (synapses reinforcement should trigger +0.1)
        self.mem.associate("a", "b", weight=0.5)
        edges = self.mem.graph_store.get_edges_from("a")
        weight_2 = [e.weight for e in edges if e.target == "b"][0]
        self.assertEqual(weight_2, 0.6)

    def test_edge_weight_decay_and_abstraction_pipelines(self):
        # Setup memories
        v_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mem.register_memory("user", "Yiwu Sourcing Agent", v_a, MemoryNodeType.ENTITY)
        self.mem.register_memory("a", "Dropshipping logistics API Shopify", v_b, MemoryNodeType.FACT)
        
        self.mem.associate("user", "a", weight=0.8)
        
        # Instantiate Consolidator
        consolidator = MemoryConsolidator(
            self.mem, 
            decay_interval_seconds=1.0, 
            decay_rate=0.01,
            consolidation_interval_seconds=5.0
        )
        
        # 1. Test Pipeline 1: Forgetting Weight Decay
        consolidator.trigger_manual_decay(elapsed_seconds=10.0)
        edges = self.mem.graph_store.get_edges_from("user")
        weight = [e.weight for e in edges if e.target == "a"][0]
        # Should decay from 0.8 to 0.8 - (10 * 0.01) = 0.7
        self.assertAlmostEqual(weight, 0.7, places=4)
        
        # 2. Test Pipeline 2: Knowledge Distillation & Consolidation
        summary = consolidator.trigger_manual_consolidation()
        self.assertIsNotNone(summary)
        self.assertTrue("CONSOLIDATED NEOCORTICAL SUMMARY" in summary)
        
        # Verify the new summary node was registered in graph & vector index
        summary_node = self.mem.graph_store.get_node(list(ladybug_conn.nodes.keys())[-1])
        self.assertIsNotNone(summary_node)

if __name__ == "__main__":
    unittest.main()
