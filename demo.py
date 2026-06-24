import numpy as np
import os
import shutil
import sys
import time
from unittest.mock import MagicMock

# =====================================================================
# Sandbox Test-Time Mock Setup (Isolating the Environment)
# =====================================================================

# 1. Mock zvec
mock_zvec = MagicMock()
mock_zvec.DataType.STRING = "string"
mock_zvec.DataType.FLOAT_VECTOR = "float_vector"
mock_zvec.MetricType.COSINE = "cosine"

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

# Now we safely import production classes
from nebulamem import (
    NebulaMem, MemoryNodeType, MemoryEdgeType, SpreadingActivationConfig, MemoryConsolidator
)

# Deterministic vector generator for mock embeddings
class DeterministicPRNG:
    def __init__(self, seed: int):
        self.seed = seed

    def generate_vector(self, dim: int = 64) -> list:
        np.random.seed(self.seed)
        self.seed += 1
        vec = np.random.randn(dim)
        normalized = vec / np.linalg.norm(vec)
        return list(normalized.astype(float))

def run_real_world_demo():
    print("=============================================================")
    print("      NebulaMem Python v2: In-Process Cognitive Demo        ")
    print("   Database Stack: Zvec HNSW Vector Index & LadybugDB Graph ")
    print("=============================================================\n")

    store_dir = "./nebulamem_production_store"
    prng = DeterministicPRNG(seed=2026)
    mem = NebulaMem(db_dir=store_dir, dimension=64)

    # 1. Generate real-world fact vectors
    embeddings = {
        "user_profile": prng.generate_vector(),
        "ghostty_terminal": prng.generate_vector(),
        "zellij_layout": prng.generate_vector(),
        "dropshipping_api": prng.generate_vector(),
        "sanda_martial_arts": prng.generate_vector(),
        "old_sync_engine": prng.generate_vector(),
        "new_sync_engine": prng.generate_vector()
    }

    # Semantically link rust/ladybug sync engine with old python sync engine
    embeddings["new_sync_engine"] = list(
        (0.8 * np.array(embeddings["new_sync_engine"]) + 0.2 * np.array(embeddings["old_sync_engine"])).astype(float)
    )

    # 2. Register memories into the system
    print(">>> Ingesting real-world developer & procurement memories...")
    mem.register_memory(
        "user_profile",
        "I am an AI developer based in Yiwu, China, managing dropshipping and logistics APIs (Shopify, DSers, Made-in-China B2B).",
        embeddings["user_profile"],
        MemoryNodeType.ENTITY
    )
    mem.register_memory(
        "ghostty_terminal",
        "Ghostty terminal is an open-source terminal written in Zig, utilizing custom GPU Metal/OpenGL shaders for text rasterization, and natively supporting the Kitty image rendering protocol.",
        embeddings["ghostty_terminal"],
        MemoryNodeType.FACT
    )
    mem.register_memory(
        "zellij_layout",
        "Zellij uses custom KDL layouts (e.g. tab and pane templates) and WebAssembly compiled plugins (e.g. zjstatus) to manage project workspaces via a CLI-native experience.",
        embeddings["zellij_layout"],
        MemoryNodeType.FACT
    )
    mem.register_memory(
        "dropshipping_api",
        "Our Dropshipping logistics API handles cross-border order mapping, linking Zhejiang verified manufacturers with Shopify stores via the DSers OpenAPI interface.",
        embeddings["dropshipping_api"],
        MemoryNodeType.FACT
    )
    mem.register_memory(
        "sanda_martial_arts",
        "I have trained in Sanda (Chinese Kickboxing) for two years in Yiwu, assisting coaches in student sparring and practice sessions.",
        embeddings["sanda_martial_arts"],
        MemoryNodeType.FACT
    )

    # Register the contradiction
    mem.register_memory(
        "old_sync_engine",
        "Our inventory sync engine is a legacy Python script storing state in local SQLite files, causing concurrent write lock contention during peak dropship hours.",
        embeddings["old_sync_engine"],
        MemoryNodeType.STATE
    )
    mem.register_memory(
        "new_sync_engine",
        "We migrated our inventory sync engine to a multi-threaded Rust service utilizing RocksDB and LadybugDB graph database for sub-millisecond locks.",
        embeddings["new_sync_engine"],
        MemoryNodeType.STATE
    )

    # 3. Build associative connections (Star Constellations)
    print(">>> Structuring synapses (Graph edges) inside LadybugDB...")
    mem.associate("user_profile", "dropshipping_api", weight=0.9)
    mem.associate("user_profile", "ghostty_terminal", weight=0.8)
    mem.associate("user_profile", "sanda_martial_arts", weight=0.6)
    mem.associate("ghostty_terminal", "zellij_layout", weight=0.7)

    mem.associate("dropshipping_api", "new_sync_engine", weight=0.95)
    mem.associate("dropshipping_api", "old_sync_engine", weight=0.6)

    # 4. Apply State Suppressions (Contradiction solver)
    print(">>> Injecting state suppression: New Rust sync engine actively inhibits old Python engine...")
    mem.update_state_with_suppression("old_sync_engine", "new_sync_engine")

    # ----------------------------------------------------
    # Query 1: Sourcing/Dropshipping DB sync inquiry
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("SIMULATION 1: Inquiring about dropshipping DB sync stack")
    print("Query: 'What database and script do we use for dropshipping inventory synchronization?'")
    print("=" * 50)
    
    # Query embedding is close to the dropshipping sync state
    query_vector_1 = list((0.85 * np.array(embeddings["new_sync_engine"]) + 0.15 * np.array(embeddings["dropshipping_api"])).astype(float))
    
    activated_1 = mem.retrieve(query_vector_1, SpreadingActivationConfig(steps=3, decay=0.6, fire_threshold=0.15))
    markdown_1 = mem.compile_to_markdown(activated_1)
    print(markdown_1)

    # ----------------------------------------------------
    # Query 2: General Developer Tools inquiry
    # ----------------------------------------------------
    print("=" * 50)
    print("SIMULATION 2: Inquiring about developer CLI workspace tools")
    print("Query: 'What terminal emulator and layouts are we using?'")
    print("=" * 50)

    # Query embedding is close to ghostty terminal
    query_vector_2 = list((0.9 * np.array(embeddings["ghostty_terminal"]) + 0.1 * np.array(embeddings["zellij_layout"])).astype(float))

    activated_2 = mem.retrieve(query_vector_2, SpreadingActivationConfig(steps=3, decay=0.6, fire_threshold=0.15))
    markdown_2 = mem.compile_to_markdown(activated_2)
    print(markdown_2)

    # ----------------------------------------------------
    # Asynchronous Consolidator Pipeline Demonstration
    # ----------------------------------------------------
    print("=" * 50)
    print("SIMULATION 3: Running Asynchronous Consolidation Pipelines")
    print("=" * 50)
    
    consolidator = MemoryConsolidator(
        mem, 
        decay_interval_seconds=1.0, 
        decay_rate=0.01,
        consolidation_interval_seconds=5.0
    )
    
    # Run a manual weight decay to show forgetting
    consolidator.trigger_manual_decay(elapsed_seconds=10.0)
    
    # Run a manual knowledge distillation to compile long-term semantic summary
    summary = consolidator.trigger_manual_consolidation()
    print("\n>>> Abstraction & Distillation Output:")
    print(summary)

if __name__ == "__main__":
    run_real_world_demo()
