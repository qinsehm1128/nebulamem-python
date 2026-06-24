"""Unit tests for the model-free NebulaMem (no mocks, no models, no LLM)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nebulamem import (NebulaMem, MemoryConsolidator, SpreadingActivationConfig,
                       MemoryNodeType, BM25Index, RandomIndex, estimate_tokens)


class TestLexical(unittest.TestCase):
    def test_bm25_ranks_and_is_on_demand(self):
        idx = BM25Index()
        idx.add("d1", "the rust inventory sync engine uses rocksdb")
        idx.add("d2", "sanda kickboxing training in the gym")
        idx.add("d3", "rust ownership memory safety concurrency")
        ranked = idx.search("rust sync engine", limit=3)
        self.assertEqual(ranked[0][0], "d1")
        # on-demand: only docs sharing a query term are scored (d2 shares none)
        self.assertLessEqual(idx.touched_docs, 2)

    def test_remove(self):
        idx = BM25Index()
        idx.add("d1", "alpha beta")
        idx.add("d2", "alpha gamma")
        idx.remove("d1")
        ranked = idx.search("alpha", limit=5)
        self.assertEqual([r[0] for r in ranked], ["d2"])


class TestSemantic(unittest.TestCase):
    def test_random_indexing_deterministic_and_nonzero(self):
        ri = RandomIndex(dim=128)
        ri.add("rust systems programming language memory safety")
        v = ri.vector("rust language")
        self.assertEqual(v.shape[0], 128)
        # determinism: same term -> same index vector across instances
        self.assertTrue((RandomIndex(dim=128)._index_vector("rust")
                         == ri._index_vector("rust")).all())


class TestRetrieval(unittest.TestCase):
    def _mem(self):
        m = NebulaMem()
        m.register_memory("user", "I am a developer in Yiwu doing dropshipping.", MemoryNodeType.ENTITY)
        m.register_memory("sync_new", "We migrated the inventory sync engine to a Rust service with RocksDB.", MemoryNodeType.STATE)
        m.register_memory("rust", "Rust gives ownership-based memory safety and fearless concurrency.", MemoryNodeType.FACT)
        m.register_memory("sync_old", "The inventory sync engine is a legacy Python script using SQLite.", MemoryNodeType.STATE)
        m.update_state_with_suppression("sync_old", "sync_new")
        return m

    def test_multi_hop_cascade(self):
        m = self._mem()
        cfg = SpreadingActivationConfig(seed_limit=3, steps=3, decay=0.7, fire_threshold=0.15)
        # query never says "Rust"; it must be reached via the bridge entity
        fired = {n.id for n in m.retrieve("what does our inventory sync engine run on", cfg)}
        self.assertIn("sync_new", fired)
        self.assertIn("rust", fired)  # recovered by cascade

    def test_lateral_inhibition(self):
        m = self._mem()
        cfg = SpreadingActivationConfig(seed_limit=3, steps=3, decay=0.7, fire_threshold=0.15)
        fired = {n.id for n in m.retrieve("inventory sync engine database", cfg)}
        self.assertIn("sync_new", fired)
        self.assertNotIn("sync_old", fired)  # superseded fact suppressed

    def test_token_budget_caps_context(self):
        m = NebulaMem()
        for i in range(40):
            m.register_memory(f"n{i}", "rust inventory sync engine rocksdb " * 8, MemoryNodeType.FACT)
        cfg = SpreadingActivationConfig(seed_limit=3, steps=2, max_results=100, token_budget=120)
        fired = m.retrieve("rust inventory sync engine", cfg)
        self.assertLessEqual(estimate_tokens(m.compile_to_markdown(fired)), 200)


class TestConsolidation(unittest.TestCase):
    def test_decay_and_dedup(self):
        m = NebulaMem()
        m.register_memory("a", "Rust inventory sync engine on RocksDB.", MemoryNodeType.STATE)
        m.register_memory("b", "Rust inventory sync engine on RocksDB.", MemoryNodeType.STATE)  # dup
        m.register_memory("c", "Sanda kickboxing twice a week.", MemoryNodeType.FACT)
        cons = MemoryConsolidator(m, dedup_threshold=0.8)
        cons.decay(idle_seconds=10.0)
        stats = cons.deduplicate()
        self.assertGreaterEqual(stats["merged"], 1)


if __name__ == "__main__":
    unittest.main()
