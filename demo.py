"""NebulaMem model-free demo — no embedding model, no LLM, no mocks.

Shows the three things the blueprint promises, on a tiny in-memory star-field:
  1. multi-hop cascade  — a fact never named in the query is recalled via a bridge
  2. lateral inhibition  — a superseded fact is actively suppressed
  3. token-budgeted Wiki — compiled context stays bounded
"""
from nebulamem import (NebulaMem, MemoryConsolidator, SpreadingActivationConfig,
                       MemoryNodeType as T)


def main():
    mem = NebulaMem(use_semantic=False)

    mem.register_memory("user", "I am a developer in Yiwu running cross-border "
                        "dropshipping logistics on Shopify and DSers.", T.ENTITY)
    mem.register_memory("sync_new", "We migrated the inventory sync engine to a "
                        "multi-threaded Rust service using RocksDB for sub-millisecond locks.", T.STATE)
    mem.register_memory("rust", "Rust is a systems language with ownership-based "
                        "memory safety and fearless concurrency.", T.FACT)
    mem.register_memory("sync_old", "The inventory sync engine is a legacy Python "
                        "script storing state in local SQLite files.", T.STATE)
    mem.register_memory("sanda", "I train Sanda kickboxing twice a week in Yiwu.", T.FACT)

    # the new Rust engine supersedes the old Python one
    mem.update_state_with_suppression("sync_old", "sync_new")

    cfg = SpreadingActivationConfig(seed_limit=3, steps=3, decay=0.7,
                                    fire_threshold=0.18, max_results=8, token_budget=300)

    # NOTE: the query never mentions "Rust" — it must be reached by cascade.
    q = "what does our inventory sync engine run on"
    print(f"QUERY: {q}\n")
    fired = mem.retrieve(q, cfg)
    print(mem.compile_to_markdown(fired))

    ids = {n.id for n in fired}
    print("\n--- checks ---")
    print("multi-hop: 'rust' recalled via bridge (not in query):", "rust" in ids)
    print("inhibition: stale 'sync_old' suppressed:", "sync_old" not in ids)

    cons = MemoryConsolidator(mem)
    cons.decay(idle_seconds=10.0)
    print("consolidation dedup stats:", cons.deduplicate())


if __name__ == "__main__":
    main()
