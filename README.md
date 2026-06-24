# NebulaMem v3 — Model-Free Spreading-Activation Memory

NebulaMem is a local-first, **in-process** long-term memory engine for AI agents,
modeled on the brain's *spreading activation* and *lateral inhibition*. v3 is
**model-free**: it uses **no embedding model and no LLM** — only classical IR and
self-developed graph algorithms. The only runtime dependency is `numpy`. No GPU,
no external database, no daemon, no network.

## Why model-free?

The blueprint's cognitive core — spreading activation, lateral inhibition,
synaptic decay, Wiki compilation — is pure graph algorithm and needs no model.
The two places that *looked* like they needed heavy models are replaced:

| Needed a model? | Replaced with |
| --- | --- |
| Seed retrieval (dense embeddings) | **BM25** inverted index (+ optional self-developed **Random Indexing**, zero-pretrained semantics grown from your own corpus) |
| Dreaming / consolidation (LLM rewrite) | **Extractive** dedup-merge + edge decay (no text generation) |

## How it works

1. **Seed** — BM25 locates 1–3 seed star-points for the query (on-demand: only
   the query terms' posting lists are touched).
2. **Spread** — energy diffuses across a *self-organizing* associative graph;
   nodes that share proper-noun anchors are auto-wired, so the second hop of a
   multi-hop query is reached even when the query never names it.
3. **Inhibit** — superseded facts carry inhibitory edges; activating the new
   fact drives the stale one below threshold (conflict resolution).
4. **Compile** — fired nodes become a **token-budgeted** Markdown constellation,
   so a large recall can never blow up the LLM context window.

## Measured results (HotpotQA, multi-hop, 1000 questions, CPU, 3.3 s)

| | BM25 only | NebulaMem (+ cascade) |
| --- | --- | --- |
| Recall | 0.52 | **0.75** |
| Multi-hop both-paragraph recovery | 0.42 | **0.71** |
| Bridge-type both-paragraph recovery | 0.44 | **0.74** |

- **On-demand loading**: over a 33,000-node pool, a query materializes only
  ~10 nodes (**0.03%** of the corpus).
- **Context safety**: unbounded recall compiles to **~1.2M tokens**; with the
  `max_results` + `token_budget` guards it stays at **~450 tokens**.

### On-disk, on-demand (LMDB)

For corpora too big to keep in RAM, persist to an **LMDB** memory-mapped backend
(`mem.save(path)` / `NebulaMem.open_disk(path)`). Queries demand-page only the
postings of the query terms plus the content/adjacency of fired nodes — the full
index is never resident. Indexing the **entire** [sub2api](https://github.com/Wei-Shaw/sub2api)
project (2,324 files → 50,133 chunks) and running 2,000 comment→code queries:

- **file recall@30 = 97.9%** (@20 = 94.9%)
- only **~0.06%** of chunk content materialized per query

LMDB (not SQLite) because retrieval is a hand-built inverted index — a pure
key-lookup access pattern that a memory-mapped B+tree serves on-demand.

Full tables, the recall/precision operating curve, and the honest trade-offs are
in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Quick start

```python
from nebulamem import NebulaMem, SpreadingActivationConfig, MemoryNodeType as T

mem = NebulaMem()
mem.register_memory("sync_new", "We migrated the sync engine to a Rust service with RocksDB.", T.STATE)
mem.register_memory("rust", "Rust gives ownership-based memory safety.", T.FACT)
mem.register_memory("sync_old", "The sync engine is a legacy Python script.", T.STATE)
mem.update_state_with_suppression("sync_old", "sync_new")   # new supersedes old

cfg = SpreadingActivationConfig(seed_limit=3, steps=3, max_results=10, token_budget=400)
fired = mem.retrieve("what does our sync engine run on", cfg)  # 'rust' recalled via cascade
print(mem.compile_to_markdown(fired))
```

## Layout

```text
nebulamem/
  text.py           # tokenizer + proper-noun/entity mining (no model)
  lexical.py        # BM25 inverted index (on-demand seed retrieval)
  semantic.py       # self-developed Random Indexing (zero-pretrained semantics)
  graph.py          # associative graph: plasticity, decay, inhibition, auto-wiring
  store.py          # sqlite/in-memory content store with on-demand loading
  core.py           # orchestrator: seed -> spread -> inhibit -> token-budgeted compile
  consolidation.py  # extractive dedup-merge + forgetting (no LLM)
docs/
  BLUEPRINT.md      # cognitive vision
  ARCHITECTURE.md   # model-free technical architecture
  BENCHMARK.md      # measured recall / precision / multi-hop / scale results
bench/benchmark.py  # HotpotQA benchmark harness (reproducible)
demo.py             # runnable cascade + inhibition demo
tests/              # unit tests (no mocks, no models)
```

## Run

```bash
pip install -r requirements.txt
python demo.py                              # demo
python -m unittest discover -s tests        # tests
python bench/benchmark.py --sample 1000     # benchmark (auto-downloads HotpotQA)
```
