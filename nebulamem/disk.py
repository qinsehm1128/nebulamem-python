"""On-disk LMDB backend — demand-paged, not loaded wholesale into memory.

Why LMDB (and not SQLite): our retrieval is a hand-built inverted index, so the
access pattern is pure key lookups — `term -> posting list`, `id -> content`,
`id -> adjacency`. LMDB is a memory-mapped B+tree: the OS demand-pages only the
pages a query actually touches, so a query over a multi-GB corpus resident-loads
only the postings of its query terms plus the content/adjacency of fired nodes.
That is exactly "按需加载，不全量进内存", enforced at the storage layer. It is
embedded (no server), single-file, ACID, and read-optimized.

Build once in memory, `persist_to_lmdb`, then `NebulaMem.open_disk` serves
queries from the mmap. The reader holds no full index in RAM.
"""
import json
import math
import struct
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import lmdb

from .text import tokenize
from .types import MemoryEdge, MemoryEdgeType, MemoryNode, MemoryNodeType

# sub-databases inside one env
_DBS = ("stats", "post", "content", "meta", "edges")
_MAP_SIZE = 8 * 1024 * 1024 * 1024  # 8 GiB virtual cap (sparse; not preallocated)


def persist_to_lmdb(mem, path: str) -> dict:
    """Dump an in-memory NebulaMem (lexical + graph + content) to an LMDB env."""
    env = lmdb.open(path, map_size=_MAP_SIZE, max_dbs=len(_DBS), writemap=True)
    handles = {name: env.open_db(name.encode()) for name in _DBS}
    lex, graph, store = mem.lexical, mem.graph, mem.store

    # postings: term -> [[node_id, tf, doc_len], ...] (doc_len inlined to avoid
    # a second lookup per posting at query time)
    with env.begin(write=True) as txn:
        txn.put(b"N", str(lex.N).encode(), db=handles["stats"])
        txn.put(b"TL", str(lex._total_len).encode(), db=handles["stats"])
        for term, posting in lex.postings.items():
            rows = [[doc, tf, lex.doc_len.get(doc, 0)] for doc, tf in posting.items()]
            txn.put(term.encode(), json.dumps(rows).encode(), db=handles["post"])
        for nid, content in store._mem.items():
            txn.put(nid.encode(), content.encode(), db=handles["content"])
        for nid, m in graph.meta.items():
            rec = {"type": m["type"].value, "created_at": m["created_at"],
                   "last_activated": m["last_activated"], "cluster": m.get("cluster")}
            txn.put(nid.encode(), json.dumps(rec).encode(), db=handles["meta"])
        for src, dsts in graph.out.items():
            rows = [[e.target, e.weight, e.edge_type.value, e.updated_at]
                    for e in dsts.values()]
            txn.put(src.encode(), json.dumps(rows).encode(), db=handles["edges"])
    with env.begin() as txn:
        stat = {name: txn.stat(db=handles[name])["entries"] for name in _DBS}
    env.sync()
    env.close()
    return stat


class _DiskNodeStore:
    def __init__(self, env, db):
        self.env, self.db, self.loads = env, db, 0

    def get(self, node_id: str) -> Optional[str]:
        self.loads += 1
        with self.env.begin(db=self.db) as txn:
            v = txn.get(node_id.encode())
        return v.decode() if v is not None else None

    def reset_counter(self):
        self.loads = 0


class _DiskBM25Index:
    def __init__(self, env, post_db, stats_db, k1=1.5, b=0.75):
        self.env, self.post_db, self.k1, self.b = env, post_db, k1, b
        with env.begin(db=stats_db) as txn:
            self.N = int(txn.get(b"N") or 0)
            tl = int(txn.get(b"TL") or 0)
        self.avgdl = (tl / self.N) if self.N else 1.0
        self.touched_docs = 0

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        with self.env.begin(db=self.post_db) as txn:
            for t in q_terms:
                raw = txn.get(t.encode())            # one demand-paged read per term
                if not raw:
                    continue
                rows = json.loads(raw)
                idf = math.log(1.0 + (self.N - len(rows) + 0.5) / (len(rows) + 0.5))
                for doc, tf, dl in rows:
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    scores[doc] += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        self.touched_docs = len(scores)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]


class _DiskGraphStore:
    def __init__(self, env, meta_db, edges_db):
        self.env, self.meta_db, self.edges_db = env, meta_db, edges_db
        self.node_entities = {}  # unused on the read path (clusters drive capping)

    def node_meta(self, node_id: str) -> Optional[dict]:
        with self.env.begin(db=self.meta_db) as txn:
            v = txn.get(node_id.encode())
        if v is None:
            return None
        r = json.loads(v)
        return {"type": MemoryNodeType(r["type"]), "created_at": r["created_at"],
                "last_activated": r["last_activated"], "cluster": r.get("cluster")}

    def edges_from(self, source: str) -> List[MemoryEdge]:
        with self.env.begin(db=self.edges_db) as txn:
            v = txn.get(source.encode())
        if v is None:
            return []
        return [MemoryEdge(source, dst, w, MemoryEdgeType(t), u)
                for dst, w, t, u in json.loads(v)]


def open_disk_backend(path: str):
    """Return (node_store, bm25, graph) read-backed by the LMDB env at `path`."""
    env = lmdb.open(path, readonly=True, max_dbs=len(_DBS), lock=False)
    h = {name: env.open_db(name.encode()) for name in _DBS}
    return (env,
            _DiskNodeStore(env, h["content"]),
            _DiskBM25Index(env, h["post"], h["stats"]),
            _DiskGraphStore(env, h["meta"], h["edges"]))
