"""Index a whole codebase into NebulaMem on the LMDB disk backend, then test
retrieval recall — entirely on-demand from disk (no full load into memory).

Corpus: a real project (default: the cloned sub2api source). Every source/doc
file is chunked (cluster = file path) and registered. The index is persisted to
LMDB and reopened read-only; queries demand-page postings + fired-node content.

Recall queries are self-supervised, leakage-aware:
  * comment->code : a doc comment above a func/type is the query; gold = the
    chunk containing that definition (natural-language -> code retrieval).
  * heading->section : a markdown heading is the query; gold = its chunk.

Reports recall@k, MRR, and the on-demand load fraction over the full corpus.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nebulamem import NebulaMem, SpreadingActivationConfig, MemoryNodeType

CODE_EXT = {".go", ".ts", ".tsx", ".js", ".vue", ".sql", ".py", ".sh", ".yaml", ".yml"}
DOC_EXT = {".md", ".mdx", ".txt", ".rst"}
SKIP_DIRS = {".git", "node_modules", "dist", "build", "vendor", ".next", "assets"}
CHUNK_LINES = 30
CHUNK_STRIDE = 18  # overlap so a comment and the def below it co-occur in a chunk

_GO_DEF = re.compile(r"^\s*(?:func|type)\s+[A-Za-z_]")
_TS_DEF = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|const|type)\s+[A-Za-z_]")
_COMMENT = re.compile(r"^\s*(?://|\*|/\*\*?)\s?(.*)")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)")


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in CODE_EXT or ext in DOC_EXT:
                yield os.path.join(dirpath, fn), ext


def chunk_file(lines):
    """Overlapping chunks. Returns (chunks, line_to_chunks) where a line may map
    to several chunk indices because of the overlap."""
    chunks, line_to_chunks = [], {}
    starts = list(range(0, max(1, len(lines)), CHUNK_STRIDE))
    for ci, start in enumerate(starts):
        block = lines[start:start + CHUNK_LINES]
        chunks.append("".join(block))
        for ln in range(start, min(start + CHUNK_LINES, len(lines))):
            line_to_chunks.setdefault(ln, []).append(ci)
    return chunks, line_to_chunks


def _gold_chunks(rel, lo, hi, l2c):
    """All chunk ids covering the line region [lo, hi] (the relevant code unit)."""
    s = set()
    for ln in range(lo, hi + 1):
        for ci in l2c.get(ln, ()):
            s.add(f"{rel}#{ci}")
    return s


def extract_queries(lines, ext, rel, l2c):
    """Yield (query_text, gold_chunk_set) pairs, leakage-aware."""
    out = []
    if ext in DOC_EXT:
        for i, line in enumerate(lines):
            m = _HEADING.match(line)
            if m and len(m.group(2).split()) >= 2:
                out.append((m.group(2).strip(), _gold_chunks(rel, i, min(i + 5, len(lines) - 1), l2c)))
        return out
    defre = _GO_DEF if ext == ".go" else _TS_DEF
    for i, line in enumerate(lines):
        if not defre.match(line):
            continue
        comment, j = [], i - 1
        while j >= 0:
            cm = _COMMENT.match(lines[j])
            if cm and cm.group(1).strip():
                comment.append(cm.group(1).strip())
                j -= 1
            else:
                break
        if not comment:
            continue
        comment.reverse()
        q = re.sub(r"[*/]+", " ", " ".join(comment)).strip()
        if len(q.split()) >= 5:  # substantive NL comment only
            out.append((q, _gold_chunks(rel, j + 1, i, l2c)))  # comment..def region
    return out


def build_index(root, db_path, file_limit=None):
    mem = NebulaMem()
    queries = []
    n_files = n_chunks = 0
    files = list(iter_files(root))
    if file_limit:
        files = files[:file_limit]
    for path, ext in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue
        if not lines:
            continue
        rel = os.path.relpath(path, root)
        chunks, l2c = chunk_file(lines)
        ntype = MemoryNodeType.FACT if ext in DOC_EXT else MemoryNodeType.ENTITY
        for ci, body in enumerate(chunks):
            if body.strip():
                mem.register_memory(f"{rel}#{ci}", body, ntype,
                                    extra_entities=[rel], cluster=rel, auto_link=True)
                n_chunks += 1
        queries.extend(extract_queries(lines, ext, rel, l2c))
        n_files += 1
    t = time.time()
    stats = mem.save(db_path)
    return {"files": n_files, "chunks": n_chunks, "queries": len(queries),
            "lmdb_entries": stats, "persist_sec": round(time.time() - t, 2)}, queries


def test_recall(db_path, queries, total_chunks, sample, ks=(1, 3, 5, 10, 20, 30)):
    mem = NebulaMem.open_disk(db_path)
    # For direct lookup, seeds must fill the result set (seed_limit >= max_results)
    # so lexically-ranked hits are not displaced by cascade neighbours; cascade is
    # for multi-hop, not single-shot retrieval.
    cfg = SpreadingActivationConfig(seed_limit=max(ks), steps=1, decay=0.6,
                                    fire_threshold=0.02, max_results=max(ks))
    import random
    qs = queries if len(queries) <= sample else [queries[i] for i in
          range(0, len(queries), max(1, len(queries) // sample))][:sample]
    def fresh():
        return {k: 0 for k in ks}
    reg, fil = fresh(), fresh()                 # NebulaMem (BM25 seed + cascade)
    breg, bfil = fresh(), fresh()               # BM25-only ablation
    rr, load_fracs, touched = 0.0, [], []
    n = 0
    for q, gold in qs:
        if not gold:
            continue
        gold_files = {g.split("#")[0] for g in gold}

        def score(ranked, reg_d, fil_d):
            for k in ks:
                topk = ranked[:k]
                if any(nid in gold for nid in topk):
                    reg_d[k] += 1
                if any(nid.split("#")[0] in gold_files for nid in topk):
                    fil_d[k] += 1

        mem.store.reset_counter()
        fired = mem.retrieve(q, cfg)
        ranked = [f.id for f in fired]
        load_fracs.append(mem.store.loads / total_chunks)
        touched.append(mem.lexical.touched_docs)
        n += 1
        best = next((r for r, nid in enumerate(ranked, 1) if nid in gold), None)
        if best:
            rr += 1.0 / best
        score(ranked, reg, fil)
        # ablation: pure BM25 top-k, no graph
        bm = [nid for nid, _ in mem.lexical.search(q, max(ks))]
        score(bm, breg, bfil)
    return {
        "queries_tested": n,
        "nebula_region_recall_at_k": {f"@{k}": round(reg[k] / n, 4) for k in ks},
        "nebula_file_recall_at_k": {f"@{k}": round(fil[k] / n, 4) for k in ks},
        "bm25_only_file_recall_at_k": {f"@{k}": round(bfil[k] / n, 4) for k in ks},
        "mrr": round(rr / n, 4),
        "avg_ondemand_load_fraction": round(sum(load_fracs) / n, 6),
        "avg_postings_scanned_docs": round(sum(touched) / n, 1),
        "corpus_chunks": total_chunks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--db", default="/tmp/nebula_code.lmdb")
    ap.add_argument("--file-limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "code_results.json"))
    args = ap.parse_args()

    os.system(f"rm -rf {args.db}")
    t0 = time.time()
    build, queries = build_index(args.root, args.db, args.file_limit)
    build["build_sec"] = round(time.time() - t0, 2)
    recall = test_recall(args.db, queries, build["chunks"], args.sample)
    result = {"corpus": os.path.basename(args.root.rstrip("/")), "build": build,
              "recall": recall}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
