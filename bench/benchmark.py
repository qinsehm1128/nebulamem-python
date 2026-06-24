"""NebulaMem empirical benchmark on HotpotQA (distractor, multi-hop).

Measures, with NO embedding model and NO LLM:
  * supporting-fact recall / precision / F1 (sentence level, HotpotQA sp metric)
  * paragraph-level recall and the multi-hop *bridge* (2nd hop) recovery rate
  * cascade ablation: BM25-only vs BM25 + spreading activation
  * on-demand load fraction over a large pooled memory
  * context-window safety: compiled token size, uncapped vs token-budgeted

Run:  python bench/benchmark.py --sample 1000
Data: HotpotQA distractor validation parquet (auto-downloaded if missing).
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nebulamem import NebulaMem, SpreadingActivationConfig, MemoryNodeType, estimate_tokens
from nebulamem.lexical import BM25Index
from nebulamem.text import normalize_phrase

DATA_URL = ("https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/"
            "parquet/distractor/validation/0.parquet")


def load_rows(path, sample):
    if not os.path.exists(path):
        import urllib.request
        print(f"downloading HotpotQA -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    t = pq.read_table(path)
    rows = t.slice(0, sample).to_pylist() if sample else t.to_pylist()
    return rows


def build_memory(row, use_semantic=False):
    """One memory per question: sentence = node, paragraph title = shared anchor."""
    mem = NebulaMem(use_semantic=use_semantic)
    titles = row["context"]["title"]
    sents = row["context"]["sentences"]
    node_meta = {}  # node_id -> (title, sent_id)
    for p_idx, (title, para) in enumerate(zip(titles, sents)):
        for s_idx, sent in enumerate(para):
            if not sent or not sent.strip():
                continue
            nid = f"{p_idx}:{s_idx}"
            mem.register_memory(nid, sent, MemoryNodeType.FACT,
                                extra_entities=[title])
            node_meta[nid] = (normalize_phrase(title), s_idx)
    return mem, node_meta


def gold_set(row):
    """Gold supporting facts as (normalized_title, sent_id)."""
    g = set()
    for title, sid in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        g.add((normalize_phrase(title), sid))
    return g


def prf(pred, gold):
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_row(row, cfg, use_semantic=False):
    """Build one memory, derive the SHARED seeds, then compare:
      * bm25_pred  = the lexical seeds only (no graph)        [existing algorithm]
      * neb_pred   = same seeds + spreading activation cascade [self-developed]
    Identical starting point => the only variable is the cascade.
    """
    mem, meta = build_memory(row, use_semantic)
    seeds = mem._seeds(row["question"], cfg)            # nid -> energy
    bm25_pred = {meta[nid] for nid in seeds if nid in meta}
    fired = mem.retrieve(row["question"], cfg)
    neb_pred = {meta[n.id] for n in fired if n.id in meta}
    return bm25_pred, neb_pred


def paragraph_recall(pred, gold):
    """Did we recover at least one supporting sentence from each gold paragraph?"""
    gold_titles = {t for t, _ in gold}
    pred_titles = {t for t, _ in pred}
    recovered = gold_titles & pred_titles
    full = 1.0 if gold_titles and recovered == gold_titles else 0.0
    return len(recovered), len(gold_titles), full


def run_quality(rows, cfg):
    agg = defaultdict(list)
    by_type = defaultdict(lambda: defaultdict(list))
    for row in rows:
        gold = gold_set(row)
        if not gold:
            continue
        qtype, level = row.get("type", "?"), row.get("level", "?")

        # Shared seeds: BM25 seeds (no graph) vs same seeds + cascade
        b_pred, n_pred = evaluate_row(row, cfg)
        bp, br, bf = prf(b_pred, gold)
        _, _, b_full = paragraph_recall(b_pred, gold)
        np_, nr, nf = prf(n_pred, gold)
        _, _, n_full = paragraph_recall(n_pred, gold)

        agg["bm25_P"].append(bp); agg["bm25_R"].append(br); agg["bm25_F1"].append(bf)
        agg["bm25_paraFull"].append(b_full)
        agg["neb_P"].append(np_); agg["neb_R"].append(nr); agg["neb_F1"].append(nf)
        agg["neb_paraFull"].append(n_full)
        by_type[qtype]["bm25_paraFull"].append(b_full)
        by_type[qtype]["neb_paraFull"].append(n_full)
        by_type[qtype]["bm25_F1"].append(bf)
        by_type[qtype]["neb_F1"].append(nf)
    return agg, by_type


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def run_scale(rows, n_pool, n_probe, cfg_budget):
    """Pool many questions into ONE memory; test on-demand load + context size."""
    mem = NebulaMem()
    probe = []
    node_meta = {}
    total_nodes = 0
    for q_idx, row in enumerate(rows[:n_pool]):
        titles, sents = row["context"]["title"], row["context"]["sentences"]
        for p_idx, (title, para) in enumerate(zip(titles, sents)):
            for s_idx, sent in enumerate(para):
                if not sent or not sent.strip():
                    continue
                nid = f"q{q_idx}:{p_idx}:{s_idx}"
                mem.register_memory(nid, sent, MemoryNodeType.FACT, extra_entities=[title])
                node_meta[nid] = (q_idx, normalize_phrase(title), s_idx)
                total_nodes += 1
        if len(probe) < n_probe:
            probe.append((q_idx, row))

    loads_frac, recalls = [], []
    uncapped_tokens, capped_tokens = [], []
    # truly unbounded recall to expose context blow-up, vs the budgeted guard
    cfg_uncapped = SpreadingActivationConfig(**{**cfg_budget.__dict__,
                                                "token_budget": None,
                                                "max_results": 100000})
    for q_idx, row in probe:
        gold = {(q_idx, t, s) for (t, s) in gold_set(row)}
        # NORMAL operation (max_results cap + token budget): measures the real
        # on-demand load fraction and recall a deployment would see.
        mem.store.reset_counter()
        fired_b = mem.retrieve(row["question"], cfg_budget)
        loads_frac.append(mem.store.loads / total_nodes)
        pred = {node_meta[n.id] for n in fired_b if n.id in node_meta}
        recalls.append(len(pred & gold) / len(gold) if gold else 0.0)
        capped_tokens.append(estimate_tokens(mem.compile_to_markdown(fired_b)))
        # UNCAPPED (no max_results, no budget): demonstrates context blow-up.
        fired_u = mem.retrieve(row["question"], cfg_uncapped)
        uncapped_tokens.append(estimate_tokens(mem.compile_to_markdown(fired_u)))
    return {
        "pool_questions": n_pool,
        "total_nodes": total_nodes,
        "probe_count": len(probe),
        "avg_ondemand_load_fraction": mean(loads_frac),
        "avg_recall_at_scale": mean(recalls),
        "avg_compiled_tokens_uncapped": mean(uncapped_tokens),
        "max_compiled_tokens_uncapped": max(uncapped_tokens) if uncapped_tokens else 0,
        "avg_compiled_tokens_capped": mean(capped_tokens),
        "max_compiled_tokens_capped": max(capped_tokens) if capped_tokens else 0,
        "token_budget": cfg_budget.token_budget,
    }


def run_maxresults_sweep(rows, base_cfg, caps):
    """Expose the recall/precision operating curve as max_results varies."""
    out = []
    for k in caps:
        cfg = SpreadingActivationConfig(**{**base_cfg.__dict__, "max_results": k})
        P, R, F1, full = [], [], [], []
        for row in rows:
            gold = gold_set(row)
            if not gold:
                continue
            _, n_pred = evaluate_row(row, cfg)
            p, r, f = prf(n_pred, gold)
            P.append(p); R.append(r); F1.append(f)
            _, _, fl = paragraph_recall(n_pred, gold)
            full.append(fl)
        out.append({"max_results": k, "P": mean(P), "R": mean(R),
                    "F1": mean(F1), "both_paragraphs_recovered": mean(full)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "..",
                    "scratchpad", "hotpot_val.parquet"))
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--probe", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()

    rows = load_rows(args.data, max(args.sample, args.pool))
    print(f"loaded {len(rows)} questions")

    cfg = SpreadingActivationConfig(seed_limit=3, steps=3, decay=0.7,
                                    inhibition=0.95, fire_threshold=0.18,
                                    max_results=10)
    cfg_budget = SpreadingActivationConfig(seed_limit=3, steps=3, decay=0.7,
                                           inhibition=0.95, fire_threshold=0.18,
                                           max_results=10, token_budget=400)

    t0 = time.time()
    agg, by_type = run_quality(rows[:args.sample], cfg)
    q_time = time.time() - t0

    sweep = run_maxresults_sweep(rows[:min(args.sample, 500)], cfg,
                                 [2, 3, 5, 8, 12, 20])
    scale = run_scale(rows, args.pool, args.probe, cfg_budget)

    result = {
        "n_questions": args.sample,
        "config": cfg.__dict__,
        "quality": {
            "bm25_only": {"P": mean(agg["bm25_P"]), "R": mean(agg["bm25_R"]),
                          "F1": mean(agg["bm25_F1"]),
                          "both_paragraphs_recovered": mean(agg["bm25_paraFull"])},
            "nebulamem": {"P": mean(agg["neb_P"]), "R": mean(agg["neb_R"]),
                          "F1": mean(agg["neb_F1"]),
                          "both_paragraphs_recovered": mean(agg["neb_paraFull"])},
        },
        "by_type": {
            qt: {"bm25_bothParas": mean(d["bm25_paraFull"]),
                 "nebula_bothParas": mean(d["neb_paraFull"]),
                 "bm25_F1": mean(d["bm25_F1"]),
                 "nebula_F1": mean(d["neb_F1"]),
                 "n": len(d["neb_F1"])}
            for qt, d in by_type.items()
        },
        "max_results_sweep": sweep,
        "scale": scale,
        "timing_sec": {"quality_eval": round(q_time, 1)},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
