"""Real end-to-end '3 DB + LLM' test on LoCoMo dialogue memory.

Stack (all real, all local):
  * LLM   : OpenBMB MiniCPM5-1B (Q4 GGUF, llama.cpp) — extracts atomic facts
  * Vector: Alibaba Zvec (HNSW, cosine) — stores + retrieves fact embeddings
  * KV    : LMDB — stores fact/turn content
  * embed : bge-small-en-v1.5 (ONNX) — 384-d vectors for Zvec

Compares two memory pipelines on the same questions:
  A) raw dialogue turns -> Zvec dense retrieval        (no LLM)
  B) MiniCPM-extracted facts -> Zvec dense retrieval   (the 3DB+LLM design)

Metric: evidence recall@k — do the retrieved items' source turns cover the gold
evidence dia_ids annotated in LoCoMo.
"""
import ast
import json
import os
import shutil
import sys
import time

import numpy as np
import lmdb
import zvec
from llama_cpp import Llama
from fastembed import TextEmbedding

LOCOMO = "/tmp/claude-0/-home-user-nebulamem-python/9b2ac62b-a52d-5099-9b7c-f06297e13720/scratchpad/locomo/data/locomo10.json"
GGUF = "/tmp/claude-0/-home-user-nebulamem-python/9b2ac62b-a52d-5099-9b7c-f06297e13720/scratchpad/minicpm.gguf"
N_CONV = int(sys.argv[1]) if len(sys.argv) > 1 else 3
K = 5

emb = TextEmbedding("BAAI/bge-small-en-v1.5")
def E(xs):
    v = np.array(list(emb.embed(list(xs))), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

llm = Llama(model_path=GGUF, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)
def extract_facts(speaker, text):
    prompt = (f"Rewrite the information in this chat message as one short standalone "
              f"fact in the third person (mention the speaker by name). "
              f"If it is only small talk, answer NONE.\n"
              f"{speaker}: {text}\nFact:")
    out = llm(prompt, max_tokens=56, temperature=0.1, stop=["\n", "Message", speaker + ":"])["choices"][0]["text"]
    s = out.strip(" -*\"'").strip()
    if len(s) > 8 and "NONE" not in s.upper():
        return [s]
    return []


def zvec_store(path, ids, srcs, vecs):
    shutil.rmtree(path, ignore_errors=True)
    schema = zvec.CollectionSchema(
        name="locomem",
        fields=[zvec.FieldSchema("src", zvec.DataType.STRING)],
        vectors=[zvec.VectorSchema("vec", zvec.DataType.VECTOR_FP32, dimension=int(vecs.shape[1]),
                                   index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE))])
    col = zvec.create_and_open(path, schema)
    def safe(x):
        return x.replace(":", "_").replace("#", "_")
    col.insert([zvec.Doc(id=safe(ids[i]), fields={"src": srcs[i]}, vectors={"vec": vecs[i].tolist()})
                for i in range(len(ids))])
    col.flush()
    return col


def zvec_topk_srcs(col, qv, k):
    res = col.query(queries=[zvec.Query("vec", vector=qv.tolist())], topk=k, output_fields=["src"])
    return [r.fields["src"] for r in res]


def main():
    data = json.load(open(LOCOMO))[:N_CONV]
    lmdb_env = lmdb.open("/tmp/e2e_kv", map_size=2 << 30)  # KV layer (content)
    arms = {"A raw-turn + Zvec": [0.0, 0.0], "B MiniCPM-fact + Zvec": [0.0, 0.0]}
    nq = 0
    t_extract = 0.0
    n_turns = n_facts = 0

    for ci, s in enumerate(data):
        conv = s["conversation"]
        sess = [k for k in conv if k.startswith("session") and isinstance(conv[k], list)]
        turn_ids, turn_txt = [], []
        for k in sess:
            for t in conv[k]:
                turn_ids.append(t["dia_id"])
                turn_txt.append(f'{t["speaker"]}: {t["text"]}')
        allids = set(turn_ids)
        n_turns += len(turn_ids)

        # Arm A: raw turns
        tv = E(turn_txt)
        with lmdb_env.begin(write=True) as txn:
            for i, t in zip(turn_ids, turn_txt):
                txn.put(f"{ci}:{i}".encode(), t.encode())
        colA = zvec_store(f"/tmp/zc_A_{ci}", turn_ids, turn_ids, tv)

        # Arm B: MiniCPM-extracted facts (each tagged with its source turn id)
        f_ids, f_src, f_txt = [], [], []
        t0 = time.time()
        for tid, txt in zip(turn_ids, turn_txt):
            sp, _, body = txt.partition(": ")
            for j, fact in enumerate(extract_facts(sp, body)):
                f_ids.append(f"{tid}#f{j}"); f_src.append(tid); f_txt.append(fact)
        t_extract += time.time() - t0
        n_facts += len(f_ids)
        if not f_ids:
            continue
        fv = E(f_txt)
        with lmdb_env.begin(write=True) as txn:
            for i, t in zip(f_ids, f_txt):
                txn.put(f"{ci}:{i}".encode(), t.encode())
        colB = zvec_store(f"/tmp/zc_B_{ci}", f_ids, f_src, fv)

        # questions
        qs = [qa for qa in s["qa"] if qa.get("evidence")]
        questions = [qa["question"] for qa in qs]
        if not questions:
            continue
        qvs = E(questions)
        for qa, qv in zip(qs, qvs):
            try:
                gold = set(ast.literal_eval(qa["evidence"])) if isinstance(qa["evidence"], str) else set(qa["evidence"])
            except Exception:
                continue
            gold = {g for g in gold if g in allids}
            if not gold:
                continue
            nq += 1
            covA = set(zvec_topk_srcs(colA, qv, K))
            covB = set(zvec_topk_srcs(colB, qv, K))
            arms["A raw-turn + Zvec"][0] += len(gold & covA) / len(gold)
            arms["A raw-turn + Zvec"][1] += 1.0 if gold <= covA else 0.0
            arms["B MiniCPM-fact + Zvec"][0] += len(gold & covB) / len(gold)
            arms["B MiniCPM-fact + Zvec"][1] += 1.0 if gold <= covB else 0.0

    print(json.dumps({
        "conversations": N_CONV, "questions": nq,
        "turns_indexed": n_turns, "facts_extracted": n_facts,
        "llm_extract_seconds": round(t_extract, 1),
        "llm_sec_per_turn": round(t_extract / max(n_turns, 1), 2),
        f"evidence_recall@{K}": {a: round(v[0] / nq, 3) for a, v in arms.items()},
        f"all_evidence@{K}": {a: round(v[1] / nq, 3) for a, v in arms.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
