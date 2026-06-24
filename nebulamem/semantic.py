"""Self-developed semantic layer: incremental Random Indexing (zero pretrained).

This is the "不惜一切代价自研" answer to dropping the embedding model. Instead of
a trained network, semantic similarity is *grown locally* from the user's own
corpus using Random Indexing (Kanerva/Sahlgren) — a 1990s-vintage pure
algorithm:

  * every term gets a fixed sparse ternary random "index vector" (seeded
    deterministically from a hash of the term, so no RNG state is needed);
  * each term also accumulates a dense "context vector" = the sum of the index
    vectors of the terms it co-occurs with (second-order statistics);
  * a document/query vector is the sum of its terms' context vectors.

Terms that appear in similar contexts (synonyms, domain jargon) drift to similar
context vectors — so cosine similarity captures paraphrase that pure BM25 misses.
It needs no pretrained weights and improves as the agent accumulates memory.
"""
import hashlib
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .text import tokenize


class RandomIndex:
    def __init__(self, dim: int = 256, nonzero: int = 10, window: int = 8):
        self.dim = dim
        self.nonzero = nonzero  # number of +-1 entries in each sparse index vector
        self.window = window
        self._index_cache: Dict[str, np.ndarray] = {}
        self.context: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.dim, dtype=np.float32))

    def _index_vector(self, term: str) -> np.ndarray:
        v = self._index_cache.get(term)
        if v is not None:
            return v
        vec = np.zeros(self.dim, dtype=np.float32)
        # Deterministic placement of +1/-1 from the term's hash digest.
        h = hashlib.sha256(term.encode("utf-8")).digest()
        for i in range(self.nonzero):
            # two bytes pick a position, one bit picks the sign
            pos = (h[(2 * i) % len(h)] << 8 | h[(2 * i + 1) % len(h)]) % self.dim
            sign = 1.0 if (h[(i) % len(h)] & 1) else -1.0
            vec[pos] += sign
        self._index_cache[term] = vec
        return vec

    def add(self, text: str) -> None:
        """Update term context vectors from a sliding window over this text."""
        toks = tokenize(text)
        n = len(toks)
        for i, t in enumerate(toks):
            lo, hi = max(0, i - self.window), min(n, i + self.window + 1)
            acc = self.context[t]
            for j in range(lo, hi):
                if j == i:
                    continue
                acc += self._index_vector(toks[j])

    def vector(self, text: str) -> np.ndarray:
        """Compose a dense semantic vector for a document or query."""
        toks = tokenize(text)
        v = np.zeros(self.dim, dtype=np.float32)
        for t in toks:
            ctx = self.context.get(t)
            if ctx is not None:
                v += ctx
            else:
                v += self._index_vector(t)  # fall back to first-order signature
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
