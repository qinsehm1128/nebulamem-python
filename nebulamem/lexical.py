"""BM25 lexical seed retrieval — a classic IR algorithm, zero model.

This replaces the dense-embedding "vector model". Documents are scored by
Okapi BM25 over an inverted index. Retrieval is *on-demand*: only the posting
lists of the query's terms are ever touched, so cost scales with the query, not
with the corpus size. This is the property the blueprint calls "按需加载".
"""
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_len: Dict[str, int] = {}
        self.doc_terms: Dict[str, List[str]] = {}
        self.N = 0
        self._total_len = 0
        self._idf_cache: Dict[str, float] = {}
        self._dirty = True

    def add(self, doc_id: str, text: str) -> None:
        terms = tokenize(text)
        self.doc_terms[doc_id] = terms
        self.doc_len[doc_id] = len(terms)
        self._total_len += len(terms)
        tf: Dict[str, int] = defaultdict(int)
        for t in terms:
            tf[t] += 1
        for t, c in tf.items():
            self.postings[t][doc_id] = c
        self.N += 1
        self._dirty = True

    def remove(self, doc_id: str) -> None:
        terms = self.doc_terms.pop(doc_id, None)
        if terms is None:
            return
        self._total_len -= self.doc_len.pop(doc_id, 0)
        for t in set(terms):
            self.postings[t].pop(doc_id, None)
            if not self.postings[t]:
                del self.postings[t]
        self.N -= 1
        self._dirty = True

    @property
    def avgdl(self) -> float:
        return (self._total_len / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
        if self._dirty:
            self._idf_cache.clear()
            self._dirty = False
        if term in self._idf_cache:
            return self._idf_cache[term]
        n_q = len(self.postings.get(term, {}))
        # Okapi BM25 idf with +0.5 smoothing (non-negative floor).
        idf = math.log(1.0 + (self.N - n_q + 0.5) / (n_q + 0.5))
        self._idf_cache[term] = idf
        return idf

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Score only the documents that share a term with the query.

        Returns (doc_id, score) sorted desc. `touched_docs` after a call reports
        how many distinct docs were scored (for on-demand-load accounting).
        """
        q_terms = tokenize(query)
        avgdl = self.avgdl or 1.0
        scores: Dict[str, float] = defaultdict(float)
        for t in q_terms:
            posting = self.postings.get(t)
            if not posting:
                continue
            idf = self._idf(t)
            for doc_id, tf in posting.items():
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        self.touched_docs = len(scores)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
