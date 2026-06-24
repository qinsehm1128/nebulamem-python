"""Optional dense-vector layer for hybrid retrieval.

NebulaMem stays model-free by default. When a caller supplies an `embedder`
(any callable `List[str] -> np.ndarray` of L2-normalized rows), the engine adds a
dense channel and fuses it with the BM25 / spreading-activation score. Vectors
are stored **int8-quantized** (4x smaller than float32) so the footprint goal
stays reachable.

A thin loader for a local CPU embedder (fastembed/bge-small, ONNX, no torch) is
provided but not required — you can plug Alibaba Zvec's DefaultLocalDenseEmbedding
or Qwen3-Embedding just as well.
"""
from typing import Callable, List, Optional

import numpy as np

Embedder = Callable[[List[str]], np.ndarray]


def quantize_int8(v: np.ndarray) -> np.ndarray:
    """L2-normalize then map to int8 (values assumed in [-1,1])."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return np.clip(np.round(v * 127.0), -127, 127).astype(np.int8)


def dequantize_int8(q: np.ndarray) -> np.ndarray:
    v = q.astype(np.float32) / 127.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_int8(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two int8 vectors (decoded)."""
    av, bv = dequantize_int8(a), dequantize_int8(b)
    return float(av @ bv)


def load_fastembed(model: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """Local ONNX embedder (≈130 MB, CPU, no torch). Lazy import."""
    from fastembed import TextEmbedding
    _m = TextEmbedding(model)

    def _embed(texts: List[str]) -> np.ndarray:
        vecs = np.array(list(_m.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    return _embed
