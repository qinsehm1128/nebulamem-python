"""Pure-algorithm text processing: tokenization + entity mining.

Zero models, zero pretrained data. Tokenization is a plain Unicode-aware
splitter with a small stopword list; entity mining extracts proper-noun
phrases (capitalized runs and numbers) which act as the "synaptic anchors"
that wire conceptually-related memories together for multi-hop spreading.
"""
import re
from typing import List, Set

# Small in-line stopword set (English). Kept tiny on purpose: lexical recall on
# personal/agent memory benefits from keeping most tokens.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "by", "for", "with", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "them", "his", "her", "their", "i", "you", "we", "do", "does", "did", "has",
    "have", "had", "not", "no", "so", "than", "which", "who", "whom", "what",
    "when", "where", "how", "why", "from", "into", "about", "over", "after",
    "before", "between", "out", "up", "down", "also", "can", "will", "would",
    "there", "here", "both", "more", "most", "some", "such", "only", "own",
}

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
# Capitalized run: one or more Capitalized tokens, optionally joined by short
# connectors (of/the/and/&), e.g. "University of California", "AT&T".
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9.&'’-]*)(?:\s+(?:of|the|and|for|de|von|van|&)\s+|\s+)"
    r"?(?:[A-Z][A-Za-z0-9.&'’-]*)*\b"
)
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9.&'’-]{1,}\b")
_NUM_RE = re.compile(r"\b\d{2,4}\b")  # years / numbers are strong bridge anchors


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """Lowercase alphanumeric tokens. Used by the BM25 lexical index."""
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def char_ngrams(token: str, n: int = 3) -> List[str]:
    """Character n-grams for fuzzy/morphological matching (typos, inflection)."""
    t = f"#{token}#"
    if len(t) <= n:
        return [t]
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def _normalize_entity(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(".,;:'’\"-").lower()


def extract_entities(text: str) -> Set[str]:
    """Mine proper-noun phrases + salient numbers as associative anchors.

    These anchors are what let two memories that mention the same concept get
    auto-wired together — the substrate for cascade / multi-hop retrieval, with
    no embedding model involved.
    """
    ents: Set[str] = set()
    for m in _CAP_TOKEN_RE.findall(text):
        e = _normalize_entity(m)
        if e and e not in _STOPWORDS and len(e) > 1:
            ents.add(e)
    # multi-word capitalized phrases (more specific bridge anchors)
    for m in _ENTITY_RE.findall(text):
        e = _normalize_entity(m)
        if " " in e and len(e) > 2:
            ents.add(e)
    for m in _NUM_RE.findall(text):
        ents.add(m)
    return ents


def normalize_phrase(s: str) -> str:
    """Public helper so callers (e.g. known titles) can register exact anchors."""
    return _normalize_entity(s)
