from app.core.paths import workspace_path
from collections import defaultdict
import json
import logging
import numpy as np
import re

logger = logging.getLogger("chatmemory.bm25")

_index_cache: dict[str, tuple[float, "Bm25Index"]] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class _Bm25Scorer:
    """Inverted-index BM25 backed by numpy arrays."""

    def __init__(self, corpus_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        n = len(corpus_tokens)
        self._n = n

        doc_lens = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self._avgdl = float(doc_lens.mean()) if n else 1.0
        self._doc_lens = doc_lens

        # Build inverted index: term -> {doc_id: term_freq}
        raw: dict[str, dict[int, int]] = defaultdict(dict)
        df: dict[str, int] = {}
        for doc_id, tokens in enumerate(corpus_tokens):
            tf_local: dict[str, int] = {}
            for t in tokens:
                tf_local[t] = tf_local.get(t, 0) + 1
            for t, freq in tf_local.items():
                raw[t][doc_id] = freq
                df[t] = df.get(t, 0) + 1

        # Convert postings to numpy arrays for fast scatter-add at query time
        self._inverted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        for term, postings in raw.items():
            ids = np.array(list(postings.keys()), dtype=np.int32)
            freqs = np.array(list(postings.values()), dtype=np.float32)
            self._inverted[term] = (ids, freqs)
            freq_count = df[term]
            self._idf[term] = float(np.log((n - freq_count + 0.5) / (freq_count + 0.5) + 1.0))

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self._n, dtype=np.float32)
        for term in set(query_tokens):
            if term not in self._inverted:
                continue
            doc_ids, tf = self._inverted[term]
            idf = self._idf[term]
            dl = self._doc_lens[doc_ids]
            denom = tf + self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
            scores[doc_ids] += idf * (tf * (self._k1 + 1.0)) / np.maximum(denom, 1e-9)
        return scores


class Bm25Index:
    def __init__(self, corpus: list[dict]) -> None:
        self._corpus = corpus
        tokenized = [_tokenize(row["text"]) for row in corpus]
        self._scorer = _Bm25Scorer(tokenized)

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._scorer.score(tokens)
        # filter positives then partial sort to avoid a full O(n log n) sort
        positive = np.where(scores > 0)[0]
        if len(positive) == 0:
            return []
        top_idx = positive[np.argsort(scores[positive])[::-1][:top_k]]
        results: list[dict] = []
        for i in top_idx:
            row = self._corpus[i]
            results.append(
                {
                    "message_id": row["messageId"],
                    "speaker": row["speaker"],
                    "timestamp": row["timestamp"],
                    "snippet": row["text"][:500],
                    "score": float(scores[i]),
                }
            )
        return results


def clear_index_cache(workspace_id: str | None = None) -> None:
    """Drop cached BM25 indexes (call after ingest rewrites corpus.json)."""
    if workspace_id is None:
        _index_cache.clear()
        return
    _index_cache.pop(workspace_id, None)


def load_index(workspace_id: str) -> Bm25Index | None:
    path = workspace_path(workspace_id) / "bm25" / "corpus.json"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _index_cache.get(workspace_id)
    if cached and cached[0] == mtime:
        return cached[1]
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not corpus:
        return None
    index = Bm25Index(corpus)
    _index_cache[workspace_id] = (mtime, index)
    return index


def hybrid_merge(semantic: list[dict], keyword: list[dict], limit: int = 40) -> list[dict]:
    by_id: dict[str, dict] = {}
    for item in semantic + keyword:
        mid = item["message_id"]
        if mid not in by_id or item["score"] > by_id[mid]["score"]:
            by_id[mid] = item
    merged = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)
    return merged[:limit]
