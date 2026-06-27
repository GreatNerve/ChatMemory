import json
import math
import re

from app.core.paths import workspace_path


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class _Bm25Scorer:
    """Pure-Python BM25 — no numpy (avoids DLL blocks on locked-down Windows)."""

    def __init__(self, corpus_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs = corpus_tokens
        self._n = len(corpus_tokens)
        self._doc_len = [len(d) for d in corpus_tokens]
        self._avgdl = sum(self._doc_len) / self._n if self._n else 0.0
        self._df: dict[str, int] = {}
        for doc in corpus_tokens:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def score(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for i, doc in enumerate(self._docs):
            doc_len = self._doc_len[i]
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            total = 0.0
            for term in query_tokens:
                freq = self._df.get(term, 0)
                if freq == 0:
                    continue
                idf = math.log((self._n - freq + 0.5) / (freq + 0.5) + 1.0)
                term_freq = tf.get(term, 0)
                denom = term_freq + self._k1 * (
                    1.0 - self._b + self._b * doc_len / (self._avgdl or 1.0)
                )
                total += idf * (term_freq * (self._k1 + 1.0)) / (denom or 1.0)
            scores.append(total)
        return scores


class Bm25Index:
    def __init__(self, corpus: list[dict]) -> None:
        self._corpus = corpus
        tokenized = [_tokenize(row["text"]) for row in corpus]
        self._scorer = _Bm25Scorer(tokenized)

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = _tokenize(query)
        scores = self._scorer.score(tokens)
        ranked = sorted(
            zip(scores, self._corpus, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        results: list[dict] = []
        for score, row in ranked:
            if score <= 0:
                continue
            results.append(
                {
                    "message_id": row["messageId"],
                    "speaker": row["speaker"],
                    "timestamp": row["timestamp"],
                    "snippet": row["text"][:500],
                    "score": float(score),
                }
            )
        return results


def load_index(workspace_id: str) -> Bm25Index | None:
    path = workspace_path(workspace_id) / "bm25" / "corpus.json"
    if not path.exists():
        return None
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not corpus:
        return None
    return Bm25Index(corpus)


def hybrid_merge(
    semantic: list[dict], keyword: list[dict], limit: int = 40
) -> list[dict]:
    by_id: dict[str, dict] = {}
    for item in semantic + keyword:
        mid = item["message_id"]
        if mid not in by_id or item["score"] > by_id[mid]["score"]:
            by_id[mid] = item
    merged = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)
    return merged[:limit]
