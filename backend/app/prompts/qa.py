"""Q&A pipeline prompts — query rewrite, chunk rerank, grounded answer."""

from __future__ import annotations


def qa_rewrite_query(question: str) -> tuple[str, str]:
    """Return (system_text, user_text) for the multi-query rewrite step.

    The system prompt asks for 2-3 search queries covering different phrasings
    and language variants so cross-language retrieval (Hinglish ↔ English) works.
    Each query is on its own line with no numbering or prefixes.
    """
    system = (
        "Generate 2-3 search queries for a chat history search engine.\n"
        "Output 2-3 search queries (one per line) covering:\n"
        "1. The original question rephrased for search\n"
        "2. Key English terms/concepts from the question\n"
        "3. The Hinglish phrasing if applicable\n\n"
        "Output only the queries, one per line, no numbering, no prefixes."
    )
    return system, question


def qa_rerank_chunks(question: str, numbered_snippets: str) -> str:
    """Return full prompt string for JSON-score reranking of retrieved snippets.

    ``numbered_snippets`` is a newline-joined list of "[N] (speaker, ts): text" lines.
    The model replies with a JSON array of {id, score} objects.
    """
    return (
        "Score each snippet 0.0 to 1.0 for relevance to the question.\n"
        "Reply with ONLY a JSON array of objects. Each object must have numeric fields "
        '"id" (snippet number) and "score" (single number 0.0-1.0).\n'
        'Example: [{"id":1,"score":0.9},{"id":2,"score":0.1}]\n\n'
        f"Question: {question}\n\n{numbered_snippets}"
    )


def qa_grounded_answer(question: str, context: str) -> tuple[str, str]:
    """Return (system_text, user_text) for final grounded-answer generation.

    ``context`` is the formatted string of chat messages (raw snippets or
    context-window blocks built by ``_format_context_blocks``).
    """
    system = (
        "Answer ONLY using the provided chat messages. "
        "Match the language style of the question (Hinglish or English). "
        "If insufficient, say you cannot find it in the chat."
    )
    user = f"Question: {question}\n\nMessages:\n{context}"
    return system, user
