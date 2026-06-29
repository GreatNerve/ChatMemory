"""History-router classify prompt — Gemini structured intent detection.

Used by ``history_router.classify_history_need`` when the fast heuristic route
returns "ambiguous".  The model returns JSON with ``needs_history``,
``needs_rewrite``, ``query_intent``, and ``search_queries`` (list of 1-4 search
phrases covering different language phrasings so cross-language Hinglish↔English
lookups succeed).

``query_intent`` controls recency boost direction in retrieval:
- ``"current"``    — question is about present state / recent events
- ``"historical"`` — question explicitly references the past
- ``"neutral"``    — ambiguous or general facts (default when unsure)
"""

from __future__ import annotations


def rewrite_query_prompt(user_message: str, context_block: str) -> str:
    """Return a prompt that rewrites a short follow-up into 3-5 standalone search query variations.

    ``user_message`` is the raw (short) follow-up, e.g. "kaha?" or "kab?".
    ``context_block`` is the last 4 turns formatted as "User: …\\nPersona: …" lines.

    The model must reply with a JSON array of 3-5 short search phrases (2-8 words each)
    that cover BOTH English and Hinglish phrasings so the retrieval layer can run parallel
    embedding + BM25 searches and merge results across languages.

    ALWAYS include:
    1. One English query (for embedding/semantic search).
    2. One query in the SAME language/style as the original message (for BM25 keyword match).
    3. One mixed Hinglish query if the conversation context is Hinglish.
    4. One short keyword-only phrase (2-4 words, no pronouns).

    Example: "lag gayi?" after an internship discussion →
        ["internship offer received", "offer letter aa gaya", "intern lag gayi offer",
         "internship got selected", "internship offer"]

    CRITICAL RULES to prevent hallucination:
    - DO NOT answer the question or fill in any facts (company names, dates, locations).
    - DO NOT use proper nouns from the persona name or context unless the USER explicitly
      stated them in their own messages — never infer organisation names from club/society
      names or assume facts not directly asserted by the user.
    - The rewritten queries are for SEARCH only — describe what to look for, not the answer.
    - Keep the person's name if mentioned, but DO NOT assume specific company/place/time values.
    - Bad: "wahan kaisi thi?" → ["How was the experience in Paris?"]
      (fills in a location that was never stated by the user)
    - Good: "wahan kaisi thi?" → ["how was the experience there",
      "wahan kaisi thi experience", "what was it like at that place", "experience there"]
    """
    return (
        "You are rewriting a short follow-up message into 3-5 standalone search query variations.\n"
        "Use the conversation context to understand the TOPIC of the follow-up, "
        "but do NOT fill in facts or answers — only describe what to search for.\n\n"
        "MULTILINGUAL REQUIREMENT:\n"
        "Return a JSON array of 3-5 search queries. ALWAYS include:\n"
        "1. One English query (for embedding/semantic search).\n"
        "2. One query in the SAME language/style as the original message (for BM25 keyword match).\n"
        "3. One mixed Hinglish query if the conversation context is Hinglish.\n"
        "4. One short keyword-only phrase (2-4 words, no pronouns).\n"
        "CRITICAL: If the original message or context is Hinglish, at least 2 of the queries "
        "MUST be in Hinglish/Hindi romanized form so BM25 can match stored Hinglish messages.\n\n"
        "CRITICAL RULES:\n"
        "- DO NOT answer the question or fill in any facts (company names, dates, locations).\n"
        "- DO NOT use proper nouns from the persona name or context unless the USER explicitly stated them.\n"
        "- The queries are used for SEARCH only — describe what to look for, not the answer.\n"
        "- Keep the person's name if mentioned, but DO NOT assume specific company/place/time values.\n"
        "- Bad: 'wahan kaisi thi?' → [\"How was the experience in Paris?\"]\n"
        "  (fills in a location that was never stated by the user)\n"
        "- Good: 'wahan kaisi thi?' → [\"how was the experience there\", "
        "\"wahan kaisa tha\", \"what was it like at that place\", \"experience wahan\"]\n\n"
        "Replace ALL pronouns and vague references with the actual topic from context.\n\n"
        f"Conversation context (last 4 turns):\n{context_block}\n\n"
        f"Follow-up message: {user_message}\n\n"
        'Output ONLY a JSON array of 3-5 search phrases, no markdown fences:\n'
        '["phrase 1", "phrase 2", "phrase 3", "phrase 4"]'
    )


def persona_classify_history_need(context_block: str, user_message: str) -> str:
    """Return the full prompt string for Gemini history-need classification.

    ``context_block`` is up to 6 recent turns formatted as "User: …\\nPersona: …" lines,
    or "(no prior turns)" when the conversation has just started.
    ``user_message`` should already be normalised to lowercase by the caller.

    The model must reply with ONLY valid JSON — no markdown fences.

    ``search_queries`` rationale: chat messages are often stored in a different
    language from the query (e.g. a Hinglish query like "kab aya tha?" may
    match an English stored message like "came back last Tuesday").
    Returning multiple phrasings — original + rephrased + topic keywords —
    lets the retrieval layer run parallel embedding searches and merge results,
    which dramatically improves cross-language recall.

    Context-resolved queries: short follow-ups like "kaha lagi?" are meaningless
    in isolation.  The prompt instructs the model to resolve pronouns and implicit
    references using the recent conversation turns before generating queries —
    e.g. "kaha lagi?" after an internship discussion → ["EY internship location",
    "intern company name", "got internship offer where"].

    ``needs_rewrite`` is true whenever the raw user message contains unresolved
    references (pronouns, deictic words, implicit topics) that would make it a
    poor standalone search query.  Retrieval quality is improved by expanding the
    query before hitting BM25/Chroma.
    """
    return (
        "You are deciding whether a follow-up message in a WhatsApp persona chat requires\n"
        "looking up OLD chat history (messages from weeks/months ago, not this exchange).\n\n"
        "STEP 1 — Understand what the follow-up is really about using the conversation below.\n"
        "  Resolve any pronouns (woh, wahan, uska, there, it) or implicit references\n"
        "  to their actual topic using the recent turns.\n\n"
        "STEP 2 — Decide if answering requires OLD history.\n"
        "  needs_history=false when the answer is already visible in the recent conversation\n"
        "  shown below, OR when it is casual chat with no factual content.\n"
        "  needs_history=true when the question asks about facts, dates, names, places, or\n"
        "  events that are NOT in the current exchange and may only exist in old chat history.\n\n"
        "STEP 3 — Generate FULLY RESOLVED search queries.\n"
        "  Replace ALL pronouns and implicit references with the actual topic from context.\n"
        "  Bad: ['kaha lagi?']  Good: ['internship company name', 'got job offer where',\n"
        "  'got internship offer which company']\n\n"
        "STEP 4 — Decide if the message needs rephrasing for retrieval (needs_rewrite).\n"
        "  Set needs_rewrite=true when:\n"
        "  - The message contains unresolved pronouns or references\n"
        "    (wahan, woh, uska, there, it, that, he, she, they, woh wala, us din).\n"
        "  - The message is a follow-up fragment that only makes sense in context\n"
        "    (e.g. 'kaha?', 'kab?', 'aur wahan ka?', 'aur us baad?').\n"
        "  - The message is short (≤5 words) AND asks about a specific fact.\n"
        "  - Even if longer, it has implicit context references that retrieval cannot resolve.\n"
        "  Set needs_rewrite=false when:\n"
        "  - The message is a complete, self-contained question with no pronouns/references.\n"
        "  - The message is casual or greeting with no fact-seeking intent.\n"
        "  Note: needs_rewrite=true only matters when needs_history=true;\n"
        "  set it false whenever needs_history=false.\n\n"
        "STEP 5 — Classify the temporal direction of the question (query_intent).\n"
        "  Set query_intent based on whether the question is asking about the present,\n"
        "  the past, or is ambiguous:\n"
        '  - "current":    question is about present state, recent events, or current status\n'
        '    (e.g. "lag gayi?", "kya chal raha?", "abhi kya kr rhi?", "kya hua abhi?").\n'
        '  - "historical": question explicitly references the past — includes temporal\n'
        '    markers like "pehle", "tab", "woh wali baat", "us time", "remember when",\n'
        '    "purana", "last time", "kab tha", "kab hua tha", "woh incident".\n'
        '  - "neutral":    ambiguous or asking about general/timeless facts — use this\n'
        "    as the default when the question could apply to any time period.\n"
        "  Note: query_intent is independent of needs_history — even casual memory\n"
        "  questions about current plans should be 'current', not 'historical'.\n\n"
        "Recent conversation (last 6 turns — this IS the current exchange):\n"
        f"{context_block}\n\n"
        f"Latest user message: {user_message}\n\n"
        "Rules:\n"
        "- needs_history=true: question asks about past facts/events/dates/names/places\n"
        "  that are NOT answered by the current exchange above.\n"
        "- needs_history=false: answer is already in the exchange above, OR it is\n"
        "  casual/emotional (reactions, opinions, small talk).\n"
        "- needs_rewrite=true: raw message is a poor standalone search query due to\n"
        "  unresolved pronouns, implicit references, or extreme brevity.\n"
        "- needs_rewrite=false: message is self-contained or needs_history is false.\n"
        '- query_intent: "current" | "historical" | "neutral" — temporal direction of the ask.\n'
        "- search_queries: 2-4 short phrases (2-6 words each), ALL context-resolved.\n"
        "  * First entry: context-resolved English equivalent of what is being asked.\n"
        "  * Include both Hinglish phrasing AND English equivalents when message is Hinglish.\n"
        "  * Never use pronouns or vague words (woh, wahan, there, it) in search_queries.\n"
        "  * Empty list when needs_history=false.\n\n"
        "Return ONLY valid JSON, no markdown fences:\n"
        '{"needs_history": boolean, "needs_rewrite": boolean, '
        '"query_intent": "current"|"historical"|"neutral", "search_queries": ["...", "..."]}'
    )
