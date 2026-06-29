"""Persona activation / build prompts — personality, writing style, chat analysis, listening style.

All prompts are called at build time (persona_train graph → workspace.py refresh_person_* functions)
and use the shared rate limiter. Each function returns a single string passed directly to
``gemini_service.chat([{"role": "user", "content": prompt}], ...)``.

Training data format
--------------------
Exchanges are presented in the format::

    [Other person]: <their message>
    [Name]: <Name's response>

Standalone messages (no immediately preceding other-sender message) are::

    [Name]: <message>

This gives the LLM the stimulus-response context needed to analyse the *relational dynamic*,
not just the target person's isolated outgoing messages.
"""

from __future__ import annotations


def persona_extract_personality(name: str, samples: str) -> str:
    """Personality keynote extraction prompt — 3–6 sentences in third person.

    ``samples`` is a newline-separated collection of conversational exchanges in the
    format ``[Other]: message / [Name]: response``.  Standalone messages appear as
    ``[Name]: message``.
    Low temperature (0.4) for factual, grounded extraction.
    """
    return (
        f"You are analysing a WhatsApp conversation involving {name}.\n"
        f"The data below shows conversational exchanges: each exchange starts with what the "
        f"other person wrote, then shows {name}'s response. Some entries show only {name}'s "
        f"standalone messages (no preceding context).\n\n"
        f"Based only on these exchanges, write a concise personality profile in 3–6 sentences.\n"
        f"Cover: communication style, vocabulary and slang, humour, recurring themes or interests, "
        f"emotional tone, and — importantly — how {name} RELATES TO and responds to the other person "
        f"(does their tone change based on what the other person says? how do they engage?)\n"
        f"Be specific and factual — ground every claim in the exchanges. Write in third person.\n\n"
        f"Exchanges:\n{samples}"
    )


def persona_extract_writing_style(name: str, samples: str) -> str:
    """Writing-style (surface typing patterns) extraction prompt — 3–5 plain sentences.

    Captures HOW they type — casing, punctuation, abbreviations, emoji, sentence structure.
    ``samples`` is a collection of conversational exchanges; the prompt instructs the LLM
    to focus only on ``[Name]:`` lines for surface-pattern analysis.
    Very low temperature (0.3) for factual style extraction.
    """
    return (
        f"You are analysing WhatsApp messages sent by {name}.\n"
        f"The data below shows conversational exchanges in the format [Sender]: message. "
        f"Focus ONLY on {name}'s messages — lines that start with [{name}]:.\n\n"
        f"Based on {name}'s messages, describe their writing style in 3–5 plain sentences.\n"
        f"Focus on HOW they type — NOT on personality or topics. Cover these points:\n"
        f"1. Capitalisation: do they capitalise sentence-starts? ALL CAPS? Mostly lowercase?\n"
        f"2. Punctuation: do they use periods, question marks, commas — or skip them?\n"
        f"3. Abbreviations and shorthand they actually use (list the specific ones you see).\n"
        f"4. Emoji: do they use them? How frequently? Which ones appear most?\n"
        f"5. Sentence structure: short fragments vs complete sentences; terse or verbose?\n"
        f"Be specific. Quote or paraphrase actual examples from the messages where helpful.\n"
        f"Do NOT infer mood, personality, or intent — only describe surface typing patterns.\n\n"
        f"Exchanges:\n{samples}"
    )


def persona_extract_chat_analysis(
    name: str,
    chunk: str,
    chunk_num: int,
    total_chunks: int,
) -> str:
    """Per-chunk deep analysis prompt — 3–5 bullet-point observations.

    ``chunk`` is a collection of conversational exchanges in ``[Sender]: message`` format
    for this segment.  ``chunk_num`` is 1-indexed; ``total_chunks`` is the total number of
    chunks.  Low temperature (0.3) for factual, grounded observations.
    """
    return (
        f"Analyse these WhatsApp conversational exchanges involving {name}.\n"
        f"The format is: [Other person]: their message, then [{name}]: their response. "
        f"Some entries show only [{name}]'s standalone messages.\n"
        f"This is chunk {chunk_num} of {total_chunks} (newest messages listed first).\n\n"
        f"Write 3–5 specific bullet-point observations about:\n"
        f"  \u2022 vocabulary patterns and recurring phrases\n"
        f"  \u2022 recurring topics or interests\n"
        f"  \u2022 emotional tone and patterns\n"
        f"  \u2022 relationship dynamics: how {name} responds differently to different message types\n"
        f"  \u2022 stimulus-response patterns (e.g. 'when the other person sends X, {name} responds with Y')\n"
        f"  \u2022 any time-of-day or frequency patterns\n"
        f"Ground every observation in the exchanges. Be specific and factual.\n\n"
        f"Exchanges:\n{chunk}"
    )


def persona_extract_chat_analysis_consolidate(
    name: str,
    analyses: str,
    num_chunks: int,
) -> str:
    """Consolidation prompt — synthesises per-chunk observations into 5–10 sentences.

    ``analyses`` is all chunk observations joined by "\\n\\n---\\n\\n".
    ``num_chunks`` is the number of analysis chunks that produced ``analyses``.
    Low temperature (0.3) for factual synthesis.
    """
    return (
        f"You have structured observations about {name}'s WhatsApp messaging "
        f"patterns from {num_chunks} analysis chunk(s).\n"
        f"Synthesise these into a coherent chat-pattern analysis of 5\u201310 sentences covering:\n"
        f"vocabulary patterns, recurring topics, emotional patterns, relationship dynamics, "
        f"stimulus-response patterns (how {name}'s tone or style shifts based on what the other "
        f"person sends), and any notable time-of-day or activity patterns.\n"
        f"Write in third person. Be specific \u2014 ground every claim in the observations.\n\n"
        f"Observations:\n{analyses}"
    )


def persona_extract_relationship_emotional(name: str, exchanges: str) -> str:
    """Gemini Call 1 — relationship dynamic + emotional profile as a single JSON object.

    Returns JSON: {"relationshipDynamic": "...", "emotionalProfile": "..."}
    Use paired exchanges so the LLM sees the relational stimulus-response context.
    Includes temporal depth: note how the dynamic or traits have shifted over time.
    Low temperature (0.3) for factual, grounded extraction.
    """
    return (
        f"You are analysing a WhatsApp conversation involving {name}.\n"
        f"The data below shows conversational exchanges in [Sender]: message format.\n"
        f"The exchanges are ordered from OLDEST to NEWEST — pay attention to how the "
        f"dynamic and tone shift across the timeline.\n\n"
        f"Return ONLY a JSON object with exactly two keys:\n\n"
        f"\"relationshipDynamic\": 4-6 sentences describing the specific dyad dynamic between "
        f"{name} and their conversation partner(s). Cover: role (mentor/peer/junior), power "
        f"balance, how tone shifts based on context, emotional dependency patterns, trust level. "
        f"IMPORTANT — also note if the relationship has changed over time: was it more formal "
        f"earlier and now casual? Did the power balance shift? Include one sentence about "
        f"how {name} has changed or grown across the timeline if it is detectable. "
        f"Write in third person. Be specific and ground every claim in the exchanges.\n\n"
        f"\"emotionalProfile\": 3-4 sentences describing how {name} handles stress, offers "
        f"support, and reacts to conflict. Note any RECURRING personal traits that appear "
        f"consistently across the timeline (e.g. commitment patterns, stress responses, "
        f"emotional availability). Keep factual — grounded in what the exchanges actually show.\n\n"
        f"Return ONLY valid JSON — no markdown, no code fences, no explanation.\n\n"
        f"Exchanges:\n{exchanges}"
    )


def persona_extract_typing_fingerprint(name: str, solo_messages: str) -> str:
    """Gemini Call 2 — structured typing fingerprint as JSON.

    ``solo_messages`` is a newline-separated list of only the target person's messages
    (no paired context needed — this is purely surface-pattern extraction).
    Returns a JSON object with keys: capsStyle, abbreviations, emojis, punctuation,
    emphasisStyle, avgMessageLength.
    Very low temperature (0.2) for factual, observational extraction.
    """
    return (
        f"You are analysing messages sent by {name} in a WhatsApp conversation.\n"
        f"Below are their messages, one per line.\n\n"
        f"Extract their exact typing patterns and return ONLY a JSON object with these keys:\n\n"
        f"\"capsStyle\": one of \"mostly_lowercase\", \"mixed\", \"ALL_CAPS_SOMETIMES\" — "
        f"pick the one that best fits what you actually observe.\n\n"
        f"\"abbreviations\": array of objects {{\"from\": \"abbrev\", \"to\": \"full\"}} — "
        f"extract abbreviations you actually see in these messages.\n\n"
        f"CRITICAL RULES for abbreviations:\n"
        f"- ONLY include informal personal shorthand unique to THIS person's Hinglish typing style.\n"
        f"- EXCLUDE standard technical acronyms (UI, UX, API, HTML, CSS, etc.).\n"
        f"- EXCLUDE universally-known English abbreviations (ok, lol, btw, omg, etc.).\n"
        f"- EXCLUDE correctly-spelled English or Hindi words — only abbreviations count.\n"
        f"- INCLUDE Hinglish phonetic shorthand where vowels are dropped: "
        f"nhi=nahi, kr=kar, yr=yaar, bt=baat, bta=bata, fst=fast, kru=karunga, h=hai, "
        f"sb=sab, pr=par, bje=baje, rkh=rakh, grp=group, and similar patterns.\n"
        f"- INCLUDE elongated emphasis forms this person specifically uses "
        f"(e.g. fstttt, noiceee, bhaiii).\n"
        f"- Maximum 10 abbreviations — only the most distinctive and frequent ones.\n"
        f"- If fewer than 3 genuine Hinglish abbreviations are found, "
        f"return an empty list [] rather than padding with technical terms.\n\n"
        f"\"emojis\": array of emoji characters that appear in the messages (max 8). "
        f"List only what you actually observe.\n\n"
        f"\"punctuation\": one short sentence describing their punctuation habits "
        f"(periods, question marks, commas — how often they appear or are skipped).\n\n"
        f"\"emphasisStyle\": describe elongation or caps-emphasis patterns you observe "
        f"(e.g. 'elongation: fstttt, noiceee, bhaiii'). Write 'none' if not observed.\n\n"
        f"\"avgMessageLength\": integer — estimated average character count per message "
        f"based on what you see.\n\n"
        f"Return ONLY valid JSON — no markdown, no code fences, no explanation. "
        f"Only include patterns you actually observe in the messages.\n\n"
        f"Messages:\n{solo_messages}"
    )


def persona_extract_response_patterns_topic_map(name: str, exchanges: str) -> str:
    """Gemini Call 3 — response patterns + topic map as a single JSON object.

    Returns JSON: {"responsePatterns": "...", "topicMap": "..."}
    Use paired exchanges so the LLM sees the stimulus-response context.
    Low temperature (0.3) for factual, grounded extraction.
    """
    return (
        f"You are analysing a WhatsApp conversation involving {name}.\n"
        f"The data below shows conversational exchanges in [Sender]: message format.\n\n"
        f"Return ONLY a JSON object with exactly two keys:\n\n"
        f"\"responsePatterns\": exactly 3-5 behavioral stimulus→response patterns as a single "
        f"string. Each pattern on its own line, starting with '• '. Format each as: "
        f"'When [X], {name} [does Y].' Max 15 words per bullet. "
        f"Examples: '• When stressed about deadlines, sends rapid short fragments and abbreviations.' "
        f"'• When asked something uncertain, deflects with urgency rather than admitting uncertainty.' "
        f"Ground every pattern in what the exchanges actually show.\n\n"
        f"\"topicMap\": 2-4 sentences describing recurring subjects and how {name} engages with "
        f"each — which topics they drive vs. deflect, which trigger emotional shifts. "
        f"Write in third person.\n\n"
        f"Return ONLY valid JSON — no markdown, no code fences, no explanation.\n\n"
        f"Exchanges:\n{exchanges}"
    )


def persona_label_voice_samples(person_name: str, exchanges: str) -> str:
    """Prompt to assign a 2–4 word context label to each voice sample exchange.

    ``exchanges`` is a numbered list of exchange blocks (Exchange 1:, Exchange 2:, …).
    Returns a JSON array of label strings in the same order — one string per exchange.
    Low temperature (0.3) keeps labels grounded and non-generic.
    """
    short_name = person_name.split(",")[0].strip()
    return (
        f"You are reviewing WhatsApp exchanges involving {short_name}.\n"
        f"For each numbered exchange below, write a 2–4 word label describing what "
        f"{short_name} is doing in that specific exchange.\n"
        f"Examples: 'deflecting a concern', 'driving task urgency', 'offering emotional support', "
        f"'casual banter', 'setting a boundary', 'coordinating logistics', 'expressing frustration', "
        f"'making a decision', 'sharing an update'.\n\n"
        f"Rules:\n"
        f"- Labels must be grounded in what the exchanges actually show.\n"
        f"- Do not use generic labels like 'speaking' or 'responding'.\n"
        f"- Each label should describe the ROLE or ACTION, not just the topic.\n\n"
        f"Return ONLY a JSON array of label strings in the same order as the exchanges — "
        f"one string per exchange. No markdown, no code fences, no explanation.\n\n"
        f"Exchanges:\n{exchanges}"
    )


def persona_extract_listening_style(name: str, samples: str) -> str:
    """Active-listening style extraction prompt — 3–5 sentences, behaviour-specific.

    Captures reactive patterns (how they respond when others share problems/news).
    ``samples`` is a collection of conversational exchanges in ``[Sender]: message`` format,
    giving the LLM the actual stimulus (other person's message) alongside the response.
    Low temperature (0.3) for factual, grounded extraction.
    """
    return (
        f"Analyse how {name} listens and responds when others share problems, "
        f"emotions, news, or updates in this WhatsApp conversation.\n\n"
        f"The data shows conversational exchanges in the format:\n"
        f"  [Other person]: their message\n"
        f"  [{name}]: their response\n"
        f"Use the other person's messages as context to understand WHAT {name} is responding to.\n\n"
        f"Focus on their SPECIFIC behavioral patterns \u2014 NOT generic empathy theory:\n"
        f"- Do they ask follow-up questions? If so, what style? (sharp/curious/minimal/probing)\n"
        f"- Do they validate feelings or jump straight to advice/action?\n"
        f'- What filler or reaction words do they use? (e.g. "bhai sach mein?", "mkc", "arey", "bc yaar")\n'
        f"- How quickly do they redirect vs sit with the topic?\n"
        f"- Do they share their own experience in return, or stay focused on the other person?\n"
        f"- Any recurring phrases when someone vents to them?\n"
        f"- Does their response length or tone change based on how long or emotional the other "
        f"  person's message is? (e.g. long concerned message → brief reassurance, or vice versa)\n\n"
        f"Write 3-5 sentences. Be specific to this person's actual patterns. "
        f'Do NOT use generic phrases like "they are empathetic" or "they are a good listener". '
        f"Extract only what the exchanges actually show.\n\n"
        f"Exchanges:\n{samples}"
    )
