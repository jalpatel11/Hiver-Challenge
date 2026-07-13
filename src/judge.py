"""
Rubric judge, five axes, 1-5 each: resolution, grounding, completeness,
tone, clarity.

Grounding used to mean "matches the internal CRM context" back when this
used a hand-authored support-ticket dataset. Enron threads don't have that,
so grounding here just checks the draft against the thread itself: no
invented names, numbers, dates, or commitments that aren't actually there.
Arguably a more honest hallucination check anyway, since a real assistant
usually only has the thread to go on too.

Judge sees the real reply as one example of a good answer, not the only
correct one, and is told not to reward matching it word for word. It's a
real reply someone actually sent, not a hand-picked perfect one, so it
won't always read as flawless either.

Temperature 0 so scores don't drift between runs. Falls back to a heuristic
if the judge returns bad JSON instead of crashing the run.
"""
import json
import re

from src import llm

AXES = ["resolution", "grounding", "completeness", "tone", "clarity"]

JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of suggested email replies. You grade "
    "the DRAFT reply, using the thread (the incoming email and its prior "
    "message, if any) and a reference reply as ground truth for what is "
    "correct and complete.\n\n"
    "Score each axis from 1 (poor) to 5 (excellent):\n"
    "- resolution: Does the draft actually address the sender's core ask? A "
    "reply that dodges, stalls, or answers the wrong thing scores low.\n"
    "- grounding: Does every specific claim in the draft (a name, number, "
    "date, cause, or commitment) actually trace back to the incoming email or "
    "its prior thread context? Inventing a specific that isn't in the thread, "
    "or asserting something the thread contradicts, scores 1-2 even if it "
    "sounds plausible and helpful. A draft that correctly asks for missing "
    "specifics instead of guessing scores high here.\n"
    "- completeness: Are ALL distinct questions and requests handled? Missing "
    "one of several questions caps this axis low.\n"
    "- tone: Appropriate to the situation and consistent with how this "
    "correspondence actually reads (e.g. terse and direct where that fits, "
    "warmer where that fits) rather than generically formal.\n"
    "- clarity: Clear, concise, easy to act on. Rambling or confusing scores low.\n\n"
    "Important:\n"
    "- The reference reply is ONE example of a good answer, and it is a REAL "
    "reply that was actually sent, not a hand-polished ideal. Reward correct "
    "alternative solutions equally. Do NOT reward mere similarity to the "
    "reference, and do NOT penalize a draft for being worded differently or "
    "for being clearer/more complete than the reference.\n\n"
    "Return ONLY a JSON object of this exact shape:\n"
    "{\"resolution\": {\"score\": int, \"why\": str}, "
    "\"grounding\": {\"score\": int, \"why\": str}, "
    "\"completeness\": {\"score\": int, \"why\": str}, "
    "\"tone\": {\"score\": int, \"why\": str}, "
    "\"clarity\": {\"score\": int, \"why\": str}}"
)

JUDGE_USER_TMPL = (
    "PRIOR MESSAGE IN THIS THREAD (if any):\n{context}\n\n"
    "INCOMING EMAIL\nSubject: {subject}\n\n{body}\n\n"
    "REFERENCE REPLY (one example of a good answer, actually sent):\n{reference}\n\n"
    "DRAFT REPLY TO GRADE:\n{draft}\n\n"
    "Score the DRAFT. Return only the JSON."
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def _mock_judge(ticket: dict, draft: str) -> dict:
    """Heuristic judge for offline runs. Crude but monotonic: rewards length,
    lexical overlap with the reference, and answering, so real replies beat the
    generic mock reply beat the empty floor. Not the real metric, just keeps the
    pipeline runnable without a key."""
    ref = ticket["reference_reply"].lower()
    d = draft.lower()
    ref_tokens = set(re.findall(r"[a-z0-9]+", ref))
    d_tokens = set(re.findall(r"[a-z0-9]+", d))
    overlap = len(ref_tokens & d_tokens) / max(1, len(ref_tokens))
    length_ok = 1.0 if len(draft.split()) >= 25 else 0.4
    base = 1 + round(4 * overlap * length_ok)
    base = max(1, min(5, base))
    out = {}
    for ax in AXES:
        out[ax] = {"score": base, "why": f"mock heuristic: overlap={overlap:.2f}"}
    return out


def judge_reply(ticket: dict, draft: str, mock: bool = False) -> dict:
    if mock or not llm.have_key():
        return _mock_judge(ticket, draft)
    user = JUDGE_USER_TMPL.format(
        context=ticket.get("context", "(none)"),
        subject=ticket.get("subject", ""),
        body=ticket["incoming_email"],
        reference=ticket["reference_reply"],
        draft=draft,
    )
    raw = llm.complete(JUDGE_SYSTEM, user, model=llm.JUDGE_MODEL,
                       max_tokens=700, temperature=0.0)
    try:
        parsed = _extract_json(raw)
        # normalize + clamp
        for ax in AXES:
            s = int(parsed[ax]["score"])
            parsed[ax]["score"] = max(1, min(5, s))
        return parsed
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # if the judge misbehaves, fail safe to the heuristic rather than crash
        fallback = _mock_judge(ticket, draft)
        for ax in AXES:
            fallback[ax]["why"] = "judge parse failed; heuristic fallback"
        return fallback
