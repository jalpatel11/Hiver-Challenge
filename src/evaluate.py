"""
Combines the rubric judge with a handful of hard-coded guardrails, then
rolls per-response scores into an aggregate.

Guardrails exist because some failures shouldn't depend on the judge's mood:
an empty reply, one that leaks our own prompt labels back, or a bare "hi"
isn't 80% good no matter how the rubric scores it. Each failed check costs
20 points off the quality score.
"""
import re

from src.judge import AXES, judge_reply

# model echoing our own prompt labels back instead of writing an actual reply
LEAK_MARKERS = [
    "prior message in this thread", "incoming email", "draft reply",
    "reference reply", "past email for style", "past example",
]


def guardrails(ticket: dict, draft: str) -> dict:
    """Deterministic checks. Each returns True = pass."""
    body = draft.strip()
    low = body.lower()
    checks = {
        "non_empty": len(body) > 0,
        "substantive": len(body.split()) >= 10,
        "no_context_leak": not any(m in low for m in LEAK_MARKERS),
        "no_placeholder": not re.search(r"\[(insert|todo|name|xxx)\b", low),
        "not_just_greeting": not re.fullmatch(
            r"(hi|hello|hey)[!,. ]*", low
        ),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def score_response(ticket: dict, draft: str, mock: bool = False) -> dict:
    gr = guardrails(ticket, draft)
    rubric = judge_reply(ticket, draft, mock=mock)

    axis_scores = {ax: rubric[ax]["score"] for ax in AXES}
    quality = sum(axis_scores.values()) / len(AXES)  # 1..5
    quality_100 = (quality - 1) / 4 * 100  # rescale 1-5 -> 0-100

    # guardrail penalty: each failed hard gate removes 20 points, floored at 0
    failed = sum(1 for k, v in gr.items() if k != "all_passed" and not v)
    penalty = min(quality_100, failed * 20)
    final = round(max(0.0, quality_100 - penalty), 1)

    return {
        "id": ticket["id"],
        "category": ticket["category"],
        "thread_type": ticket.get("thread_type"),
        "axis_scores": axis_scores,
        "rubric": rubric,
        "guardrails": gr,
        "quality_100": round(quality_100, 1),
        "guardrail_penalty": penalty,
        "final_score": final,
    }


def aggregate(rows: list) -> dict:
    if not rows:
        return {}
    n = len(rows)
    overall = round(sum(r["final_score"] for r in rows) / n, 1)
    per_axis = {}
    for ax in AXES:
        per_axis[ax] = round(sum(r["axis_scores"][ax] for r in rows) / n, 2)
    guardrail_pass_rate = round(
        sum(1 for r in rows if r["guardrails"]["all_passed"]) / n * 100, 1
    )
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["final_score"])
    by_cat = {k: round(sum(v) / len(v), 1) for k, v in by_cat.items()}
    return {
        "n": n,
        "overall_score": overall,
        "per_axis_avg_1to5": per_axis,
        "guardrail_pass_rate": guardrail_pass_rate,
        "score_by_category": by_cat,
    }
