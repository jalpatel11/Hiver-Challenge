"""
End-to-end runner.

  python run.py            # real Claude for generation + judging (needs API key)
  python run.py --mock     # fully offline, deterministic, zero setup
  python run.py --limit 3  # first N incoming emails only (fast smoke test)
  python run.py --k 3      # number of retrieved past examples for grounding

For each held-out email (incoming.jsonl) we pull neighbors from history.jsonl,
generate a reply, and score three versions with the same judge: a degenerate
floor reply, our model output, and the real reply as reference. Scoring floor
and reference alongside the model is the actual check that the metric works -
if it doesn't rank reference >= model >> floor, the model's score can't be
trusted either, so that comparison prints on every run instead of living in
a one-off notebook somewhere.
"""
import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.evaluate import aggregate, score_response
from src.generate import generate_reply
from src.retrieve import TfidfRetriever

ROOT = Path(__file__).parent
HISTORY = ROOT / "data" / "history.jsonl"
INCOMING = ROOT / "data" / "incoming.jsonl"
RESULTS = ROOT / "results"
FLOOR_REPLY = "Thanks for reaching out. We'll get back to you."


def load(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run(mock, limit, k):
    history = load(HISTORY)
    incoming = load(INCOMING)
    if limit:
        incoming = incoming[:limit]
    retriever = TfidfRetriever(history)
    RESULTS.mkdir(exist_ok=True)

    systems = {"floor": [], "model": [], "reference": []}
    per_response = []

    for t in incoming:
        model_reply, neighbor_ids = generate_reply(t, retriever, k=k, mock=mock)
        replies = {
            "floor": FLOOR_REPLY,
            "model": model_reply,
            "reference": t["reference_reply"],
        }
        row = {"id": t["id"], "subject": t.get("subject"),
               "retrieved_from": neighbor_ids, "replies": {}, "scores": {}}
        for name, reply in replies.items():
            scored = score_response(t, reply, mock=mock)
            systems[name].append(scored)
            row["replies"][name] = reply
            row["scores"][name] = scored["final_score"]
        row["model_detail"] = next(s for s in systems["model"] if s["id"] == t["id"])
        per_response.append(row)
        print(f"[{t['id']:<20}] retrieved={neighbor_ids}  "
              f"floor={row['scores']['floor']:>5}  model={row['scores']['model']:>5}  "
              f"ref={row['scores']['reference']:>5}")

    summary = {name: aggregate(rows) for name, rows in systems.items()}
    (RESULTS / "scored.jsonl").write_text("\n".join(json.dumps(r) for r in per_response))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, per_response, mock, k)
    print_summary(summary)


def print_summary(summary):
    print("\n=== OVERALL ===")
    for name in ["floor", "model", "reference"]:
        s = summary[name]
        print(f"{name:<10} overall={s['overall_score']:>5}/100   "
              f"guardrail_pass={s['guardrail_pass_rate']}%")
    print("\nModel per-axis (1-5):", summary["model"]["per_axis_avg_1to5"])


def write_report(summary, per_response, mock, k):
    m, f, r = summary["model"], summary["floor"], summary["reference"]
    L = []
    L.append("# Evaluation report\n")
    L.append(f"Mode: {'MOCK (offline heuristic)' if mock else 'Claude'}  |  "
             f"Incoming emails: {m['n']}  |  Retrieved examples per email: {k}\n")
    L.append("## Overall scores (0-100)\n")
    L.append("| System | Overall | Guardrail pass |")
    L.append("|---|---|---|")
    L.append(f"| Reference (human ceiling) | {r['overall_score']} | {r['guardrail_pass_rate']}% |")
    L.append(f"| Model (retrieval-grounded) | {m['overall_score']} | {m['guardrail_pass_rate']}% |")
    L.append(f"| Floor (degenerate) | {f['overall_score']} | {f['guardrail_pass_rate']}% |")
    L.append("")
    L.append("A valid metric must rank reference >= model >> floor. "
             f"Here: {r['overall_score']} >= {m['overall_score']} >> {f['overall_score']}.\n")
    L.append("## Model per-axis average (1-5)\n")
    L.append("| Axis | Avg |")
    L.append("|---|---|")
    for ax, v in m["per_axis_avg_1to5"].items():
        L.append(f"| {ax} | {v} |")
    L.append("")
    L.append("## Model score by category\n")
    L.append("| Category | Score |")
    L.append("|---|---|")
    for c, v in m["score_by_category"].items():
        L.append(f"| {c} | {v} |")
    L.append("")
    L.append("## Per-response (model)\n")
    L.append("| Incoming | Retrieved from | Score | Guardrails |")
    L.append("|---|---|---|---|")
    for row in per_response:
        d = row["model_detail"]
        gp = "pass" if d["guardrails"]["all_passed"] else "FAIL"
        L.append(f"| {row['id']} | {', '.join(row['retrieved_from'])} | {d['final_score']} | {gp} |")
    (RESULTS / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="offline deterministic run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3, help="retrieved examples per email")
    args = ap.parse_args()
    run(mock=args.mock, limit=args.limit, k=args.k)
