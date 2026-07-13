"""
Builds history.jsonl and incoming.jsonl from the Enron Email-Reply Dataset
(Kaggle, oanannv/enron-email-reply-dataset) - 15,377 real email/reply pairs
pulled from the actual Enron corpus, not written for this project.

Cleaning is structural, not judgment-based, since there's no ground truth to
grade rows against: drop anything missing a subject or with a mangled
From/To field (a parser artifact in the source CSV, roughly 1.5% of rows,
where forwarded-email headers leaked into those columns), keep pairs in a
sane word-count range, dedupe on the incoming email text. That leaves about
6k clean pairs out of 15k.

From there we sample 200 into history (what the generator retrieves from)
and hold out 20 for incoming (what we test on), seeded so it's reproducible.
Both are spread across a keyword-based category so the held-out set isn't
accidentally all one kind of email - see categorize() below. That category
is a heuristic for the report breakdown, nothing more; don't read it as a
real taxonomy.

  python data/generate_dataset.py
    pulls the CSV via the Kaggle API (needs pip install kaggle and
    ~/.kaggle/kaggle.json), caches it under data/raw/ (gitignored). The two
    output files are committed, so this step is a one-time thing - nobody
    downstream needs Kaggle credentials just to run the pipeline.

  python data/generate_dataset.py --csv /path/to/EnronEmailReplyPairsWithContext.csv
    skip the download, build from a CSV you already have.
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
RAW_CSV = RAW_DIR / "EnronEmailReplyPairsWithContext.csv"
HISTORY_OUT = DATA_DIR / "history.jsonl"
INCOMING_OUT = DATA_DIR / "incoming.jsonl"

KAGGLE_DATASET = "oanannv/enron-email-reply-dataset"

SEED = 42
HISTORY_N = 200
INCOMING_N = 20
INCOMING_PER_CATEGORY_CAP = 4  # keep the held-out set from being one category

SEND_WORDS = (20, 150)
REPLY_WORDS = (10, 120)

_BAD_HEADER_LEAK = re.compile(r"\n|Sent:|Subject:|To:", re.I)


def download_csv() -> Path:
    if RAW_CSV.exists():
        return RAW_CSV
    RAW_DIR.mkdir(exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "Need the source CSV. Either:\n"
            "  pip install kaggle   (then re-run; needs ~/.kaggle/kaggle.json)\n"
            "or download it yourself from\n"
            f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and pass --csv /path/to/EnronEmailReplyPairsWithContext.csv"
        )
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
    if not RAW_CSV.exists():
        sys.exit(f"Download finished but {RAW_CSV} not found; check {RAW_DIR}")
    return RAW_CSV


def _word_count(s: str) -> int:
    return len(s.split())


def is_clean(row: dict) -> bool:
    send, reply = row["EmailSend"], row["EmailReply"]
    subj, frm, to = row["SubjectSend"], row["From"], row["To"]
    if not subj or subj == "nan":
        return False
    if not frm or _BAD_HEADER_LEAK.search(frm) or not (3 <= len(frm) <= 60):
        return False
    if to and _BAD_HEADER_LEAK.search(to):
        return False
    if not (SEND_WORDS[0] <= _word_count(send) <= SEND_WORDS[1]):
        return False
    if not (REPLY_WORDS[0] <= _word_count(reply) <= REPLY_WORDS[1]):
        return False
    if send.strip() == reply.strip():
        return False
    return True


# keyword buckets, just for the score_by_category breakdown in src/evaluate.py
CATEGORY_RULES = [
    ("scheduling", r"\b(meeting|schedule|calendar|lunch|call at|available|conference room|next week|this week works)\b"),
    ("approval_review", r"\b(approve|approval|review|sign off|comments on|attached|draft|redline)\b"),
    ("hr_recruiting", r"\b(intern|candidate|interview|resume|résumé|hire|recruit|offer letter)\b"),
    ("status_update", r"\b(update|fyi|status|progress|heads up|as discussed)\b"),
    ("negotiation_deal", r"\b(agree|propose|contract|deal|counterpart|isda|term sheet|pricing)\b"),
    ("request_info", r"\?|(\bcan you\b|\bcould you\b|\bplease send\b|\blet me know\b)"),
    ("personal_social", r"\b(how are you|great to hear|thanks for|dinner|golf|party|weekend|vacation)\b"),
]


def categorize(subject: str, body: str) -> str:
    text = f"{subject} {body}".lower()
    for name, pattern in CATEGORY_RULES:
        if re.search(pattern, text):
            return name
    return "other"


def load_rows(csv_path: Path) -> list:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if is_clean(row)]
    # de-dup on the incoming email text, keep first occurrence
    seen = set()
    deduped = []
    for row in rows:
        key = row["EmailSend"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def to_ticket(row: dict, idx: int, category: str, prefix: str) -> dict:
    context = row.get("Context") or ""
    context = context.strip()
    if not context or context == "nan":
        context = "(no prior message in this thread)"
    return {
        "id": f"{prefix}-{idx:04d}",
        "category": category,
        "thread_type": "reply_in_thread" if context != "(no prior message in this thread)" else "opening_message",
        "subject": row["SubjectSend"].strip(),
        "incoming_email": row["EmailSend"].strip(),
        "context": context,
        "reference_reply": row["EmailReply"].strip(),
        "from_email": row.get("From", "").strip(),
        "to_email": row.get("To", "").strip(),
        "date_sent": row.get("DateSend", "").strip(),
    }


def sample(rows: list) -> tuple:
    rng = random.Random(SEED)
    rng.shuffle(rows)

    by_cat = defaultdict(list)
    for row in rows:
        cat = categorize(row["SubjectSend"], row["EmailSend"])
        by_cat[cat].append((row, cat))

    # one from each category per pass, so the 20-item held-out set doesn't
    # end up mostly one email type
    incoming, used = [], set()
    cat_counts = defaultdict(int)
    cat_iters = {cat: iter(items) for cat, items in by_cat.items()}
    while len(incoming) < INCOMING_N and cat_iters:
        for cat in list(cat_iters.keys()):
            if len(incoming) >= INCOMING_N:
                break
            if cat_counts[cat] >= INCOMING_PER_CATEGORY_CAP:
                del cat_iters[cat]
                continue
            row = next(cat_iters[cat], None)
            if row is None:
                del cat_iters[cat]
                continue
            row, c = row
            key = row["EmailSend"].strip()
            if key in used:
                continue
            incoming.append((row, c))
            used.add(key)
            cat_counts[c] += 1
    # top up if categories were too thin to hit INCOMING_N
    if len(incoming) < INCOMING_N:
        for row in rows:
            if len(incoming) >= INCOMING_N:
                break
            key = row["EmailSend"].strip()
            if key in used:
                continue
            incoming.append((row, categorize(row["SubjectSend"], row["EmailSend"])))
            used.add(key)

    history = []
    for row in rows:
        if len(history) >= HISTORY_N:
            break
        key = row["EmailSend"].strip()
        if key in used:
            continue
        history.append((row, categorize(row["SubjectSend"], row["EmailSend"])))
        used.add(key)

    return history, incoming


def main():
    global HISTORY_N, INCOMING_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None,
                     help="Use an already-downloaded copy instead of fetching via Kaggle API")
    ap.add_argument("--history-n", type=int, default=HISTORY_N)
    ap.add_argument("--incoming-n", type=int, default=INCOMING_N)
    args = ap.parse_args()

    HISTORY_N, INCOMING_N = args.history_n, args.incoming_n

    csv_path = args.csv or download_csv()
    rows = load_rows(csv_path)
    print(f"Loaded {len(rows)} clean, de-duplicated candidate pairs "
          f"from {csv_path.name}")

    history_rows, incoming_rows = sample(rows)

    history = [to_ticket(row, i, cat, "hist") for i, (row, cat) in enumerate(history_rows, 1)]
    incoming = [to_ticket(row, i, cat, "in") for i, (row, cat) in enumerate(incoming_rows, 1)]

    with HISTORY_OUT.open("w") as f:
        for r in history:
            f.write(json.dumps(r) + "\n")
    with INCOMING_OUT.open("w") as f:
        for r in incoming:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(history)} past pairs to {HISTORY_OUT} (learned from)")
    print(f"Wrote {len(incoming)} held-out incoming emails to {INCOMING_OUT} (tested on)")
    hist_cats = defaultdict(int)
    for _, c in history_rows:
        hist_cats[c] += 1
    inc_cats = defaultdict(int)
    for _, c in incoming_rows:
        inc_cats[c] += 1
    print("History category mix:", dict(hist_cats))
    print("Incoming category mix:", dict(inc_cats))


if __name__ == "__main__":
    main()
