# AI email suggested-response system

Given a new incoming email, this drafts a suggested reply by learning from a
set of past emails and the replies that were actually sent to them, then
scores how good that draft is and explains why. The scoring half is the
point of this exercise, so most of this README is spent on it: what
"accurate" means for a reply, why the metric I picked is the right one, and
how I checked that it actually tracks quality instead of just producing a
number that looks plausible.

It runs end to end either way. With an `ANTHROPIC_API_KEY` set, generation
and judging both go through Claude. Without one, `python run.py --mock`
runs the same pipeline offline with a deterministic stand-in, so anyone
grading this can confirm it works in about a second without spending
anything.

## Running it

```bash
pip install -r requirements.txt

# data/history.jsonl and data/incoming.jsonl are already committed, built
# from the Enron Email-Reply Dataset (see below). To rebuild them from
# scratch you'd need a Kaggle account:
#   pip install kaggle && python data/generate_dataset.py

python run.py --mock                # offline, zero setup, proves the pipeline
cp .env.example .env                # then add your ANTHROPIC_API_KEY for a real run
python run.py                       # retrieval-grounded generation + LLM judge
python run.py --k 4                 # tune retrieved examples per email
python run.py --limit 3             # first 3 incoming emails only, cheap smoke test
```

Results land in `results/`: `report.md` is the human-readable summary,
`summary.json` has the overall/per-axis/per-category numbers, and
`scored.jsonl` has every reply for every system with full rubric rationales
and which past emails grounded each one.

## 1. The dataset

**Source.** [Enron Email-Reply Dataset](https://www.kaggle.com/datasets/oanannv/enron-email-reply-dataset)
on Kaggle (Oana-Mariana Ilie, 2025, MIT license): 15,377 real email/reply
pairs pulled from the Enron corpus, which FERC put into the public domain
in 2002 and which has been re-processed for NLP research ever since. Each
row is a message someone actually sent and the reply someone actually sent
back, with the prior message in the thread when there was one.

I'd originally hand-authored a small dataset for this repo: twelve support
tickets with invented account details and reference replies I wrote myself.
That gives you clean ground truth, since you made up the facts, so you know
exactly what "correct" looks like. But it's fiction. Enron trades that clean
ground truth for something a "learn from past replies" system actually
needs proof of: what real replies from real people, to real emails, look
like. That's a stronger claim about representativeness for the generation
half of the task. It costs something on the evaluation half too, covered in
section 3, since there's no internal database of facts to check a reply
against anymore. Grounding has to be checked against the thread itself
instead.

**Processing** (`data/generate_dataset.py`, standard library only, no
pandas). The raw CSV needed cleaning that's structural rather than judgment
calls, since there's no ground truth to grade a row against:

- drop rows with a missing subject, or a mangled `From`/`To` field. About
  1.5% of rows have forwarded-email header text leaked into those columns,
  a parsing artifact in the source data rather than anything I introduced.
- keep only pairs in a reasonable word-count band, 20-150 words for the
  incoming email and 10-120 for the reply. Long enough to carry real
  content, short enough that it's still a normal email and not an
  attachment dump or a one-word "ok."
- de-duplicate on the incoming email text.

That leaves 6,086 clean pairs out of 15,377. From that pool I sample, with a
fixed seed so the result is reproducible, 200 pairs into `data/history.jsonl`
(what the generator retrieves from) and hold out 20 into `data/incoming.jsonl`
(what we test on, each carrying the real reply as its reference). No held-out
email can retrieve itself; the two sets are disjoint before either is built.

Both sets are spread across a keyword-matched category (`scheduling`,
`approval_review`, `hr_recruiting`, `status_update`, `negotiation_deal`,
`request_info`, `personal_social`, `other`) so the 20-item test set doesn't
end up being all one kind of email. That category is a regex over keywords,
built for the report breakdown, not a real taxonomy. A team running this for
real would use whatever tags they already have on their tickets.

**Where this dataset falls short, plainly stated:**
- It's business (and some personal) correspondence between Enron employees
  and their contacts, circa 1999-2002, not customer-support tickets. The
  brief asks for "past emails and the replies that were sent," which this
  is, but if the target product is specifically a support inbox, this
  dataset covers the *reply* half of that problem well and the
  *support-ticket content* half not at all.
- 200 history pairs and 20 test emails are sized to run on a laptop for a
  few dollars in API calls, not to be a benchmark. The clean pool (6,086
  pairs, up to 15,377 unfiltered) is there to scale into via `--history-n`
  and `--incoming-n` if you want more.
- The corpus contains real employees' names and email addresses. I'm
  shipping it unmodified rather than anonymizing it. This is a 20+ year old
  research corpus that's already been published widely, unmodified, in
  papers and public repos, so scrubbing it here wouldn't protect anyone
  who isn't already long since public, and it'd make the data harder to
  verify against its source. That reasoning is specific to how Enron's
  data became public in the first place (a federal investigation, not a
  scrape); it wouldn't hold for an ordinary private mailbox.
- The reference replies are what people actually sent, not something I
  polished to be a model answer. Real people are sometimes short, sometimes
  a little curt. That's a more honest ceiling for the model to be measured
  against than a hand-written "perfect" reply would be. Section 3 covers
  how the metric handles a ceiling that isn't a flat 100.

## 2. The generator

`src/generate.py` and `src/retrieve.py`. For each incoming email:

1. Pull the closest past emails from history with a TF-IDF retriever,
   excluding the email itself.
2. Feed those (email, reply) pairs to Claude as few-shot examples.
3. Generate the reply, told to match the style of the examples but pull
   its facts only from this email and its own thread, not from the examples.

The two sources of grounding are kept deliberately separate. The retrieved
neighbors are there to teach style: real directness, real structure, how a
person actually phrases a decline or a follow-up question, learned from
replies real people sent rather than from a style guide I'd write myself.
The email's own thread is where the facts have to come from, since the
neighbors belong to someone else's conversation entirely. A generator that
just parrots the nearest past reply will drag in a name or a number that
belongs to a different thread, and that's exactly the failure the grounding
axis is built to catch.

**Why retrieval and not fine-tuning.** Fine-tuning would bake the style in
and shrink the prompt, but it's slow to iterate on, costs real money per
run, and 200 history pairs isn't a large enough corpus for it to clearly
beat retrieval. Retrieval gets most of the same benefit, updates the moment
you add one new past email with no retraining step, and stays inspectable:
you can see exactly which past email drove a given reply, logged in
`scored.jsonl`. Fine-tuning would make more sense at production scale, with
thousands of consistent examples and a stable policy to bake in.

**Why retrieval and not plain prompting.** Prompting alone gives the model
nothing to anchor its tone to; it ends up inventing a house style rather
than reflecting how this particular set of senders actually writes.

**Why TF-IDF and not embeddings.** The corpus is small and same-topic
emails here tend to share enough vocabulary that lexical overlap finds a
reasonable neighbor most of the time, and it keeps the repo free of extra
dependencies and fully deterministic. It's weakest on very short or vague
incoming emails, where there's little text to match against. Swapping in an
embedding retriever as the corpus grows would be a small, contained change,
`TfidfRetriever.topk()` is the seam.

## 3. Measuring accuracy

The metric is a reference-anchored rubric judge (`src/judge.py`), backed by
deterministic guardrails (`src/evaluate.py`), checked against floor and
ceiling baselines every time you run `run.py`.

### What "accurate" means for a reply

A reply doesn't have one correct string. "Let's do Tuesday at 2, I'll send
an invite" and "Tuesday 2pm works, expect a calendar hold shortly" are both
fine answers and share almost no words. BLEU, ROUGE, exact match, anything
built on string overlap, would punish the second one for not looking like
the first, which tells you nothing about whether either reply actually did
its job. I'm treating "accurate" as: did the reply address what was
actually asked, using only facts that are actually in the thread, covering
everything that was asked, in a tone that fits. That's what the rubric
scores. There's a lexical-overlap number computed too, but only as a rough
sanity check, and it's the crude heuristic the offline `--mock` judge uses
in place of a real model call, never the real score.

### Five axes, not one number

- **resolution** - did the draft address the sender's actual ask, or dodge it
- **grounding** - does every specific claim (a name, number, date,
  commitment) trace back to something actually in the thread, with nothing
  invented. This is the axis built to catch confident hallucination, and it
  rewards a draft that asks for a missing detail instead of guessing at one.
- **completeness** - were all the questions answered, not just the first one
- **tone** - fitting the situation and how this kind of correspondence
  actually reads
- **clarity** - concise, easy to act on

Splitting it into five named axes instead of one blended score makes the
judge more consistent and makes a low score diagnosable: a bad grounding
score points at hallucination specifically, a bad completeness score points
at a dropped question. Each axis comes with a short written reason, logged
per response in `scored.jsonl`, so "why" isn't a mystery either.

**On grounding, specifically, since it changed from an earlier version of
this project.** The dataset doesn't have a synthetic internal database of
facts to check a claim against; Enron threads aren't support tickets sitting
on top of a CRM. So grounding here means thread-faithfulness: every specific
claim in the draft has to trace back to the incoming email or the prior
message in its thread, not just sound plausible. I'd argue this is actually
closer to how a real suggested-reply tool fails in practice. Most systems
don't have a hidden fact database either; they have the thread, and a
suggested reply that invents a name or date nobody mentioned is the exact
failure mode that would make someone stop trusting the tool.

### Why it's anchored to a reference and not just similarity to it

The judge sees the real reply as one example of a good answer, and uses it
to figure out what a complete, correct response has to cover. That gives it
something concrete to reason from without turning the whole thing into
string matching: it's explicitly told to reward a different-but-correct
answer just as much as one that resembles the reference, and not to
penalize a draft for being clearer or more complete than the reference is.
That instruction matters more here than it would with a hand-written
reference, since the reference is a real reply someone sent under time
pressure, not a polished ideal one.

### Deterministic guardrails on top of the judge

Some failures shouldn't depend on a language model's mood. The guardrails
check that a reply is non-empty and substantive, doesn't leak the prompt's
own scaffolding back (the model echoing a label like "prior message in this
thread" instead of writing an actual reply), has no unfilled placeholder,
and isn't a bare "hi." Each failed check costs 20 points off the quality
score, because an empty or scaffolding-leaking reply isn't 80% good no
matter how the rubric alone would have scored it.

### How I checked the metric actually reflects quality

This is the part I didn't want to leave as an assertion. Every incoming
email gets scored, with the same judge, three ways: a degenerate floor
reply, the model's actual output, and the real reply as a ceiling. A metric
that's doing its job has to rank reference >= model >> floor. If it doesn't,
the metric is broken and the model's number can't be trusted either, so
this comparison runs and prints on every single execution instead of being
a one-time check I ran once and moved on from.

Under `--mock` (no API key, a crude lexical-overlap heuristic standing in
for the real judge), that ordering still holds: the reference well ahead of
the generic mock reply, the floor at zero and failing every guardrail. The
mock heuristic isn't the real metric though, and it's sensitive to reply
length in a way the actual rubric judge isn't, so a short real reference can
score lower under `--mock` than it would under Claude. That's an artifact of
the cheap offline stand-in, not a claim about that reply's quality; the
stand-in exists purely so the pipeline is provably runnable for free.

Two more things keep the scoring honest: judging runs at temperature 0 so
the same input gets the same score run to run, and if the judge ever returns
malformed JSON, the scorer falls back to the heuristic instead of crashing
or silently awarding a high score.

### Reporting

Per response: quality is the mean of the five axis scores (1-5), rescaled
to 0-100, minus guardrail penalties, with full rationales written to
`scored.jsonl`. Overall is the mean across the held-out set. `summary.json`
and `report.md` also break the model down by axis and by category, so a
specific weak spot (grounding on multi-question threads, say) shows up
instead of getting averaged into an unremarkable overall number, and each
response lists which past emails grounded it.

## Layout

```
data/generate_dataset.py    filters/samples the Enron CSV into history + incoming
data/history.jsonl          past emails + real replies (learned from, committed)
data/incoming.jsonl         held-out new emails (tested on, committed)
data/raw/                   downloaded source CSV (gitignored, not committed)
src/retrieve.py             TF-IDF retriever over past emails
src/generate.py             retrieval-grounded Gen-AI reply generator
src/judge.py                reference-anchored rubric judge (thread-faithfulness grounding)
src/evaluate.py             guardrails + per-response + aggregate scoring
run.py                      floor / model / reference, scored end to end
results/                    report.md, summary.json, scored.jsonl
```

## Limitations and what I'd do next

- 20 incoming emails is enough to see the floor/model/reference ordering
  hold and to exercise every axis, not enough for tight confidence
  intervals. `--incoming-n` and `--history-n` are how you'd grow it, up to
  the 6,086-pair clean pool.
- 67% of the source rows have no prior thread message (`thread_type:
  opening_message`), meaning the incoming email alone doesn't always carry
  enough to answer with certainty. The generator is told to ask for missing
  specifics rather than guess in that case, and grounding rewards it for
  doing so, but it's a harder setting than the earlier hand-authored dataset,
  where every ticket had a complete answer built in by construction.
- An LLM judge inherits that model's blind spots. Guardrails and the
  floor/ceiling checks put a bound on that, and splitting into five axes
  plus anchoring to a reference cuts down on variance, but a human spot
  check of the rationales in `scored.jsonl` is the right next step before
  trusting this in production.
- Judge and generator are the same model family right now. Using a
  different model as judge would cut the risk of them sharing blind spots,
  and it's a one-line change via `JUDGE_MODEL`.
- The category label is a keyword heuristic for the report breakdown, not a
  real taxonomy, covered in section 1.
- Lexical retrieval struggles on very short or vague incoming emails; an
  embedding retriever would help there as the corpus grows.

## How I used AI tools

Claude helped scaffold the repo, write the Enron cleaning and sampling
script, and stress-test the evaluation design, particularly the question of
how to define "grounding" once the dataset stopped having synthetic
ground-truth facts to check against. I reviewed and edited all of it. The
actual decisions, Enron as the source, the train/held-out split, the
cleaning filters, the retrieval approach, the five axes, thread-faithfulness
as what grounding means here, the guardrails, the floor/ceiling check, are
mine, and I read the raw CSV and looked at sampled rows myself before
settling on the filters rather than working from a description of the data.
At runtime the system calls the Anthropic API for both generation and
judging, with the models configurable via `GEN_MODEL` and `JUDGE_MODEL`. The
offline mock path uses only the Python standard library; the dataset
builder uses the standard library plus the optional `kaggle` package for
the one-time download.
