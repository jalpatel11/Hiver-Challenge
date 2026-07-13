"""
Retrieval-grounded generator. Two sources of grounding, kept deliberately
separate: the TF-IDF neighbors (src/retrieve.py) teach style, since they're
real replies real people sent to similar emails. The new email's own thread
supplies the actual facts. Neighbors belong to someone else's conversation,
so if the model starts pulling names or numbers from them instead of the
new thread, that's exactly the grounding failure the judge should catch.

Considered fine-tuning instead - would bake in style and shrink the prompt,
but 200 history pairs isn't enough for it to earn its cost, and retrieval
lets you see which past email actually drove a given reply (logged in
scored.jsonl). Plain prompting with no retrieval was the other option; it
just has nothing to anchor the model's tone to.
"""
from src import llm

SYSTEM = (
    "You are drafting a suggested reply to an incoming email inside a shared "
    "inbox. Below are a few PAST emails and the replies that were actually "
    "sent for them. Learn the tone, directness, and structure from those "
    "examples, they are real replies from real threads.\n\n"
    "Then write a reply for the NEW email.\n\n"
    "Rules:\n"
    "1. Match the tone and shape of the past replies, but write for the NEW "
    "email and its own thread.\n"
    "2. Use ONLY facts that are actually in the NEW email or its own prior "
    "thread context. The past examples are for STYLE, not facts, they belong "
    "to different people and different threads. Never copy a name, number, "
    "date, or commitment from an example into this reply. Never invent a "
    "fact, name, or commitment that isn't in the new email or its context.\n"
    "3. Answer every distinct question or request in the new email.\n"
    "4. If the new email lacks the detail needed to give a real answer, ask "
    "for the specific missing information instead of guessing or stalling.\n"
    "5. Be concise and direct, matching how these threads actually read. "
    "Output only the email body, no subject line.\n"
    "6. Write like the person in the examples actually would, not like an "
    "AI assistant. No em dashes. Skip stock phrases like 'I hope this "
    "email finds you well', 'I wanted to reach out', 'please don't "
    "hesitate to', or 'let me know if you have any questions'. No "
    "unnecessary hedging, no restating the question back before answering it."
)

USER_TMPL = (
    "{examples}\n\n"
    "======== NEW EMAIL TO ANSWER ========\n"
    "PRIOR MESSAGE IN THIS THREAD (if any):\n{context}\n\n"
    "INCOMING EMAIL\nSubject: {subject}\n\n{body}\n\n"
    "Write the reply."
)


def _format_examples(neighbors):
    if not neighbors:
        return "(no similar past emails found)"
    blocks = []
    for i, (score, d) in enumerate(neighbors, 1):
        blocks.append(
            f"--- PAST EXAMPLE {i} (similarity {score}) ---\n"
            f"PAST EMAIL:\n{d['incoming_email']}\n\n"
            f"REPLY THAT WAS ACTUALLY SENT:\n{d['reference_reply']}"
        )
    return "PAST EMAILS FOR STYLE REFERENCE:\n\n" + "\n\n".join(blocks)


def _mock_reply(ticket, neighbors):
    """Deterministic offline stand-in. Notes that retrieval ran (shows wiring)
    but stays generic so it scores below the reference."""
    nb = ", ".join(d["id"] for _, d in neighbors) or "none"
    return (
        f"Hi,\n\nThanks for the note about \"{ticket['subject']}\". I've seen this "
        "and will follow up shortly.\n\n"
        f"Best\n\n[mock generator; retrieved neighbors: {nb}]"
    )


def generate_reply(ticket, retriever, k=3, mock=False):
    neighbors = retriever.topk(ticket["incoming_email"], k=k, exclude_id=ticket["id"])
    neighbor_ids = [d["id"] for _, d in neighbors]
    if mock or not llm.have_key():
        return _mock_reply(ticket, neighbors), neighbor_ids
    user = USER_TMPL.format(
        examples=_format_examples(neighbors),
        context=ticket.get("context", "(none)"),
        subject=ticket.get("subject", ""),
        body=ticket["incoming_email"],
    )
    text = llm.complete(SYSTEM, user, model=llm.GEN_MODEL, max_tokens=800, temperature=0.2)
    return text, neighbor_ids
