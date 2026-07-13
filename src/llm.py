"""
Thin Claude wrapper. Calls the real API if ANTHROPIC_API_KEY is set; the
callers fall back to mock output otherwise so `python run.py --mock` proves
the pipeline works with zero setup and no cost.
"""
import os

GEN_MODEL = os.getenv("GEN_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")


def have_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _client():
    from anthropic import Anthropic
    return Anthropic()


def complete(system: str, user: str, model: str, max_tokens: int = 1024,
             temperature: float = 0.2) -> str:
    """Single-turn completion. Returns concatenated text blocks."""
    client = _client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()
