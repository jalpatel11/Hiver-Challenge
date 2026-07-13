"""
LiteLLM wrapper. Calls the real API if LLM_API_KEY is set; the
callers fall back to mock output otherwise so `python run.py --mock` proves
the pipeline works with zero setup and no cost.
"""
import os
import litellm

GEN_MODEL = os.getenv("GEN_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")


def have_key() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def complete(system: str, user: str, model: str, max_tokens: int = 1024,
             temperature: float = 0.2) -> str:
    # NOTE: Some reasoning models (e.g., OpenAI o-series) reject custom temperature
    # values or handle parameters differently. If targeting a reasoning model, 
    # you may need to adjust or omit these parameters to prevent provider errors.
    resp = litellm.completion(
        model=model,
        api_key=os.getenv("LLM_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()
