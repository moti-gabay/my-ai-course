"""
judge.py - Sonnet judge, following the Assignment 2 pattern (lesson2/task5_judge.py):
rubric in the system prompt, a Pydantic schema, `explanation` BEFORE `verdict` so the
verdict is conditioned on the reasoning.

Blind by construction: each function receives only what its rubric needs. None of them
takes the config (single/team), the task type, or the answerable label.

Every result is cached in judge_cache.jsonl, keyed by a hash of model + function + inputs,
so re-scoring the same traces never pays twice.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Type

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).with_name(".env"))

JUDGE_MODEL = "claude-sonnet-5"
CACHE_PATH = Path(__file__).with_name("judge_cache.jsonl")

_client = None
_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Schemas (explanation first)
# ---------------------------------------------------------------------------

class SuccessVerdict(BaseModel):
    explanation: str = Field(description="Compare the answer with the reference, citing the key facts. 2-4 sentences.")
    verdict: Literal["pass", "fail"]


class FaithfulnessVerdict(BaseModel):
    explanation: str = Field(description="List the answer's factual claims and whether each is supported by the tool outputs.")
    verdict: Literal["supported", "partially_supported", "unsupported", "no_claims"]


class AgentTurnVerdict(BaseModel):
    explanation: str = Field(description="What the agent was asked, what it received, and whether its output does that job.")
    verdict: Literal["pass", "fail"]


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

SUCCESS_RUBRIC = """You judge whether an assistant's ANSWER to an insurance QUESTION is correct, using the REFERENCE ANSWER as ground truth.

PASS when the answer reaches the same substantive conclusion as the reference: the same key facts, figures and yes/no outcome. Extra correct detail is fine. Language, wording, length and formatting do not matter.
If the reference says the question cannot be answered from the policy documents, PASS only when the answer declines or says the information is not available, without inventing a value.
FAIL when a key fact or figure is wrong or missing, when the answer contradicts the reference, or when it declines although the reference gives an answer.
If a TASK-SPECIFIC CRITERION is given, the answer must also meet it.

Judge only against the reference and the criterion, not your own knowledge of insurance."""

FAITHFULNESS_RUBRIC = """You judge whether an assistant's ANSWER is faithful to the TOOL OUTPUTS it had: policy passages returned by search tools, page texts, and calculator results.

You also get the user's QUESTION. Facts the user states about their own situation (their amounts, counts, dates, circumstances) count as given: the answer may repeat them or compute with them. The question never supports facts about a policy, a company, or the world.

Identify the answer's factual claims: coverage, limits, amounts, periods, conditions, exclusions, computed figures, and any names, contact details, or other policy or company specifics. Check each claim against the tool outputs only, never against your own knowledge. A figure counts as supported when a tool output states it or a calculator output shows it. Greetings, offers to help, and statements that information is unavailable are not factual claims.

- supported: every factual claim is supported by the tool outputs.
- partially_supported: at least one claim is supported and at least one is not supported or is contradicted.
- unsupported: the answer's central claim is not supported, or it is contradicted by the tool outputs.
- no_claims: the answer asserts no specific facts: no numbers, names, contact details, or policy or company specifics (for example a greeting, a capability overview, or a refusal that only says the information is unavailable)."""

AGENT_TURN_RUBRIC = """You judge ONE agent in a multi-agent system: did it do ITS OWN job correctly, given what it received?

You get the agent's ROLE, the PAYLOAD it received (summary, constraints, facts, open_question), its TOOL OUTPUTS, and its OUTPUT.

Judge the agent against the payload it received, not against the truth of the world. If the payload itself was wrong or incomplete, the agent still PASSES when it did the right thing with what it received: that failure belongs to the layer above it.

PASS when the output does what the open_question asked, stays within the agent's role, respects every constraint in the payload, and reports tool results accurately (including reporting a tool failure or an empty search honestly).
FAIL when it ignores the request or a constraint, works outside its role, misreads or misreports a tool output, invents facts that neither the payload nor its tools provided, or returns output the next step cannot use."""


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _client_() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _load_cache() -> None:
    if _cache or not CACHE_PATH.exists():
        return
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        _cache[entry["key"]] = entry


def _judge(kind: str, system: str, user: str, schema: Type[BaseModel]) -> Dict[str, Any]:
    """-> {verdict, explanation, input_tokens, output_tokens, cached}. Never raises."""
    _load_cache()
    key = hashlib.sha256(json.dumps([JUDGE_MODEL, kind, system, user]).encode()).hexdigest()
    if key in _cache:
        return {**_cache[key]["result"], "cached": True}
    try:
        resp = _client_().messages.parse(
            model=JUDGE_MODEL, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}], output_format=schema,
        )
    except (anthropic.APIError, ValueError) as e:  # ValueError covers schema validation failures
        return {"verdict": "judge_error", "explanation": f"{type(e).__name__}: {e}",
                "input_tokens": 0, "output_tokens": 0, "cached": False}
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    parsed = resp.parsed_output
    if resp.stop_reason == "refusal" or parsed is None:
        return {"verdict": "judge_error", "explanation": f"no parsed output (stop_reason={resp.stop_reason})",
                **usage, "cached": False}
    result = {"verdict": parsed.verdict, "explanation": parsed.explanation, **usage}
    entry = {"key": key, "kind": kind, "model": JUDGE_MODEL, "result": result}
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _cache[key] = entry
    return {**result, "cached": False}


def _format_tool_outputs(tool_outputs: List[Dict[str, Any]]) -> str:
    if not tool_outputs:
        return "(no tool calls)"
    return "\n\n".join(
        f"[{i}] {t.get('tool')}({json.dumps(t.get('input'), ensure_ascii=False)})\n{t.get('output')}"
        for i, t in enumerate(tool_outputs, start=1)
    )


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

def judge_task_success(question: str, reference_answer: str, answer: str, criterion: str = "") -> Dict[str, Any]:
    user = (f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference_answer}\n\n"
            + (f"TASK-SPECIFIC CRITERION:\n{criterion}\n\n" if criterion else "")
            + f"ANSWER TO JUDGE:\n{answer}")
    return _judge("task_success", SUCCESS_RUBRIC, user, SuccessVerdict)


def judge_faithfulness(answer: str, tool_outputs: List[Dict[str, Any]], question: str = "") -> Dict[str, Any]:
    user = (f"QUESTION:\n{question or '(not provided)'}\n\n"
            f"TOOL OUTPUTS:\n{_format_tool_outputs(tool_outputs)}\n\nANSWER TO JUDGE:\n{answer}")
    return _judge("faithfulness", FAITHFULNESS_RUBRIC, user, FaithfulnessVerdict)


def judge_agent_turn(role: str, received_payload: Dict[str, Any], tool_outputs: List[Dict[str, Any]],
                     output: str) -> Dict[str, Any]:
    user = (f"ROLE:\n{role}\n\nPAYLOAD RECEIVED:\n{json.dumps(received_payload, ensure_ascii=False, indent=2)}\n\n"
            f"TOOL OUTPUTS:\n{_format_tool_outputs(tool_outputs)}\n\nOUTPUT TO JUDGE:\n{output}")
    return _judge("agent_turn", AGENT_TURN_RUBRIC, user, AgentTurnVerdict)
