"""
tools.py - the three tools shared by the single agent and the team workers.

Failure contract: every tool returns a string and never raises. Failures start with
"ERROR:"; a retrieval below the relevance threshold starts with "NO_RESULTS:".
Tool failures for the tool_fails tasks come only from explicit fault injection
(set_active_faults), never from magic strings in the input.
"""

import ast
import operator
import os
import re
from contextvars import ContextVar, Token
from typing import Dict, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import retriever

PolicyAlias = Literal["auto", "embrace", "nationwide", "travel_fl", "travel_uk", "health", "flood"]

POLICY_HELP = (
    "auto = Allstate auto policy (Virginia); embrace = Embrace pet insurance; "
    "nationwide = Nationwide pet medical plan; travel_fl = Allianz Florida travel plan; "
    "travel_uk = Allianz UK travel policy; health = MHBP Medicare health Evidence of Coverage; "
    "flood = NFIP Standard Flood Insurance Policy (44 CFR 61)."
)

# ---------------------------------------------------------------------------
# Fault injection (driven by the task field `inject_faults`)
# ---------------------------------------------------------------------------

_ACTIVE_FAULTS: ContextVar[Dict[str, str]] = ContextVar("active_faults", default={})


def set_active_faults(faults: Optional[Dict[str, str]]) -> Token:
    """Make the named tools fail for the current run, e.g. {"search_docs": "error"}."""
    return _ACTIVE_FAULTS.set(dict(faults or {}))


def clear_active_faults(token: Optional[Token] = None) -> None:
    if token is not None:
        _ACTIVE_FAULTS.reset(token)
    else:
        _ACTIVE_FAULTS.set({})


def _injected_fault(tool_name: str) -> Optional[str]:
    if tool_name in _ACTIVE_FAULTS.get():
        return f"ERROR: {tool_name} unavailable (injected fault). The service could not be reached."
    return None


# ---------------------------------------------------------------------------
# Tool 1: search_docs (Assignment 3 retriever)
# ---------------------------------------------------------------------------

def _min_score() -> Optional[float]:
    raw = os.environ.get("RAG_MIN_RERANK_SCORE", "").strip()
    return float(raw) if raw else None


def _cite(hit: retriever.Hit) -> str:
    return f"p.{hit.page}" if hit.page.isdigit() else hit.page


class SearchDocsInput(BaseModel):
    query: str = Field(description="What to look for, phrased like the policy wording would phrase it.")
    policy: Optional[PolicyAlias] = Field(
        default=None,
        description="Restrict the search to one policy when the question names it. " + POLICY_HELP,
    )


@tool("search_docs", args_schema=SearchDocsInput)
def search_docs(query: str, policy: Optional[str] = None) -> str:
    """
    Search the insurance policy corpus and return the 5 most relevant passages,
    each with its source document, page (or section for the flood policy) and relevance score.
    Use it for any fact that must come from a policy: limits, deductibles, waiting periods,
    exclusions, deadlines, definitions. Not for arithmetic or general knowledge.
    """
    fault = _injected_fault("search_docs")
    if fault:
        return fault
    if not query or not query.strip():
        return "ERROR: Query cannot be empty."
    try:
        hits = retriever.search(query, policy=policy)
    except Exception as e:
        return f"ERROR: Document search failed: {e}"
    if not hits:
        return "NO_RESULTS: no passages matched."
    threshold = _min_score()
    if threshold is not None and hits[0].score < threshold:
        return (f"NO_RESULTS: best passage scored {hits[0].score:.2f}, below the relevance "
                f"threshold {threshold:.2f}. The corpus likely does not cover this.")
    return "\n\n".join(
        f"[{i}] {h.doc_name} {_cite(h)} | score {h.score:.2f}\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


# ---------------------------------------------------------------------------
# Tool 2: calculator (AST whitelist, no eval)
# ---------------------------------------------------------------------------

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(12000 - 1000) * 1.18' or '250 * 3'.")


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Evaluate an arithmetic expression exactly: + - * / ** % and parentheses.
    ALWAYS use this tool for arithmetic instead of computing in your head.
    """
    fault = _injected_fault("calculator")
    if fault:
        return fault
    if not expression or not expression.strip():
        return "ERROR: Expression cannot be empty."
    # tolerate currency symbols and thousands separators: "$12,000 * 1.18"
    clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression)
    clean = re.sub(r"[$£€₪]", "", clean).strip()
    try:
        value = _eval_node(ast.parse(clean, mode="eval"))
    except ZeroDivisionError:
        return "ERROR: Division by zero is not allowed."
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return f"ERROR: Cannot evaluate {expression!r}: {e}"
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 6)
    return f"RESULT: {value}"


# ---------------------------------------------------------------------------
# Tool 3: read_policy_page
# ---------------------------------------------------------------------------

MAX_PAGE_CHARS = 6000


class ReadPolicyPageInput(BaseModel):
    policy: PolicyAlias = Field(description="Which policy to read. " + POLICY_HELP)
    page: str = Field(
        description="Page number as shown in search_docs results (e.g. '4'). For the flood "
                    "policy, the section label instead (e.g. '§ 61.5' or 'Appendix A(1), V. Exclusions')."
    )


@tool("read_policy_page", args_schema=ReadPolicyPageInput)
def read_policy_page(policy: str, page: str) -> str:
    """
    Return the full text of one policy page (or one flood-policy section).
    Use it after search_docs when a passage is cut off or you need the surrounding clause.
    """
    fault = _injected_fault("read_policy_page")
    if fault:
        return fault
    try:
        text = retriever.read_page(policy, page)
    except Exception as e:
        return f"ERROR: Cannot read {policy} page {page!r}: {e}"
    if not text.strip():
        return f"NO_RESULTS: {policy} page {page!r} has no extractable text."
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + f"\n[truncated: {len(text) - MAX_PAGE_CHARS} more characters]"
    return text


ALL_TOOLS = [search_docs, calculator, read_policy_page]
