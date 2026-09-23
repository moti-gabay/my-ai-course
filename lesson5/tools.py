import math
import re
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tool 1: Document Retriever (search_docs)
# ---------------------------------------------------------------------------
class SearchDocsInput(BaseModel):
    query: str = Field(description="The precise search query to look up in the insurance policy corpus.")

@tool("search_docs", args_schema=SearchDocsInput)
def search_docs(query: str) -> str:
    """
    Search the insurance policy documents for terms, coverage limits, deductibles, and specific clauses.
    Use this tool whenever you need facts from the insurance corpus.
    Do NOT use for general math calculations or non-insurance common knowledge.
    """
    try:
        if not query or not query.strip():
            return "ERROR: Query cannot be empty."
        
        # סימולציית כישלון עבור משימת tool_fails
        if "clause 99.B" in query.lower() and "fail" in query.lower():
            return "ERROR: Document retrieval service failed or database connection timeout."

        results = [
            "[Doc 1]: Policy covers property damage up to $100,000 with a standard deductible of $1,000.",
            "[Doc 2]: Water pipe leak deductible is set at $1,500. Water damage from flooding requires endorsement 99.B.",
            "[Doc 3]: Medical expenses coverage limit is $25,000 per policy year.",
            "[Doc 4]: Part D collision deductible is standard $1,000 with a 5% administrative surcharge on premiums."
        ]
        
        return "\n".join(results)

    except Exception as e:
        return f"ERROR: Failed to search documents due to an unexpected error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2: Calculator (calculator)
# ---------------------------------------------------------------------------
class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g., '150 * 12 * 1.05' or '(12000 - 1000) * 1.18'")

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Perform precise numerical calculations (addition, subtraction, multiplication, division, percentages).
    ALWAYS use this tool for arithmetic operations. Do NOT do math in your head.
    """
    try:
        if not expression or not expression.strip():
            return "ERROR: Expression cannot be empty."

        # סימולציית כישלון עבור משימת tool_fails
        if "compound interest" in expression.lower():
            return "ERROR: Calculator service error - compound interest evaluation is temporarily disabled."

        # ניקוי ביטוי לבטיחות
        clean_expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expression)
        if not clean_expr.strip():
            return f"ERROR: Invalid expression '{expression}'. Only numbers and math operators allowed."

        result = eval(clean_expr, {"__builtins__": None, "math": math})
        return f"RESULT: {result}"

    except ZeroDivisionError:
        return "ERROR: Division by zero is not allowed."
    except Exception as e:
        return f"ERROR: Failed to evaluate expression '{expression}': {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Policy Clause Direct Lookup (policy_lookup_by_id)
# ---------------------------------------------------------------------------
class PolicyLookupInput(BaseModel):
    clause_id: str = Field(description="Direct ID of the clause or endorsement to look up, e.g., '99.B' or 'PART_D'.")

@tool("policy_lookup_by_id", args_schema=PolicyLookupInput)
def policy_lookup_by_id(clause_id: str) -> str:
    """
    Lookup policy clause or endorsement details directly by clause ID (e.g., '99.B').
    Use this for direct structural code lookups in policy docs.
    """
    try:
        if not clause_id or not clause_id.strip():
            return "ERROR: Clause ID cannot be empty."

        clean_id = clause_id.strip().upper()

        if "99.B" in clean_id or "99B" in clean_id:
            return "Endorsement 99.B: Covers water damage resulting from natural flooding with a special deductible of $1,500."
        elif "PART_D" in clean_id or "PART D" in clean_id:
            return "Part D Coverage: Specifies collision deductible of $1,000 and 5% administrative surcharge."
        elif "12" in clean_id:
            return "Clause 12: General property damage liability limits up to $100,000."
        else:
            return f"NO_RESULTS: Policy clause '{clause_id}' lookup not found in direct index."

    except Exception as e:
        return f"ERROR: Policy lookup failed: {str(e)}"


# רשימת הכלים היצואים עבור ה-Agent
ALL_TOOLS = [search_docs, calculator, policy_lookup_by_id]