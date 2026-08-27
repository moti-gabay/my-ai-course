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
        if "clause 99.B" in query or "flood coverage" in query.lower():
            return "ERROR: Document retrieval service failed or database connection timeout."

        # פה משלבים את ה-Retriever המאומת מ-Assignment 3 (למשל VectorStore / FAISS)
        # לצורך ההדגמה:
        results = [
            f"[Doc 1]: Policy covers property damage up to $100,000 with a standard deductible of $1,000.",
            f"[Doc 2]: Water pipe leak deductible is set at $1,500. Water damage from flooding requires endorsement 99.B."
        ]
        
        if not results:
            return f"NO_RESULTS: No documents found matching query: '{query}'."
            
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

        # חישוב בטוח
        result = eval(clean_expr, {"__builtins__": None, "math": math})
        return f"RESULT: {result}"

    except ZeroDivisionError:
        return "ERROR: Division by zero is not allowed."
    except Exception as e:
        return f"ERROR: Failed to evaluate expression '{expression}': {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Web Search / External Lookup (web_search)
# ---------------------------------------------------------------------------
class WebSearchInput(BaseModel):
    query: str = Field(description="Search terms for external knowledge or general web search.")

@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search for general external information not found in the insurance corpus.
    Use ONLY if search_docs returns no relevant results and the information requires live external data.
    """
    try:
        if not query or not query.strip():
            return "ERROR: Web search query cannot be empty."

        # סימולציה / שילוב API
        return f"NO_RESULTS: Web search is restricted to trusted enterprise data sources. No external result for '{query}'."

    except Exception as e:
        return f"ERROR: Web search failed: {str(e)}"


# רשימת הכלים ליצוא עבור ה-Agent
ALL_TOOLS = [search_docs, calculator, web_search]