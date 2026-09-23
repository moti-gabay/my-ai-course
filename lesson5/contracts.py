"""
contracts.py - Typed Contracts, Schemas & State Definition for Multi-Agent System (Assignment 5)
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import operator


# ============================================================================
# 1. Agent Names & Enums
# ============================================================================

class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"


# ============================================================================
# 2. Handoff Schema (Explicit Payload & Handoff Object)
# ============================================================================

class HandoffPayload(BaseModel):
    summary: str = Field(
        default="",
        description="Structured summary of what has been accomplished so far. NOT full chat history."
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Mandatory rules that MUST survive the handoff (e.g. 'under 30 words', 'in Hebrew')."
    )
    facts: Dict[str, Any]= Field(
        default_factory=dict,
        description="Key factual data retrieved or calculated (e.g. {'deductible': '1000', 'vat_rate': '0.18'})."
    )
    open_question: str = Field(
        default="",
        description="Specific task or question the receiving agent is asked to fulfill."
    )
    
class Handoff(BaseModel):
    """
    Typed handoff contract controlling transition between agents.
    """
    destination: Literal["orchestrator", "researcher", "analyst", "writer"] = Field(
        description="Target agent to handle the next step."
    )
    payload: HandoffPayload = Field(
        default_factory=HandoffPayload,
        description="Data payload passed to the target agent."
    )
    reason: str = Field(
        description="Justification for the handoff for tracing purposes."
    )
    direct_answer: Optional[str] = Field(
        default=None,
        description="Only when destination is 'orchestrator': the complete reply to the user. Leave empty otherwise."
    )


# ============================================================================
# 3. LangGraph Shared State (Hybrid State Model)
# ============================================================================

class TeamState(TypedDict):
    """
    LangGraph state schema for multi-agent coordination.
    Combines message history with explicit ownership and handoff payload.
    """
    messages: Annotated[List[Dict[str, Any]], operator.add]
    task_id: str
    user_query: str
    last_active: str                        # Current owner of the conversation
    handoff_data: Optional[Dict[str, Any]]  # HandoffPayload dict representation
    agent_turns_count: int                  # Turn counter for loop & limit safety nets
    route_history: List[str]                # Audit log of agent transitions (e.g. ['orchestrator', 'researcher'])
    is_terminal: bool                       # Signal if conversation is completed
    terminal_reason: Optional[str]          # Terminal state: 'answered', 'refused', 'loop_detected', etc.


# ============================================================================
# 4. Scope Contracts (Task 2: "No AND" Rule per Agent Scope)
# ============================================================================

AGENT_SCOPE_CONTRACTS = {
    AgentName.RESEARCHER: {
        "name": "PolicyResearcher",
        "scope": "Retrieves exact insurance policy clauses, coverage limits, and deductible terms from the knowledge corpus.",
        "tools": ["search_docs", "read_policy_page"],
        "max_tools": 2,
    },
    AgentName.ANALYST: {
        "name": "FinancialAnalyst",
        "scope": "Evaluates numerical mathematical formulas, deductibles, VAT additions, and financial figures.",
        "tools": ["calculator"],
        "max_tools": 1,
    },
    AgentName.WRITER: {
        "name": "CustomerWriter",
        "scope": "Formats final user responses in strictly required languages or length constraints.",
        "tools": [],
        "max_tools": 0,
    }
}