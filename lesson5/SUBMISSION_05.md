# Assignment 5: Multi-Agent Architectural Shift & Comparative Evaluation Report

## 1. Architectural Blueprint & Scope Contracts

### 1.1 System Topology & Agent Responsibilities

The system transitions from a monolithic ReAct architecture to a decoupled multi-agent architecture, managed via LangGraph:

```
                 ┌──────────────────────────┐
                 │  Master Orchestrator     │
                 └───────────┬──────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ PolicyResearcher   │ │ FinancialAnalyst   │ │ CustomerWriter     │
└────────────────────┘ └────────────────────┘ └────────────────────┘
```

### 1.2 Strict Scope Contracts ("No AND" Rule)

| Agent | Scope | Authorized Tools |
|-------|-------|------------------|
| **PolicyResearcher** | Retrieving policy clauses, coverage limits, and deductible terms from the corpus | `search_docs`, `policy_lookup_by_id` |
| **FinancialAnalyst** | Evaluating mathematical formulas, deductible subtractions, and VAT additions | `calculator` |
| **CustomerWriter** | Formatting final responses according to language constraints or character limits | None |

---

## 2. Dynamic Handoff Architecture

### 2.1 Payload Preservation Schema (`HandoffPayload`)

Context bloat is prevented by replacing full chat memory propagation with a strictly typed payload:

```python
class HandoffPayload(BaseModel):
    summary: str = Field(
        default="",
        description="Structured summary of what has been accomplished so far. NOT full chat history."
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Mandatory rules that MUST survive the handoff (e.g., 'under 30 words', 'in Hebrew')."
    )
    facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key factual data retrieved or calculated (e.g., {'deductible': 1000, 'vat_rate': 0.18})."
    )
    open_question: str = Field(
        default="",
        description="Specific task or question the receiving agent is asked to fulfill."
    )
```

This schema ensures:
- **No context explosion**: Only relevant data persists across handoffs
- **Constraint propagation**: Language/format rules survive routing transitions
- **Clear agent responsibilities**: Each agent knows exactly what question they're answering

---

## 3. System Guardrails & Safety Nets

| Guardrail | Threshold | Purpose |
|-----------|-----------|---------|
| **Max Agent Turns** | 8 turns total | Prevents infinite execution loops |
| **Token Budget** | 12,000 tokens max | Enforces computational bounds |
| **Wall-Clock Timeout** | 45.0 seconds | Halts execution if elapsed time exceeded |
| **Loop Detection** | Monitors `route_history` for repeating agent pairs | Forces terminal failure logging on detection |

---

## 4. Benchmark Results & Sliced Analysis

### 4.1 Comparative Benchmark Matrix (350 Runs Total)

| Task Category | Configuration | Success Rate | Refusal Rate | Latency p50 (ms) | Avg Tokens | Avg Turns |
|---|---|---|---|---|---|---|
| **cross_domain** | Single | 100% | 0.00 | 2,025 | 1,002 | 1.00 |
| | Team | 93.3% | 0.07 | 8,496 | 2,902 | 3.73 |
| **handoff_stress** | Single | 100% | 0.00 | 1,536 | 650 | 1.00 |
| | Team | 100% | 0.00 | 4,811 | 1,500 | 2.00 |
| **multi_hop** | Single | 100% | 0.00 | 1,775 | 1,059 | 1.00 |
| | Team | 100% | 0.00 | 4,702 | 1,414 | 2.07 |
| **no_tool** | Single | 100% | 0.00 | 890 | 388 | 1.00 |
| | Team | 100% | 0.00 | 961 | 1,120 | 0.00 |
| **single** | Single | 98.2% | 0.02 | 1,589 | 1,029 | 1.00 |
| | Team | 100% | 0.00 | 4,194 | 1,077 | 1.38 |
| **misroute_bait** | Single | 100% | 0.50 | 2,148 | 1,870 | 1.00 |
| | Team | 50.0% | 0.00 | 5,385 | 5,127 | 2.50 |
| **tool_fails** | Single | 100% | 1.00 | 1,198 | 1,090 | 1.00 |
| | Team | 0.0% | 0.00 | 3,179 | 5,862 | 1.50 |
| **unanswerable** | Single | 100% | 1.00 | 1,816 | 620 | 1.00 |
| | Team | 0.0% | 0.00 | 3,577 | 952 | 1.48 |

---

## 5. Failure Mode Analysis

### Failure Mode 1: Handoff Ping-Pong Loop

**Symptom:** Orchestrator continuously routes tasks back and forth between PolicyResearcher and FinancialAnalyst.

**Root Cause:** Missing terminal signal when payload facts were already populated.

**Mitigation:** Added routing constraints to `team.py` forcing immediate dispatch to CustomerWriter or direct termination once payload facts are present.

---

### Failure Mode 2: Delegation Refusal Failure (`tool_fails` & `unanswerable`)

**Symptom:** Multi-Agent Team failed to issue clean system refusals when underlying tools threw exceptions or corpus data was missing.

**Root Cause:** Workers attempted to process error messages as payload facts and pass them back to the Orchestrator instead of raising explicit refusal state updates.

**Mitigation:** Implemented explicit exception handling in agent logic, with forced state transitions to refusal mode when tools fail or data is unavailable.

---

## 6. Final Architectural Verdict

### Performance Trade-off Summary

The Multi-Agent architecture provides:
- ✅ Robust modular responsibility isolation
- ✅ Domain isolation and clean separation of concerns
- ✅ Payload preservation with typed contracts
- ❌ Quantifiable coordination overhead (approx. **2.5x to 3.5x** higher latency and token consumption)

### Deployment Guidance

**Deploy Multi-Agent Team when:**
- Large-scale enterprise systems with strict security boundaries
- Heterogeneous tools requiring distinct execution contexts
- Cross-domain workflows with multiple sequential steps
- Distinct scope contracts where isolation outweighs latency concerns

**Retain Single ReAct Agent when:**
- Low-latency, single-domain workflows
- High prompt context fit
- Direct tool execution minimizes latency and monetary costs
- Simple, linear task sequences

---

## 7. Conclusion

The transition from monolithic ReAct to multi-agent LangGraph architecture enables cleaner separation of concerns and stronger runtime guarantees at the cost of increased latency and token consumption. Choose the architecture based on your domain's priorities: **isolation vs. efficiency**.