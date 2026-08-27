


# Assignment 04: Agent Security, Architecture & Evaluation Report

**Author:** Mordehai (Moti) Yehuda Gabay  
**Project:** Insurance Assistant Agent Benchmark & Evaluation  
**Dataset:** 25 Tasks across 5 Categories (500 total executions)  

---

## 1. System Architecture & Tools Description

The system is designed as an automated insurance assistant capable of evaluating claims, looking up policy terms, performing mathematical calculations, and accurately refusing unanswerable or out-of-scope requests.

### Tool Definitions (`tools.py`)

1. **`search_docs(query: str) -> str`**
   * **Purpose:** Performs semantic/keyword retrieval over the insurance policy knowledge base.
   * **Behavior:** Returns relevant text passages (e.g., deductible amounts, endorsement clauses, coverage limits). If no matching terms exist, returns an informative missing data response.
2. **`calculator(expression: str) -> str`**
   * **Purpose:** Evaluates exact mathematical and financial formulas (e.g., deducting deductibles, adding VAT, compounding interest).
   * **Behavior:** Uses a secure evaluation environment. Prevents LLM arithmetic hallucinations on complex multi-step math.
3. **`policy_lookup_by_id(clause_id: str) -> str`**
   * **Purpose:** Performs a direct structural lookup for specific policy clauses or endorsement IDs (e.g., `99.B`).
   * **Behavior:** Handles edge cases such as tool failures or simulated database timeouts to test agent resilience (`tool_fails`).

---

## 2. Agent Pattern Analysis & Structural Choice

### Architectural Pattern: ReAct vs. Evaluator-Optimizer
* **Baseline Configuration:** Standard ReAct (Reason + Act) loop built using `LangGraph` (`create_react_agent`).
* **Advanced Experiment (Exp 2):** Evaluator-Optimizer pattern. The agent generates a candidate response, which is then routed to an **Evaluator Node** (System Critique). If the response contains reasoning errors, missing tool invocations, or arithmetic inaccuracies, it is returned to the **Optimizer Loop** with actionable feedback.


---
+-------------------------------------------------------------------------+
|                        Evaluator-Optimizer Loop                         |
|                                                                         |
|  [User Query] --> [ReAct Agent] --> [Evaluator Node]                    |
|                         ^                 |                             |
|                         |-- (Invalid) ----+                             |
|                                           v (Valid)                     |
|                                   [Final Response]                      |
+-------------------------------------------------------------------------+

---

### Before vs. After Analysis
* **Before (Base ReAct Agent):** High performance on simple queries, but prone to minor calculation skips in complex `multi_hop` queries when trying to return an answer in a single step.
* **After (Evaluator-Optimizer):** Increased success rate on complex `multi_hop` tasks from **83.33% to 93.33%**, successfully catching and correcting arithmetic errors before returning output to the user.

---

## 3. Sliced Evaluation Table

The benchmark evaluated 4 distinct configurations across 500 execution logs (25 tasks x 5 runs x 4 configs):

| Configuration | Task Type | Total Runs | Success Rate (%) | Latency p50 (s) | Latency p95 (s) | Avg Tokens | Avg Tool Calls | Refusal Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static RAG** | `multi_hop` | 30 | 33.33% | 0.779 | 2.368 | 166.7 | 0.00 | 66.67% |
| **Static RAG** | `no_tool` | 15 | 100.00% | 0.813 | 1.041 | 130.0 | 0.00 | 66.67% |
| **Static RAG** | `single` | 55 | 0.00% | 0.652 | 0.835 | 119.8 | 0.00 | 100.00% |
| **Static RAG** | `tool_fails` | 10 | 100.00% | 0.631 | 0.854 | 100.5 | 0.00 | 100.00% |
| **Static RAG** | `unanswerable` | 15 | 100.00% | 0.643 | 0.895 | 125.0 | 0.00 | 100.00% |
| **LangGraph Agent (Base)** | `multi_hop` | 30 | 83.33% | 2.048 | 4.041 | 1086.0 | 1.87 | 0.00% |
| **LangGraph Agent (Base)** | `no_tool` | 15 | 100.00% | 1.032 | 1.555 | 410.2 | 0.20 | 0.00% |
| **LangGraph Agent (Base)** | `single` | 55 | 94.55% | 1.949 | 2.683 | 1099.8 | 1.44 | 5.45% |
| **LangGraph Agent (Base)** | `tool_fails` | 10 | 100.00% | 2.671 | 3.152 | 1104.9 | 1.50 | 100.00% |
| **LangGraph Agent (Base)** | `unanswerable` | 15 | 100.00% | 1.537 | 2.257 | 727.9 | 0.67 | 100.00% |
| **Exp 1: Prompt Refinement** | `multi_hop` | 30 | 83.33% | 1.939 | 2.868 | 850.6 | 1.50 | 0.00% |
| **Exp 1: Prompt Refinement** | `no_tool` | 15 | 100.00% | 0.957 | 2.238 | 898.8 | 0.00 | 0.00% |
| **Exp 1: Prompt Refinement** | `single` | 55 | 90.91% | 1.836 | 2.495 | 871.9 | 1.07 | 9.09% |
| **Exp 1: Prompt Refinement** | `tool_fails` | 10 | 30.00% | 2.319 | 2.981 | 808.2 | 1.50 | 0.00% |
| **Exp 1: Prompt Refinement** | `unanswerable` | 15 | 0.00% | 1.741 | 2.178 | 831.1 | 1.00 | 0.00% |
| **Exp 2: Evaluator-Optimizer** | `multi_hop` | 30 | **93.33%** | 10.514 | 15.234 | 3309.7 | 4.80 | 0.00% |
| **Exp 2: Evaluator-Optimizer** | `no_tool` | 15 | 100.00% | 2.690 | 4.078 | 525.0 | 0.00 | 26.67% |
| **Exp 2: Evaluator-Optimizer** | `single` | 55 | **98.18%** | 8.094 | 10.728 | 2550.9 | 3.09 | 1.82% |
| **Exp 2: Evaluator-Optimizer** | `tool_fails` | 10 | 20.00% | 5.125 | 19.693 | 272.7 | 4.30 | 0.00% |
| **Exp 2: Evaluator-Optimizer** | `unanswerable` | 15 | 66.67% | 2.911 | 8.769 | 1251.1 | 1.47 | 33.33% |

---

## 4. Experiments: Hypotheses, Implementation & Conclusions

### Experiment 1: System Prompt & Tool Descriptions Optimization
* **Hypothesis:** Restricting tool access rules via System Prompt instructions will eliminate unnecessary tool calls in general conversational tasks (`no_tool`) and reduce latency.
* **Implementation:** Updated the prompt with explicit boundary constraints: *"NEVER call any tool for general conversation or tasks that do not require external policy context."*
* **Results & Findings:**
  * **Success:** Reduced tool calls in `no_tool` tasks from **0.20 down to 0.00**, eliminating over-retrieval.
  * **Trade-off:** Over-constraining the prompt caused the model to attempt answering unanswerable tasks rather than triggering standard refusal safety rules, lowering safety scores in edge cases.

### Experiment 2: Evaluator-Optimizer Critique Loop
* **Hypothesis:** Adding an Evaluator Reflection node to inspect generated outputs will correct multi-step arithmetic errors and improve success rates on `multi_hop` tasks.
* **Implementation:** Wrapped the ReAct loop inside a feedback evaluation step that tests math precision and claim verification before finalizing output.
* **Results & Findings:**
  * **Success:** Multi-hop success rate increased to **93.33%**, and single-hop success reached **98.18%**.
  * **Trade-off:** Latency p50 jumped from **2.05s to 10.51s**, and token consumption tripled (averaging **3,309 tokens/task**).

---

## 5. Annotated Trace Analysis (3 Selected Runs)

### Trace 1: Best Win (Task `t01` - `multi_hop`, LangGraph Agent)
* **Goal:** Calculate claim payout after a $1,000 deductible on a $12,000 claim, plus 18% VAT.
* **Step 1 (Thought):** *"I need to search for the policy deductible and calculate the net payout plus VAT."*
* **Step 2 (Action):** `calculator(expression="12000 - 1000")` -> **Result:** `11000`.
* **Step 3 (Action):** `calculator(expression="11000 * 1.18")` -> **Result:** `12980`.
* **Step 4 (Final Answer):** *"After applying the deductible of $1,000, the insurer will pay $11,000. With 18% VAT, the total payout is $12,980."*
* **Annotation:** Perfect execution. The agent avoided doing mental math and delegated every step to the calculator tool.

### Trace 2: Worst Failure (Task `t04` - `multi_hop`, Static RAG)
* **Goal:** Calculate premium surcharge based on policy context.
* **Execution:** Static RAG retrieved static context without invoking tools. It failed to evaluate the math expression dynamically.
* **Final Answer:** *"I cannot answer that."*
* **Annotation:** Total system failure caused by static RAG limits. Demonstrates why dynamic tool access is required for multi-hop tasks.

### Trace 3: Tool Failure Graceful Handling (Task `t13` - `tool_fails`, LangGraph Agent)
* **Goal:** Retrieve Clause 99.B (Simulated API failure).
* **Step 1 (Action):** `policy_lookup_by_id(clause_id="99.B")`
* **Step 2 (Tool Output):** `Error 500: Internal Retrieval Timeout.`
* **Step 3 (Thought):** *"The policy lookup tool encountered an error. I must inform the user gracefully without hallucinating."*
* **Step 4 (Final Answer):** *"I am unable to retrieve policy clause 99.B due to a retrieval system error. Please try again later."*
* **Annotation:** Excellent resilience. The agent respected safety boundaries when external APIs failed.

---

## 6. Final Verdict & Production Recommendation

**Verdict: Hybrid Routing Architecture (Ship Workflow for Simple Tasks, Agent for Complex Tasks)**

* **Do NOT ship Static RAG alone:** Static RAG failed completely on single-step dynamic lookups (0% success) Do NOT ship Static RAG alone: Static RAG failed completely on and multi-hop arithmetic (33.33% success).
* **Do NOT ship Evaluator-Optimizer across all endpoints:** While Exp 2 achieved peak accuracy (93.33%), its 10-second latency and high token consumption make it unviable for real-time user chat.
* **Production Recommendation:** Implement an intent-based **Router**:
  1. Route general chat (`no_tool`) and standard lookups (`single`) to **Exp 1 (Prompt Refinement Agent)** for sub-second responses.
  2. Route complex multi-step financial claims (`multi_hop`) to **Exp 2 (Evaluator-Optimizer)** where execution accuracy outweighs latency concerns.
  3. Enforce strict fallback rules to preserve the 100% refusal accuracy observed in the Base Agent for unanswerable requests.

```