# Assignment 5 — Multi-Agent Systems

**Lecture 5 · Advanced Agents & Multi-Agent Systems · Final project**

---

## What this assignment is about

In Assignment 4 you built one agent: a ReAct loop, three tools, a max-iteration cap, and an eval suite that measured what autonomy cost you. It worked. Somewhere in Task 6 you also found its ceiling — a tool it kept choosing wrong, a context that got expensive, or three lookups it insisted on doing one after another.

This assignment splits that agent into a **team**, and then asks the only question that matters: **was it worth it?**

Be clear about what you're being graded on. Wiring three agents to a supervisor is an afternoon of plumbing — CrewAI will hand you one in fifty lines. The engineering is in naming which measured limit forced the split, building the system so the failures are *readable*, and producing a table that says honestly whether the team beat the soloist. **A submission where the multi-agent system loses, backed by traces and numbers, is worth more than one that wins and can't explain why.**

Here's the arc:

1. **Run the wall test and freeze your Assignment 4 agent** — name the limit in one sentence, and pre-register what you expect to happen.
2. **Design the team** — three agents with scope contracts, a topology, and a model each.
3. **Write the contract layer** — shared state vs. message passing, and a typed handoff schema.
4. **Build it** — orchestrator, workers, four safety nets, unit-of-work tracing.
5. **Give it a memory** — the cheap one, the one that's just a file.
6. **Evaluate** — single agent vs. team, plus per-agent attribution.
7. **Hunt two failure modes** — reproduce them, mitigate them, re-measure.
8. **Demo it** — ten minutes, with the trace on screen.

The through-line from class: **if you don't need a second agent, don't add one.** Most of the grade is in whether you can tell — and defend it with numbers.

> **Keep working in your course repo.** You need your Assignment 4 agent, tools and traces, your Assignment 3 retriever and index, and your Assignment 2 judges. Do not start a new project. This is the fifth layer on one system, not a fifth system.

---

## Setup

Add to your existing virtual environment:

```bash
pip install langgraph                    # multi-agent graphs, subgraphs, handoffs
# or: pip install crewai                 # role-based crews
# or: pip install pyautogen              # conversation-based
```

**Use LangGraph unless you have a reason not to.** You already know it from Assignment 4, its handoffs are explicit rather than emergent, and its state model is the one this assignment asks you to reason about. CrewAI is faster to a demo and hides exactly the mechanics you're being graded on. If you pick something else, say why in the write-up — that's a legitimate paragraph, not a penalty.

Four roles this time:

| Role | Model | Why |
|---|---|---|
| **Orchestrator** | `claude-haiku-4-5` | it classifies and dispatches — the cheapest job in the system, and it runs on *every* turn |
| **Workers** | `claude-haiku-4-5` | you'll run every task 5 times through N agents; start cheap everywhere |
| **Judge** | `claude-sonnet-5` | stronger, and not a model being judged |
| **Upgrade lever** | `claude-sonnet-5` | held in reserve for **one** agent in Task 7, not a starting point |

> **Start every agent on Haiku deliberately.** When the router misdispatches, that is a **finding** about your routing prompt, not a reason to buy a bigger model. The diagnostic order from class holds: tool and agent **descriptions** first, missing **deterministic logic** second, **safety nets** third, **context** fourth, model last.

**Cost warning — read this one properly, it's worse than Assignment 4.** A multi-agent turn costs a router call *plus* a worker trajectory, and the worker re-sends its own growing context on every step. Three agents on the same task is roughly 3× the calls, and latency becomes the **sum** of the chain rather than the max.

Before you launch the full matrix:

- Wire the four nets from Task 4 **first**, including loop detection.
- Dry-run **3 tasks × 1 run × both configs** and read the token counts.
- Multiply by your matrix size. If that number frightens you, cut the task set — never the nets.

---

## The pre-flight question

Before any of the tasks below, answer this in one sentence and put it at the top of your write-up:

> **Which wall is my single agent hitting?**
> `tool overload` · `context bloat` · `serial bottleneck` · `team boundaries`

If you can't name one from your Assignment 4 numbers, your honest answer is *"none — my single agent was fine."* **Write that down and build the team anyway**, because this is a teaching exercise and the comparison is the deliverable. But say it out loud in the report, and expect your Task 6 table to agree with you. That prediction, made in advance and then confirmed by data, is a **better** submission than a hunch that got lucky.

---

## The evaluation criteria

Nine things you'll measure. Note which are outcome metrics and which are attribution metrics — the distinction is the whole trick of evaluating a team.

| Criterion | What it's asking | Measured by |
|---|---|---|
| **Task success** | Is the **final answer / final state** correct? | **code** where checkable, else judge |
| **Faithfulness** | Is every claim supported by what the tools actually returned? | LLM judge |
| **Per-agent success** | Did each agent do **its own job** correctly, given what it received? | **code** + judge |
| **Routing accuracy** | Was the request dispatched to an agent **capable** of it? | **code** + judge |
| **Handoff correctness** | Did the payload carry what the receiver actually needed? | **code** + judge |
| **Cost** | Tokens, $, **agent turns**, tool calls | **code** |
| **Latency** | Wall-clock, **p50 and p95** | **code** |
| **Reliability** | Same task, **5 runs** — how often does it succeed? | **code** |
| **Coordination failures** | Loops, deadlocks, cap breaches, duplicated work | **code** |

Four things to notice:

- **"Grade the outcome, not the path" still holds — and routing accuracy is not a violation of it.** You are not scoring *"did it call agent B before agent C."* You are scoring *"was the agent it chose **capable** of the request."* That's a capability check, not a prescribed sequence. A team that solves a task by an unexpected route still passes; a team that hands a billing question to an agent with no billing tools fails, even if it somehow guessed right.
- **Per-agent success is the new metric, and it's the reason this assignment exists.** "The system passes 70%" is unactionable. "Researcher 95%, critic 55%" tells you what to fix tomorrow morning. Score each agent **given what it received** — a worker that produced a perfect answer to a corrupted handoff has succeeded at its job and exposed a failure in the layer above it.
- **Coordination failures don't exist in a single-agent system**, so there is no baseline to compare them to. Report them as absolute counts per 100 runs. This is the tax column.
- **Latency is now a sum, not a max.** Your p95 will be worse than Assignment 4's and you should expect it. The interesting question is whether the multi-hop or cross-domain slice bought back enough success to justify it.

---

## Task 1 — The baseline, and the task set

**Goal:** fix the target before you build, and make sure the comparison can't flatter you.

### 1. Freeze the Assignment 4 agent

Your single agent is the thing to beat. Pin it: same tools, same descriptions, same system prompt, same model, same caps. **Record the commit hash.** Do not improve it while you build the team — every number in Task 6 depends on the baseline holding still.

If you made changes in Assignment 4's Task 6 that you kept, freeze the **final** version. Beating a deliberately weakened soloist is not a result.

### 2. Extend the task set

Start from your Assignment 4 tasks — they carry over unchanged, which is the point. Add **10–15 new tasks** designed for a team, so you land at **30–45 total**.

For each task record:

| Field | What it holds |
|---|---|
| `task` | the request, phrased the way a real user would |
| `success_criteria` | a **checkable predicate** (same rules as Assignment 4) |
| `reference_answer` | the correct answer, written by you |
| `type` | see the composition below |
| `expected_agents` | which agents you *think* it needs — **debugging metadata, never scored** |
| `capable_agents` | which agents *could* legitimately serve it — **this one is scored**, as routing accuracy |
| `answerable` | `True` / `False` |

**Required composition** — the new types are where this assignment lives:

- **8+ `cross_domain`** — genuinely need **two or more agents**, with the second agent depending on the first's output. *"Find the refund window in the 2026 policy, then tell me whether order #4471 still qualifies."* These are the tasks the team should win.
- **4 `misroute_bait`** — phrased so they **sound** like agent A's domain but actually need agent B. *"Why is my invoice so confusing?"* looks like billing, is really documentation. This is your routing-accuracy detector, and it will catch more orchestrators than you expect.
- **3 `no_tool`** — answerable with no agent dispatch at all (*"what can you help me with?"*). Success = the orchestrator answers directly, **zero worker turns**. This is your coordination-overhead detector.
- **2 `handoff_stress`** — the request carries a constraint that must survive the handoff. *"Summarise the policy — **in Hebrew**, and under 50 words."* Success requires the constraint to reach the last agent intact. This is how you catch lossy payloads before they catch you.
- **2 `unanswerable`** — success is a clean refusal after a **bounded** search, not three agents taking turns hunting.
- The rest carry over from Assignment 4 as your control group.

### 3. Success criteria stay code-checkable

Same rule as Assignment 4: prefer predicates a program can decide.

- ✅ `answer contains "₪5,180.20" (±0.01)` · `worker_turns == 0` · `refused == True` · `answer word_count <= 50` · `language == "he"`
- ❌ `the agents collaborated well`

> **`capable_agents` vs. `expected_agents` is the distinction to get right.** `expected_agents` is your guess at the route — record it, never score it, and use it to read failures fast. `capable_agents` is the **set** of agents that could legitimately handle the request — score against membership in that set. A route you didn't predict but that lands inside the capable set is a **pass**.

✅ **Done when:** 30–45 tasks saved as CSV/JSON with the required type mix, both agent fields recorded, and the Assignment 4 baseline commit hash frozen.

---

## Task 2 — Design the team

**Goal:** make the split defensible before you make it work.

### 1. Write a scope contract per agent

**At least 3 agents** with genuinely distinct roles. Researcher / Writer / Critic and Planner / Executor / Validator are fine starting points, but a split that follows **your wall** is better than one that follows job titles. If your wall was tool overload, split by **tool domain**. If it was context bloat, split by **corpus**.

For each agent, one table row:

| Field | Rule |
|---|---|
| **Name** | what it is, not what it does vaguely |
| **One-sentence scope** | 🧪 **it must not contain the word "and"** |
| **Inputs** | what it needs to receive to do its job |
| **Outputs** | what it returns, as a shape |
| **Tools** | **≤ 5 each** — above that you've rebuilt wall #1 inside a worker |
| **Model** | start Haiku everywhere |

The "no *and*" test is not a gimmick. An agent you can only describe as *"searches the docs **and** does the maths **and** formats the reply"* is a monolith wearing three hats, and it will show up in Task 6 as a worker with 60% per-agent success and no diagnosable cause.

### 2. Choose a topology, and justify it in two sentences

| Topology | Use when | Scale |
|---|---|---|
| **Orchestrator–Worker** ⭐ | Almost always. Start here. | 2–20 |
| **Swarm** | Latency matters; domains are clean | 2–8 |
| **Network** | Exploration, debate, critique | ≤ 5 |
| **Hierarchical** | Many domains, many teams | 50+ |

**The supervisor/orchestrator pattern is required** for your primary system — it's what the syllabus asks for and it's the one whose failures you can actually read. If you want a swarm, build it as the Task 8 bonus and compare.

> **Hierarchical is out of scope for a three-agent project.** If you find yourself writing a router that routes to routers, you've bought two extra LLM calls per turn for nothing.

✅ **Done when:** three or more scope contracts that pass the "no *and*" test, a topology choice with a two-sentence justification tied to your wall, and a diagram of the team (hand-drawn is fine).

---

## Task 3 — The contract layer

**Goal:** decide how information moves *before* you write the agents, because retrofitting this is the misery you were warned about.

### 1. Shared state or message passing? Pick one and write down why

| | Shared state | Message passing |
|---|---|---|
| **How** | one state object every agent reads/writes | each handoff carries an explicit payload |
| **Wins** | nothing gets lost; easy to inspect | contexts stay small; boundaries are real |
| **Loses** | state grows into the context bloat you were escaping | anything not in the payload is gone |

Most LangGraph implementations end up **hybrid**: a small shared state (`messages`, `last_active`, `task_id`) plus explicit per-handoff payloads. That's a fine answer — but it has to be a **decision you wrote down**, not an accident of the tutorial you copied.

### 2. Write the handoff schema as a type

Not a dict. A `TypedDict`, dataclass or Pydantic model, with three parts:

```python
class Handoff(BaseModel):
    destination: Literal["researcher", "analyst", "writer"]   # who takes over
    payload: HandoffPayload                                    # what travels
    reason: str                                                # why — for your traces
```

And the payload is where the design work actually is:

```python
class HandoffPayload(BaseModel):
    summary: str                        # NOT the whole conversation
    constraints: list[str]              # "under 50 words", "in Hebrew" — these must survive
    facts: dict[str, str]               # what the previous agent established
    open_question: str                  # what the receiver is being asked to do
```

- Pass the **whole history** and you've re-bloated every context, defeating the entire split.
- Pass **too little** and the receiver re-asks the user something they already answered.
- The usual right answer is a **structured summary plus the specific fields the next agent needs** — a schema you design, and then test with your two `handoff_stress` tasks.

### 3. Make ownership explicit and single-valued

Exactly one agent owns the conversation at any moment. Store it (`last_active`), log every transition, and make "nobody owns it" an **impossible state** rather than a silent one.

> ⚠️ **The deadlock you will hit:** agent A hands to B assuming B will answer; B hands back assuming A wanted a sub-result. Neither replies. To the user this presents as **silence**, and in your logs it presents as nothing at all — which is why ownership has to be a recorded field, not an implicit convention.

✅ **Done when:** a written state decision, a typed handoff schema with an explicit payload type, and ownership stored and logged on every transition.

---

## Task 4 — Build it

**Goal:** get it running, and make sure it can never run away or fail invisibly.

### 1. The orchestrator

Its prompt does **one** job: classify the request and dispatch. It describes each worker in a sentence or two and nothing else.

- ❌ No business logic. No special cases. No "if the user mentions refunds, first check whether…"
- ✅ It may answer `no_tool` requests directly — that's your fast path, and it's how you avoid paying a full dispatch to say hello.

> 🪱 **Supervisor drift is the six-month failure and it starts on day one.** Every special case you add to the router is a piece of a worker's job living in the wrong place. **Watch the line count of the router prompt** — if it keeps growing, that's the alarm, and the fix is to push the logic down into a worker.

### 2. The workers

Each gets its scope contract as a system prompt, ≤ 5 tools, and **no ability to call another worker**. Workers return to the orchestrator. That invariant is what gives you one trace to read instead of three.

Carry your Assignment 4 tools over unchanged, including the **failure contract** — every tool returns an explanatory string on failure, never raises, never returns `None`.

### 3. Wire four safety nets — before your first full run

| Net | Behaviour on breach |
|---|---|
| **Max agent turns** (start at 6–8) | clean recorded failure: "could not complete within the turn limit" |
| **Token budget** per task | same |
| **Wall-clock timeout** | same |
| **Loop detection** 🆕 | the same handoff pair twice in a row → **stop** |

The fourth one is new and it is the one that protects your wallet. Two agents can pass work back and forth indefinitely, each convinced it's making progress, and neither the iteration cap nor the timeout will fire quickly. Every breach is a **recorded outcome** with a reason code — never a crash, never a silent truncation.

### 4. Trace with a unit-of-work object

Every run writes JSONL. One line per step, and the `agent` field is what makes it readable:

```json
{"task_id": "t31", "run": 2, "seq": 4, "agent": "analyst", "event": "tool_call",
 "tool": "calc_total", "input": {"price": 4390, "vat": 0.18},
 "output": "5180.20", "owner": "analyst", "duration_ms": 210,
 "input_tokens": 2840, "output_tokens": 74}

{"task_id": "t31", "run": 2, "seq": 5, "agent": "orchestrator", "event": "handoff",
 "from": "analyst", "to": "writer", "reason": "figures ready, needs prose",
 "payload_keys": ["summary", "constraints", "facts"], "owner": "writer"}
```

Plus one summary line per run: total agent turns, per-agent turns and tokens, total wall-clock, terminal state (`answered` / `refused` / `cap_breached` / `loop_detected` / `error`), and the **ordered route**.

> **Build this before the third agent, not after it.** Attribution is the hard problem here: the user sees one wrong answer, and the cause could be a misroute, a worker's bad tool call, a lossy handoff, or a synthesis step that dropped the right answer on the floor. Without a per-agent trace you cannot tell those apart, so you end up editing prompts at random and calling it debugging.

✅ **Done when:** the orchestrator dispatches, workers run in isolation, all four nets fire cleanly and are counted with reason codes, and JSONL traces carry `agent`, `owner` and handoff events.

---

## Task 5 — Give the team a memory

**Goal:** the cheapest memory that works, done properly — and an honest look at whether you need more.

### 1. Procedural memory — mandatory, and it's just a file

Write an `AGENTS.md` (or `TEAM.md`) that encodes what the system should never have to re-derive: the domain, the corpus boundaries, the house rules, the output conventions, the things it got wrong last week. Load it into the orchestrator and each worker's system prompt.

This is the highest-leverage memory type and it needs **no infrastructure at all**. It's reviewable, diffable, version-controlled and editable by a human who disagrees with it — none of which is true of a fact buried in a vector store.

**Measure it.** Run your task set with and without the file loaded, and report the delta on success and tokens. If it changes nothing, that's a finding too — and a much cheaper one to discover than the alternative.

### 2. Anything more is optional, and needs a plan first

If you add **semantic memory** (facts and preferences promoted out of past runs), you must also submit the curation plan **before** the first write: what gets promoted, by whom, when it expires, how contradictions are resolved, and how a user deletes it. `mem0` is a reasonable starting point.

> ⚠️ **Do not add a memory store without curation.** A confidently stale fact is worse than no memory at all, and an uncurated store becomes a junk drawer by the third week. *"I considered semantic memory, here's why it wasn't justified for this task"* is a **complete** answer to this task.

✅ **Done when:** a procedural memory file loaded by every agent, plus a with/without measurement on success and tokens.

---

## Task 6 — Evaluate: one agent vs. the team

**Goal:** the head-to-head, run properly — and then the parts it hides.

### 1. Run the matrix

Every task, on **both** configurations, **5 times each**:

- **A:** your frozen Assignment 4 single agent
- **B:** your multi-agent system

Where a task is structurally meaningless for a config, record `n/a` explicitly — don't silently drop rows.

### 2. Score it

- **Task success** — code-checked against `success_criteria` where possible; Sonnet judge otherwise, using the Assignment 2 pattern (rubric in the prompt, Pydantic schema, `explanation` **before** `verdict`).
- **Faithfulness** — judge, given the answer **and the tool outputs from the trace**.
- **Per-agent success** — for each agent that took a turn: did it do its job **given what it received**? Judge, with the handoff payload in the prompt.
- **Routing accuracy** — code: was the first dispatched agent in `capable_agents`?
- **Handoff correctness** — code where checkable (did the `constraints` list survive to the last agent?), judge otherwise.
- **Cost, latency, agent turns, tool calls, breaches, loops** — straight from the traces, no judge required.

### 3. The tables

**Table 1 — head to head.** Single vs. multi, per metric, **sliced by task type** (`single` / `cross_domain` / `misroute_bait` / `no_tool` / `handoff_stress` / `unanswerable`).

Report success as a rate over 5 runs (`4/5`), not a boolean. Report latency as **p50 and p95** — a team's latency has a long right tail by construction, and the mean flatters it badly.

**Table 2 — per-agent attribution.** One row per agent: turns taken, per-agent success rate, tokens, mean duration, and the failures attributed to it. This table is the reason you split the system, and it should point at exactly one agent to fix first.

### 4. Now the interesting part

Six things you are **required** to find and write up, each with the trace pasted in:

- **a. A task the team won that the soloist couldn't.** Paste the full route: the dispatch, both worker trajectories, the handoff, the synthesis. Mark the moment the second agent used something the first produced. This is the thesis of the assignment in one example.
- **b. A task the team lost.** Slower, more expensive, or wrong. The strongest version is a `no_tool` task where the orchestrator dispatched a worker anyway and paid two calls to say hello.
- **c. A misroute.** One of your `misroute_bait` tasks that went to the wrong agent. Diagnose it: was it the router's prompt, or the *worker's own description* that misled the router? (It's usually the second, and that's slide 41's prompt rot arriving early.)
- **d. A lossy handoff.** Take a `handoff_stress` task and find where the constraint got dropped. Paste the payload as it was actually sent. Was the schema wrong, or did the agent just not fill it in?
- **e. Variance across the 5 runs.** Same input, different route or different outcome. Diff the traces and find the first divergence. This is the number that decides whether you could ship this.
- **f. The cost of coordination, as a number.** Team ÷ soloist for tokens, agent turns and p95 latency. One sentence: what did that buy you, and on which slice?

> **Read the trace before you theorise.** Every claim in (a)–(f) must be backed by trace lines pasted into the write-up. *"The agents got confused"* is not analysis. *"Here is seq 7, where the researcher's payload had an empty `constraints` list and the writer produced 300 words"* is.

✅ **Done when:** both tables, sliced and over 5 runs, plus (a)–(f) with traces as evidence.

---

## Task 7 — Two failure modes, hunted and fixed

**Goal:** the syllabus requires two documented failure modes. This is where you earn them properly — by reproducing, mitigating and **re-measuring**, not by listing what could theoretically go wrong.

Pick **two** you actually hit (you will hit at least two):

| Failure mode | What it looks like in your traces |
|---|---|
| **Handoff loop** | the same agent pair alternating; `loop_detected` breaches |
| **Deadlock** | ownership transitions stop; terminal state never reached; the run just ends |
| **Context bloat** | per-agent input tokens climbing turn over turn; quality falling late in a run |
| **Hallucinated handoff** | dispatch to an agent that can't serve the request, or doesn't exist |
| **Duplicated work** | two agents independently making the same tool call |
| **Lost in handoff** | a constraint or fact present at seq 3 and absent at seq 8 |
| **Over-dispatch** | worker turns on `no_tool` tasks |

For each of the two, document four things:

1. **Reproduction** — the task, the seed conditions, and how reliably it fires (`3/5 runs`).
2. **Diagnosis** — the trace excerpt, with the exact step where it went wrong.
3. **Mitigation** — what you changed. Follow the diagnostic order: descriptions → deterministic logic → nets → context → model. **At least one of your two fixes must not be a model swap.**
4. **Re-measurement** — the same task set, re-run, with the delta. Including the case where the fix made something else worse.

> **Change one thing at a time.** Rewrite the router prompt *and* tighten the payload schema in one go and you've learned nothing about either.
>
> **Watch the noise floor.** With 35 tasks × 5 runs you're reading a rate. If a fix moves success from 78% to 81%, that's one task changing its mind — say so rather than claiming a win.

✅ **Done when:** two failure modes, each with reproduction rate, trace-backed diagnosis, a mitigation, and a re-measured delta.

---

## What to submit

1. **Your code**, committed and pushed:
   - `team.py` — orchestrator, workers, handoffs, nets, tracing
   - `contracts.py` — the state and handoff schemas (Task 3)
   - `AGENTS.md` — the procedural memory file (Task 5)
   - your evaluation runner (Task 6) and failure-mode harness (Task 7)
   - everything from Assignment 4 still in place and still runnable
2. **Your task set** as CSV/JSON, and your **raw JSONL traces** (or a representative sample).
3. **`assignment_05.xlsx`** — one row per **(task × config × run)**:
   - `task_id`, `task`, `type`, `answerable`, `success_criteria`, `capable_agents`
   - `config` (`single` / `team`), `run` (1–5)
   - `answer`, `success`, `refused`, `terminal_state`
   - `route` (ordered agent list), `agent_turns`, `tool_calls`
   - `routing_correct`, `handoff_correct`, `per_agent_success` (as a dict or one column per agent)
   - `faithfulness` verdict + explanation
   - `latency_ms`, `input_tokens`, `output_tokens`, `breach_reason`
4. **A team diagram** — agents, tools, handoff edges, where state lives.
5. **Three annotated traces**, in full: your best cross-domain win, your worst coordination failure, and one `handoff_stress` run. Mark the step where it went right or wrong.
6. **A write-up** (markdown) containing:
   - **The wall sentence**, at the top, written before you built anything
   - **Task 2–3:** scope contracts, topology justification, state decision, handoff schema
   - **Task 5:** the procedural-memory with/without measurement
   - **Task 6:** Table 1 (sliced, 5 runs), Table 2 (per-agent), and (a)–(f) with traces
   - **Task 7:** two failure modes — reproduction, diagnosis, mitigation, re-measured delta
   - **The verdict paragraph** *(this is the one that carries the assignment)*: **would you ship the team or the soloist?** Name the number that decided it. If the honest answer is *"one agent with better tool descriptions beats both,"* say that — and show the evidence.

---

## The takeaway

You'll finish with a team of agents that routes, delegates, hands off and synthesises. That's the visible result, and it's the smallest part of what you learned.

The durable part is the restraint. Multi-agent is infrastructure: it multiplies cost, latency, failure modes and debugging pain, and it earns that only against a wall you can point at in a table. You now have the instrument to tell — per-agent attribution, routing accuracy, coordination-failure counts — and having the instrument is what separates an engineer from someone who read a blog post about swarms.

Look back at where this started. Assignment 1 was a prompt. Assignment 2 measured it. Assignment 3 grounded it. Assignment 4 let it decide. Assignment 5 gave it colleagues — and the eval suite you wrote in week 2 survived all four transitions **unchanged**, gaining columns each time. That's not five projects. That's one system, five layers deep, measured the same way at every rung.

There is a real chance your team loses to the soloist. **That is not a failed final project** — reported honestly, with the traces to explain it, it's a better submission than a win you can't account for.

**If you don't need a second agent, don't add one.**
