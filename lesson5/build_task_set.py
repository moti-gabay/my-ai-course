"""
build_task_set.py - source of truth for task_set.json / task_set.csv (Assignment 5, real corpus).

Every answerable task cites evidence: {policy, page, quote}, where `quote` is an exact
(whitespace-normalised) excerpt of that page, or of that NFIP section. verify_task_set.py
re-reads each cited page and checks the quote. Cross-domain answers are computed here from
numbers that appear in the quotes, never typed by hand.

Field semantics:
  capable_agents   agents that can legitimately take the FIRST step (routing accuracy is
                   "first dispatched agent in capable_agents"); kept narrow on purpose
  expected_agents  the route I expect; debugging metadata, never scored
  source           lesson3:<eval id> | a4:<task id> (carried over unchanged) | new

Run: .venv/bin/python build_task_set.py
"""

import csv
import json
from pathlib import Path

from tools import calculator

HERE = Path(__file__).resolve().parent


def calc(expression: str) -> float:
    """Expected values come from the same AST calculator the agents use."""
    out = calculator.invoke({"expression": expression})
    assert out.startswith("RESULT: "), out
    return float(out.removeprefix("RESULT: "))


def ev(policy: str, page: str, quote: str) -> dict:
    return {"policy": policy, "page": page, "quote": quote}


# --- evidence reused across tasks -------------------------------------------------------
VA_MIN = ev("auto", "4", "1. $25,000 for each person, subject to $50,000 for each accident, with respect to "
                         "bodily injury; and 2. $20,000 for each accident with respect to property damage.")
BAIL = ev("auto", "4", "Up to $250 for the cost of bail bonds required because of any accident, including related "
                       "traffic law violations. The accident must result in bodily injury or property damage covered "
                       "under this policy.")
EARNINGS = ev("auto", "4", "Up to $200 a day for loss of earnings, but not other income, because of attendance at "
                           "hearings or trails at our request.")
TRAILER = ev("auto", "13", "However, the most we will pay for loss to: 1. Any non-owned auto which is a trailer is $1,500.")
REFUND = ev("auto", "15", "If you cancel, we will refund you 90% of the pro rata unearned premium, computed according to our manuals.")
EMBRACE_WAIT = ev("embrace", "3", "Illness Waiting Period is the fourteen (14) day period of time where the policy’s "
                                  "Coverage is restricted. The Illness Waiting Period starts from the Pet Original Start Date.")
EMBRACE_DENTAL = ev("embrace", "6", "eligible Veterinary Treatment expenses caused by Dental Illness, in excess of the "
                                    "Deductible amount, subject to Reimbursement Percentage requirements and a $1,000 "
                                    "Annual Maximum sub-limit")
NW_DIAG = ev("nationwide", "2", "We will not pay more than seven hundred fifty dollars ($750) in Specialized Diagnostic "
                                "Tests per policy term.")
FL_CLAIM = ev("travel_fl", "20", "You have 90 days from the date of your loss to submit your claim to us, except as "
                                 "otherwise provided by law.")
NFIP_DED = ev("flood", "§ 61.5", "(c) The minimum deductible for policies covering post-FIRM buildings and pre-FIRM "
                                 "buildings charged full risk rates, with building coverage amounts equal to or less than "
                                 "$100,000 is $1,000.")
NFIP_DED_CAP = ev("flood", "§ 61.5", "FEMA must provide policyholders with deductible options in various amounts, up to "
                                     "and including $10,000")
NFIP_MAX = ev("flood", "§ 61.6", "Building Coverage | Single-Family Dwelling | * $35,000 | $250,000.")

HE_ANSWER = "Answer in Hebrew"

TASKS = []


def add(task_id, type_, task, reference_answer, success_criteria, evidence, capable, expected,
        answerable=True, source="new", **extra):
    TASKS.append({"task_id": task_id, "type": type_, "task": task, "answerable": answerable,
                  "success_criteria": success_criteria, "reference_answer": reference_answer,
                  "evidence": evidence, "capable_agents": capable, "expected_agents": expected,
                  "source": source, **extra})


R, A, W, O = "researcher", "analyst", "writer", "orchestrator"

# --- single: lesson3 eval questions (verbatim), a4 carry-overs, answerable-but-sounds-unanswerable ----
add("t01", "single", "What are the minimum limits of liability required by Virginia law under this auto policy?",
    "$25,000 for each person, subject to $50,000 for each accident, for bodily injury; and $20,000 for each "
    "accident for property damage.",
    {"all": [{"contains_amount": 25000}, {"contains_amount": 50000}, {"contains_amount": 20000}]},
    [VA_MIN], [R], [R], source="lesson3:1")
add("t02", "single", "How much will the auto insurer pay for bail bonds after a covered accident?",
    "Up to $250 for the cost of bail bonds required because of an accident, including related traffic law "
    "violations, where the accident results in covered bodily injury or property damage.",
    {"contains_amount": 250}, [BAIL], [R], [R], source="lesson3:2")
add("t03", "single", "What is the most the auto policy will pay for damage to a non-owned trailer?",
    "$1,500.", {"contains_amount": 1500}, [TRAILER], [R], [R], source="lesson3:4")
add("t04", "single", "How long is the illness waiting period on the Embrace pet policy?",
    "Fourteen (14) days, starting from the Pet Original Start Date.",
    {"any": [{"contains_amount": 14}, {"contains_any": ["fourteen"]}]}, [EMBRACE_WAIT], [R], [R], source="lesson3:8")
add("t05", "single", "Is there a separate annual cap on dental illness treatment under the Embrace policy?",
    "Yes. Dental Illness treatment is subject to a $1,000 Annual Maximum sub-limit, in addition to the deductible "
    "and reimbursement percentage.", {"contains_amount": 1000}, [EMBRACE_DENTAL], [R], [R], source="lesson3:9")
add("t06", "single", "How much will the Nationwide pet medical plan pay for specialized diagnostic tests in one policy term?",
    "No more than seven hundred fifty dollars ($750) per policy term.", {"contains_amount": 750}, [NW_DIAG], [R], [R],
    source="lesson3:14")
add("t07", "single", "How long do I have to file a claim under the Allianz Florida travel plan?",
    "90 days from the date of the loss, except as otherwise provided by law.", {"contains_amount": 90}, [FL_CLAIM],
    [R], [R], source="lesson3:20")
add("t08", "single", "On the Allianz UK Premier level of cover, what is the maximum for emergency medical expenses?",
    "£10 million, with a policy excess of £50.",
    {"all": [{"contains_any": ["10 million", "10,000,000", "£10m"]}, {"contains_amount": 50}]},
    [ev("travel_uk", "6", "£10 million £10,000 £20/day up to £300 £75/day up to £750 £2,000 *£50 £50 Nil Nil £50")],
    [R], [R], source="lesson3:24")
add("t09", "single", "How quickly must the health plan answer a fast appeal?",
    "Within 72 hours after receiving the appeal.", {"contains_amount": 72},
    [ev("health", "86", "For fast appeals, we must give you our answer within 72 hours after we receive your appeal.")],
    [R], [R], source="lesson3:28")
add("t10", "single", "What is the minimum flood deductible for a post-FIRM building with $80,000 of building coverage "
                     "charged full-risk rates?",
    "$1,000: the minimum for post-FIRM buildings with building coverage of $100,000 or less.",
    {"contains_amount": 1000}, [NFIP_DED], [R], [R], source="lesson3:33")
add("t11", "single", "My dog tore a cruciate ligament. How long must the policy have been in force before that is "
                     "covered — under Embrace and under Nationwide?",
    "They differ. Embrace treats a cruciate ligament injury occurring before the end of the illness waiting period "
    "or during the first 180 days after the Pet Original Start Date as a pre-existing condition for the life of the "
    "policy. Nationwide excludes cruciate ligament or meniscal damage or rupture that occurs during the first twelve "
    "calendar months the policy is in effect.",
    {"all": [{"contains_amount": 180}, {"any": [{"contains_amount": 12}, {"contains_any": ["twelve"]}]},
             {"judge": "Attributes the 180-day rule to Embrace and the first-twelve-months rule to Nationwide."}]},
    [ev("embrace", "7", "The following Orthopedic conditions that occur before the end of the Illness Waiting Period or "
                        "during the first 180 days after the Pet Original Start Date are excluded and are Pre-existing "
                        "Conditions for the life of the policy: a. Cruciate Ligament Injury;"),
     ev("nationwide", "3", "Diagnosis or treatment of cruciate ligament or meniscal damage or rupture that occurs during "
                           "the first twelve calendar months that this policy is in effect.")],
    [R], [R], source="lesson3:37")
add("t12", "single", "How many days does the insured have to report a new vehicle acquisition?",
    "30 days: an additional vehicle must be added within 30 days after you become the owner.",
    {"contains_amount": 30},
    [ev("auto", "3", "If a newly acquired auto is in addition to any vehicle shown in the Declarations, you must ask us "
                     "to insure the additional vehicle within 30 days after you become the owner.")],
    [R], [R], source="a4:t17")
add("t13", "single", "What constitutes an 'insured person' under Part A - Liability Coverage?",
    "You or any family member for the ownership, maintenance or use of any auto or trailer; any person using your "
    "covered auto; and, for your covered auto, any person or organization for legal responsibility for acts of a "
    "covered person.",
    {"judge": "Names at least: you or a family member, and any person using your covered auto."},
    [ev("auto", "4", "B. Insured as used in this Part means: 1. You or any family member for the ownership, maintenance "
                     "or use of any auto or trailer. 2. Any person using or responsible for the use of your covered auto.")],
    [R], [R], source="a4:t19")
add("t14", "single", "What geographical territory is covered under the policy conditions?",
    "The United States of America, its territories or possessions; Puerto Rico; or Canada, plus transport between "
    "their ports.",
    {"contains_all": ["Puerto Rico", "Canada"]},
    [ev("auto", "14", "The Policy Territory is: 1. The United States of America, its territories or possessions; "
                      "2. Puerto Rico; or 3. Canada.")],
    [R], [R], source="a4:t22")
add("t15", "single", "Does the policy cover towing and labor costs by default?",
    "No. Towing and labor costs are paid only if the Declarations show Towing And Labor Costs Coverage for the auto.",
    {"judge": "Says towing and labor is covered only when the Declarations show that coverage, so not by default."},
    [ev("auto", "11", "If the Declarations indicate that Towing And Labor Costs Coverage is provided for a your covered "
                      "auto, we will pay towing and labor costs")],
    [R], [R], source="a4:t24")
add("t16", "single", "What happens if an accident occurs in a state with higher financial responsibility limits than "
                     "this policy?",
    "The policy provides the higher limit that state's financial responsibility law specifies.",
    {"judge": "Says the policy provides the higher limit required by that state's law."},
    [ev("auto", "6", "A financial responsibility or similar law specifying limits of liability for bodily injury or "
                     "property damage higher than the limit shown in the Declarations, your policy will provide the "
                     "higher specified limit.")],
    [R], [R], source="a4:t25")
add("t17", "single", "I had to miss work to attend a court hearing about my car accident. Does my auto policy pay "
                     "anything for that, or am I on my own?",
    "It pays up to $200 a day for loss of earnings (not other income) for attending hearings or trials at the "
    "insurer's request.",
    {"contains_amount": 200}, [EARNINGS], [R], [R], sounds_unanswerable=True)
add("t18", "single", "Is there any upper limit on how high a deductible I can choose on my flood insurance?",
    "Yes. FEMA offers deductible options up to and including $10,000.",
    {"contains_amount": 10000}, [NFIP_DED_CAP], [R], [R], sounds_unanswerable=True)

# --- cross_domain: a retrieved figure feeds a calculation --------------------------------------------
def cross(task_id, task, ref_fmt, expression, answer_values, evidence, source="new", **extra):
    result = calc(expression)
    add(task_id, "cross_domain", task, ref_fmt.format(result=result),
        {"all": [{"contains_amount": v} for v in answer_values(result)]},
        evidence, [R], [R, A], source=source, calculation={"expression": expression, "result": result}, **extra)


cross("t19", "I was in three separate covered accidents this policy period and needed a bail bond each time. "
             "What is the most my auto policy will pay for bail bonds in total?",
      "Up to $250 per accident, so at most 3 x $250 = ${result:,.0f}.", "250 * 3", lambda r: [r], [BAIL])
cross("t20", "I damaged a trailer I borrowed (I don't own it) and the repair is $2,300. What's the most my auto "
             "policy pays for it, and how much is left for me to cover?",
      "The policy pays at most $1,500 for a non-owned trailer, leaving $2,300 - $1,500 = ${result:,.0f}.",
      "2300 - 1500", lambda r: [1500, r], [TRAILER])
cross("t21", "My vet ran specialized diagnostic tests costing $1,200 this policy term under my Nationwide plan. "
             "How much of that is above what the plan will pay for those tests?",
      "The plan pays no more than $750 per policy term, so $1,200 - $750 = ${result:,.0f} is above the cap.",
      "1200 - 750", lambda r: [r], [NW_DIAG])
cross("t22", "My annual Allstate auto premium is $1,200. If I cancel the policy myself with exactly half of the "
             "policy period left, roughly how much refund should I expect?",
      "Unearned premium is $600 pro rata; if you cancel you get 90% of it: 0.9 x $600 = ${result:,.0f} "
      "(computed according to the insurer's manuals).", "1200 * 0.5 * 0.9", lambda r: [r], [REFUND])
cross("t23", "My dog had two dental illness treatments this policy year, $700 and $600. By how much does that total "
             "exceed Embrace's annual limit for dental illness?",
      "Dental illness has a $1,000 Annual Maximum sub-limit; $700 + $600 = $1,300, which is ${result:,.0f} over it.",
      "700 + 600 - 1000", lambda r: [r], [EMBRACE_DENTAL])
cross("t24", "My single-family home would cost $320,000 to rebuild. Under the NFIP regular program, how much of that "
             "is above the maximum building coverage I can buy?",
      "The regular-program maximum for a single-family dwelling is $250,000, so ${result:,.0f} is above it.",
      "320000 - 250000", lambda r: [r], [NFIP_MAX])
# t25 applies a cap rule on top of arithmetic: each person capped at $25,000, the accident at $50,000.
_t25_uncapped = calc("3 * 25000")
_t25_result = min(_t25_uncapped, 50000.0)
add("t25", "cross_domain", "Assume my auto policy carries Virginia's minimum liability limits. Three people are "
                           "injured in one accident and each has $30,000 of bodily injury damages. What is the most "
                           "the policy pays in total?",
    f"Each person is capped at $25,000 (3 x $25,000 = ${_t25_uncapped:,.0f}), but the per-accident cap is $50,000, "
    f"so the most paid is ${_t25_result:,.0f}.",
    {"contains_amount": _t25_result}, [VA_MIN], [R], [R, A],
    calculation={"expression": "3 * 25000, then capped at 50000 per accident", "result": _t25_result})
cross("t26", "Assume my auto policy carries Virginia's minimum liability limits. I cause one accident that does "
             "$12,000 of damage to one car and $15,000 to another. How much of that property damage is above the "
             "policy's property damage limit?",
      "The minimum property damage limit is $20,000 per accident; $12,000 + $15,000 = $27,000, so ${result:,.0f} is "
      "above it.", "12000 + 15000 - 20000", lambda r: [r], [VA_MIN])

# --- misroute_bait: sounds like one agent's job, needs another ---------------------------------------
add("t27", "misroute_bait", "Calculate my minimum flood deductible: a post-FIRM building with $80,000 of building "
                            "coverage, charged full-risk rates.",
    "$1,000. It is a lookup, not a calculation: the minimum for post-FIRM buildings with coverage of $100,000 or less.",
    {"contains_amount": 1000}, [NFIP_DED], [R], [R], bait="sounds like analyst (calculate), needs researcher")
add("t28", "misroute_bait", "Can you look up in the policy what 18% VAT on a $12,000 repair bill comes to?",
    f"18% of $12,000 is ${calc('12000 * 0.18'):,.0f} (${calc('12000 * 1.18'):,.0f} including VAT). VAT is not a "
    "policy term; this is arithmetic only.",
    {"any": [{"contains_amount": calc("12000 * 0.18")}, {"contains_amount": calc("12000 * 1.18")}]},
    [], [A], [A], bait="sounds like researcher (look up in the policy), needs analyst",
    calculation=[{"expression": "12000 * 0.18", "result": calc("12000 * 0.18")},
                 {"expression": "12000 * 1.18", "result": calc("12000 * 1.18")}])
add("t29", "misroute_bait", "Please format this nicely as bullet points: how long is the illness waiting period on the "
                            "Embrace pet policy?",
    "- Illness waiting period: fourteen (14) days\n- Starts from the Pet Original Start Date",
    {"any": [{"contains_amount": 14}, {"contains_any": ["fourteen"]}]}, [EMBRACE_WAIT], [R], [R, W],
    bait="sounds like writer (formatting), needs researcher first")
add("t30", "misroute_bait", "Work out the most Nationwide will pay for specialized diagnostic tests in one policy term.",
    "$750 per policy term. It is a stated cap, no calculation needed.", {"contains_amount": 750}, [NW_DIAG], [R], [R],
    bait="sounds like analyst (work out), needs researcher")

# --- no_tool: carried over from Assignment 4 unchanged ---------------------------------------------
add("t31", "no_tool", "Hello! What kind of tasks can you help me with today?",
    "A greeting and a short overview of what the assistant can do, answered without any tool or worker.",
    {"no_dispatch": True}, [], [O], [O], source="a4:t07")
add("t32", "no_tool", "What is the general purpose of an insurance policy deductible?",
    "A deductible is the amount the policyholder pays toward a covered loss before the insurer pays; it keeps "
    "small claims off the policy and lowers the premium.",
    {"all": [{"no_dispatch": True},
             {"judge": "Explains that a deductible is the amount the policyholder pays before the insurer pays."}]},
    [], [O], [O], source="a4:t09")
add("t33", "no_tool", "What types of tasks can this multi-agent system help me with?",
    "An overview of what the assistant can do (policy lookups, calculations, formatting), without any tool or worker.",
    {"no_dispatch": True}, [], [O], [O], source="a4:t33")

# --- handoff_stress: the constraint must reach the last agent --------------------------------------
add("t34", "handoff_stress", "What does the Allstate auto policy pay for bail bonds after a covered accident? "
                             "Answer in Hebrew, in under 30 words.",
    "הפוליסה משלמת עד $250 עבור עלות ערבות הנדרשת בעקבות תאונה מכוסה.",
    {"all": [{"language": "he"}, {"max_words": 30}, {"contains_amount": 250}]},
    [BAIL], [R], [R, W], required_constraints=["hebrew", "30"])
add("t35", "handoff_stress", "How long is the illness waiting period on the Embrace pet insurance policy? "
                             "Reply in Hebrew, no more than 30 words.",
    "תקופת ההמתנה למחלה היא 14 ימים, החל ממועד תחילת הביטוח של חיית המחמד.",
    {"all": [{"language": "he"}, {"max_words": 30}, {"contains_amount": 14}]},
    [EMBRACE_WAIT], [R], [R, W], required_constraints=["hebrew", "30"])

# --- unanswerable: a clean refusal after a bounded search ------------------------------------------
add("t36", "unanswerable", "What is the annual premium for the Allstate auto policy for a 35-year-old driver "
                           "garaged in Richmond, Virginia?",
    "Not answerable from the corpus. Premium amounts appear on the Declarations page, which is not part of the "
    "policy form in this corpus.", {"refused": True}, [], [R], [R], answerable=False, source="lesson3:39")
add("t37", "unanswerable", "What is the dollar Annual Maximum on the Embrace pet policy?",
    "Not answerable from the corpus. The Annual Maximum is 'as shown on the Schedule of Insurance', which is not part "
    "of the document; the $1,000 figure is only the Dental Illness sub-limit.",
    {"refused": True}, [], [R], [R], answerable=False, source="lesson3:40")
add("t38", "unanswerable", "What is the interest rate applied to overdue monthly premium payments after 180 days?",
    "Not answerable from the corpus. No policy states an interest rate on overdue premiums; the only interest "
    "clauses concern judgments (Allstate) and late claim payment (Allianz Florida).",
    {"refused": True}, [], [R], [R], answerable=False, source="a4:t35",
    absence_check="corpus-wide scan for 'premium' within 250 chars of interest/late/overdue found only judgment "
                  "interest, the refund clause, and 'insurable interest'")

# --- tool_fails: explicit fault injection; the right behaviour is to decline, not to guess ----------
add("t39", "tool_fails", "Is there a separate annual cap on dental illness treatment under the Embrace policy?",
    "Should decline: document search and page reading are unavailable, so the policy cannot be checked.",
    {"refused": True}, [EMBRACE_DENTAL], [R], [R], source="lesson3:9",
    inject_faults={"search_docs": "error", "read_policy_page": "error"})
add("t40", "tool_fails", "How long do I have to file a claim under the Allianz Florida travel plan?",
    "Should decline: document search and page reading are unavailable, so the policy cannot be checked.",
    {"refused": True}, [FL_CLAIM], [R], [R], source="lesson3:20",
    inject_faults={"search_docs": "error", "read_policy_page": "error"})
add("t41", "tool_fails", "Assume my auto policy carries Virginia's minimum liability limits. Three people are injured "
                         "in one accident and each has $30,000 of bodily injury damages. What is the most the policy "
                         "pays in total?",
    "Should decline: the calculator is unavailable, so the total cannot be computed with the required tool "
    "(the house rules forbid mental arithmetic).", {"refused": True}, [VA_MIN], [R], [R, A],
    inject_faults={"calculator": "error"})


def main() -> None:
    ids = [t["task_id"] for t in TASKS]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    (HERE / "task_set.json").write_text(json.dumps(TASKS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cols = ["task_id", "type", "task", "answerable", "success_criteria", "reference_answer", "capable_agents",
            "expected_agents", "evidence", "required_constraints", "inject_faults", "source"]
    with (HERE / "task_set.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in TASKS:
            w.writerow({c: json.dumps(t[c], ensure_ascii=False) if isinstance(t.get(c), (dict, list)) else t.get(c, "")
                        for c in cols})
    print(f"wrote {len(TASKS)} tasks to task_set.json and task_set.csv")


if __name__ == "__main__":
    main()
