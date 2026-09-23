import json
import os

def expand_task_set(existing_file="task_set.json", output_file="task_set.json"):
    # 1. טעינת המשימות הקיימות ממטלה 4
    if os.path.exists(existing_file):
        with open(existing_file, "r", encoding="utf-8") as f:
            existing_tasks = json.load(f)
    else:
        existing_tasks = []

    # התאמת שדות המשימות הקיימות למבנה של מטלה 5
    formatted_existing = []
    for task in existing_tasks:
        t_type = task.get("type", "single")
        
        # מיפוי ברירת מחדל לסוכנים מסוגלים ומצופים עבור משימות מטלה 4
        if t_type == "multi_hop":
            capable = ["researcher", "analyst"]
            expected = ["researcher", "analyst", "writer"]
        elif t_type == "no_tool":
            capable = ["orchestrator"]
            expected = ["orchestrator"]
        else:
            capable = ["researcher", "analyst", "writer"]
            expected = ["researcher", "writer"]

        formatted_existing.append({
            "task_id": task["task_id"],
            "task": task["task"],
            "type": t_type,
            "answerable": task.get("answerable", True),
            "success_criteria": task.get("success_criteria", "Answer contains expected keywords"),
            "reference_answer": task.get("reference_answer", ""),
            "expected_agents": expected,
            "capable_agents": capable
        })

    # 2. המשימות החדשות הייעודיות למטלה 5 (10 משימות חדשות)
    new_tasks = [
        # --- Cross Domain (מצריך לפחות 2 סוכנים בטור) ---
        {
            "task_id": "t26",
            "task": "Find the water damage deductible in the policy, then calculate the total payout for a $15,000 claim including 18% VAT.",
            "type": "cross_domain",
            "answerable": True,
            "success_criteria": "answer contains '$15,930' or '15,930'",
            "reference_answer": "Deductible is $1,500. Net claim is $13,500. With 18% VAT ($2,430), total payout is $15,930.",
            "expected_agents": ["researcher", "analyst", "writer"],
            "capable_agents": ["researcher", "analyst"]
        },
        {
            "task_id": "t27",
            "task": "Locate policy clause 99.B for flood coverage and compute the premium surcharge of 5% on a base premium of $2,000.",
            "type": "cross_domain",
            "answerable": True,
            "success_criteria": "answer contains '$2,100' or '2100'",
            "reference_answer": "Clause 99.B covers flood. Base $2,000 + 5% surcharge ($100) equals $2,100.",
            "expected_agents": ["researcher", "analyst", "writer"],
            "capable_agents": ["researcher", "analyst"]
        },
        {
            "task_id": "t28",
            "task": "What is the property damage liability limit in the policy, and how much capacity remains after two claims of $40,000 and $35,000?",
            "type": "cross_domain",
            "answerable": True,
            "success_criteria": "answer contains '$25,000' or '25000'",
            "reference_answer": "Limit is $100,000. Total claims $75,000. Remaining capacity is $25,000.",
            "expected_agents": ["researcher", "analyst", "writer"],
            "capable_agents": ["researcher", "analyst"]
        },

        # --- Misroute Bait (מנוסח כמו תחום אחד אך שייך לתחום אחר) ---
        {
            "task_id": "t29",
            "task": "Why is my premium calculation so confusing? Can you check what the policy says about administrative surcharges?",
            "type": "misroute_bait",
            "answerable": True,
            "success_criteria": "answer contains '5%' or 'surcharge'",
            "reference_answer": "The policy specifies a 5% administrative surcharge under Part D.",
            "expected_agents": ["researcher", "writer"],
            "capable_agents": ["researcher"]
        },
        {
            "task_id": "t30",
            "task": "Can you calculate how friendly your customer support is based on policy clause 12?",
            "type": "misroute_bait",
            "answerable": False,
            "success_criteria": "refused == True or answer contains 'unanswerable' or 'cannot answer'",
            "reference_answer": "The policy does not contain metrics regarding customer support friendliness.",
            "expected_agents": ["orchestrator"],
            "capable_agents": ["orchestrator", "researcher"]
        },

        # --- Handoff Stress (אילוצים שחייבים לשרוד את ההעברה) ---
        {
            "task_id": "t31",
            "task": "Summarise the collision coverage deductible policy — in Hebrew, and under 30 words.",
            "type": "handoff_stress",
            "answerable": True,
            "success_criteria": "language == 'he' and word_count <= 30 and answer contains '1,000' or '1000'",
            "reference_answer": "השתתפות עצמית סטנדרטית לנזק מכיסוי התנגשות הינה $1,000 בהתאם לתנאי הפוליסה.",
            "expected_agents": ["researcher", "writer"],
            "capable_agents": ["researcher", "writer"]
        },
        {
            "task_id": "t32",
            "task": "List the medical expenses coverage limit — formatted strictly as a bulleted list in Hebrew.",
            "type": "handoff_stress",
            "answerable": True,
            "success_criteria": "language == 'he' and answer contains '$25,000' or '25,000'",
            "reference_answer": "* תקרת כיסוי הוצאות רפואיות: $25,000",
            "expected_agents": ["researcher", "writer"],
            "capable_agents": ["researcher", "writer"]
        },

        # --- No Tool (מענה ישיר ע״י Orchestrator, 0 סיבובי עובדים) ---
        {
            "task_id": "t33",
            "task": "What types of tasks can this multi-agent system help me with?",
            "type": "no_tool",
            "answerable": True,
            "success_criteria": "worker_turns == 0 and refused == False",
            "reference_answer": "I can assist with policy lookups, claims calculations, and coverage details.",
            "expected_agents": ["orchestrator"],
            "capable_agents": ["orchestrator"]
        },

        # --- Unanswerable (סירוב מתוכנן ומוגבל) ---
        {
            "task_id": "t34",
            "task": "What is the exact coverage limit for damage caused by volcanic eruptions in Iceland?",
            "type": "unanswerable",
            "answerable": False,
            "success_criteria": "refused == True",
            "reference_answer": "The policy knowledge base does not contain information regarding volcanic eruption coverage in Iceland.",
            "expected_agents": ["researcher"],
            "capable_agents": ["researcher", "orchestrator"]
        },
        {
            "task_id": "t35",
            "task": "What is the interest rate applied to overdue monthly premium payments after 180 days?",
            "type": "unanswerable",
            "answerable": False,
            "success_criteria": "refused == True",
            "reference_answer": "The available policy documents do not specify overdue premium interest rates.",
            "expected_agents": ["researcher"],
            "capable_agents": ["researcher", "orchestrator"]
        }
    ]

    # 3. איחוד ושמירה
    combined_tasks = formatted_existing + new_tasks
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_tasks, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully updated '{output_file}'!")
    print(f"📊 Total Tasks: {len(combined_tasks)} (Existing: {len(formatted_existing)}, New: {len(new_tasks)})")

if __name__ == "__main__":
    expand_task_set()