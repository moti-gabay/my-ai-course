import json
import re
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from tools import ALL_TOOLS

# מיפוי הכלים לפי שם לגישה מהירה
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}

# Prompt המנחה את המודל לעבוד בפורמט ReAct מובנה
REACT_SYSTEM_PROMPT = """You are a helpful assistant operating in a ReAct (Reasoning + Acting) loop.

You have access to the following tools:
{tools_description}

To use a tool, format your response strictly as:
Thought: [Your step-by-step reasoning about what to do next]
Action: [Tool Name]
Action Input: [JSON object or exact string input for the tool]

When you receive the Observation from the tool, continue the thought process.
When you have the final answer, or if the question cannot be answered / requires refusal, format your response as:
Thought: I now have the final answer / I should refuse.
Final Answer: [Your complete response here]

Rules:
1. Do NOT make up information not supported by tool observations.
2. If a question is unanswerable or information is missing after checking tools, provide a clear refusal in the Final Answer.
3. Do NOT call tools if the question is general conversation or does not require tools.
"""


def format_tools_description() -> str:
    """יוצר תיאור קריא של כל הכלים והפרמטרים שלהם עבור ה-Prompt."""
    descriptions = []
    for name, tool_obj in TOOLS_BY_NAME.items():
        desc = f"- {name}: {tool_obj.description}"
        descriptions.append(desc)
    return "\n".join(descriptions)


def parse_action(text: str) -> Tuple[str, str, str]:
    """
    מחלץ מהטקסט של המודל את ה-Thought, Action ו-Action Input.
    מחזיר (action_name, action_input, final_answer).
    """
    if "Final Answer:" in text:
        final_answer = text.split("Final Answer:")[1].strip()
        return "", "", final_answer

    action_match = re.search(r"Action:\s*([^\n]+)", text)
    input_match = re.search(r"Action Input:\s*([^\n]+)", text)

    action = action_match.group(1).strip() if action_match else ""
    action_input = input_match.group(1).strip() if input_match else ""

    return action, action_input, ""


def run_raw_react_loop(
    user_query: str,
    model_name: str = "gpt-4o-mini",
    max_iterations: int = 8
) -> Dict[str, Any]:
    """
    מריץ לולאת ReAct ידנית ב-Python בלבד.
    """
    llm = ChatOpenAI(model=model_name, temperature=0)
    tools_desc = format_tools_description()
    
    # היסטוריית השיחה
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT.format(tools_description=tools_desc)},
        {"role": "user", "content": user_query}
    ]

    trace_log = []
    total_tool_calls = 0

    for iteration in range(1, max_iterations + 1):
        # 1. פנייה ל-LLM
        prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        response = llm.invoke(prompt_text)
        response_text = response.content.strip()

        trace_log.append({
            "iteration": iteration,
            "response": response_text
        })

        # 2. ניתוח התשובה
        action, action_input, final_answer = parse_action(response_text)

        # מקרה קצה: המודל החזיר תשובה סופית
        if final_answer:
            return {
                "status": "success",
                "final_answer": final_answer,
                "iterations": iteration,
                "tool_calls": total_tool_calls,
                "trace": trace_log
            }

        # מקרה קצה: המודל לא החזיר Action תקין
        if not action or action not in TOOLS_BY_NAME:
            # ניסיון תיקון / עצירה אם אין Action
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Observation: Invalid or missing Action format. Please specify Action and Action Input, or Final Answer."
            })
            continue

        # 3. הרצת ה-Tool (Acting)
        total_tool_calls += 1
        tool_obj = TOOLS_BY_NAME[action]
        
        try:
            # ניקוי פרמטרים במידה והוזן JSON
            clean_input = action_input.strip("'\"")
            if clean_input.startswith("{") and clean_input.endswith("}"):
                try:
                    parsed_json = json.loads(clean_input)
                    clean_input = list(parsed_json.values())[0] if parsed_json else clean_input
                except Exception:
                    pass

            observation = tool_obj.invoke(clean_input)
        except Exception as e:
            # Failure Contract - החזרת מחרוזת שגיאה במקום קריסה
            observation = f"ERROR: Tool execution failed with exception: {str(e)}"

        # 4. הוספת ה-Observation להשתלשלות השיחה
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # חריגה ממספר ה-Iterations המקסימלי (Timeout / Guardrail)
    return {
        "status": "max_iterations_exceeded",
        "final_answer": "ERROR: Maximum reasoning steps reached without final answer.",
        "iterations": max_iterations,
        "tool_calls": total_tool_calls,
        "trace": trace_log
    }


if __name__ == "__main__":
    # בדיקה מהירה של הלולאה הידנית
    test_task = "If I have a claim for $12,000 and my deductible is $1,000, what amount will the insurer pay after deductible and 18% VAT?"
    print(f"--- Running Raw ReAct Loop on: {test_task} ---")
    result = run_raw_react_loop(test_task)
    print("\nFinal Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))