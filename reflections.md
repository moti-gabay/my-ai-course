# Assignment 1 — Reflections

## Exercise 1b: System Prompt, Temperature & Structured Output
* **Which change had the biggest effect on the output?**  
  The **System Prompt** had the most dramatic effect on the model's behavior. While `temperature` adjusts randomness and output structure dictates formatting (e.g. JSON), the System Prompt changes the fundamental persona, rules, and scope of how the model answers.

## Exercise 1c: Grounded QA (RAG Basics)
* **Why does forcing a quote and allowing "I don't know" reduce hallucination?**  
  Requiring an explicit quote forces the model to ground its response in context tokens provided directly in the prompt. Permitting a strict exit response like *"I can't find that in the document"* removes the model's underlying bias to generate a plausible-sounding completion when context is missing.

## Exercise 2: Closed vs. Open Models
* **Closed Model (Claude via OpenAI SDK):**  
  * *Ease:* Extremely easy to set up with high-quality, fast responses out of the box.  
  * *Cost:* Requires continuous token payments, API keys, online availability, and sends data to third-party servers.
* **Open Model (Local Qwen via Hugging Face):**  
  * *Gains:* Total control over execution, complete data privacy, offline operation, and zero cost per API call.  
  * *Costs:* Consumes local CPU/RAM resources, runs noticeably slower, and delivers lower general capability compared to cloud models.