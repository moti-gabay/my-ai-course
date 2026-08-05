# Assignment 1 — Reflections

## Exercise 1b: System Prompt, Temperature & Structured Output
* **Which change had the biggest effect on the output?**
  The **System Prompt** had the most fundamental effect. While `temperature` impacts output variance and JSON schemas control standard syntax, the System Prompt modifies the model's persona, underlying capabilities, and output boundaries.

## Exercise 1c: Grounding & Hallucination
* **Why does forcing a quote and allowing "I don't know" reduce hallucination?**
  Forcing a quote explicitly grounds the model's attention in the document's provided context tokens. Allowing "I can't find that in the document" mitigates the model's inherent training bias to output a completion even when facts are absent.

## Exercise 2: Closed vs. Open Models
* **Closed Model (Claude / OpenAI SDK):**
  * *Easiness & Cost:* High setup speed and superior performance out-of-the-box, but paid per token, reliant on third-party uptime, and sends data externally.
* **Open Model (Hugging Face / Local Qwen):**
  * *Gains & Tradeoffs:* Full data privacy, offline execution, and zero token costs; offset by higher hardware demands, slower speed, and lower quality responses.