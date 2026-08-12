"""
length_scorer.py — deterministic Length scoring (Assignment 2).

The rubric computes Length in code rather than asking the LLM judge, because word
count is exactly countable. The assignment allows this explicitly (Task 5): "Length
is arguably countable in code too — if you'd rather compute it and hand the judge a
word count, say so and justify it."

Justification: a word count is deterministic and free. Sending it to the judge would
add cost and risk the model miscounting, with no upside. So Length never goes to the
judge — this function fills the Length column, and the judge scores only the three
subjective text criteria (Fluency, Grammar, Tone) plus Grounding.

Word count uses len(text.split()) — plain whitespace-split WORDS, not tokenizer
tokens (a tokenizer counts sub-word pieces and would give a different, wrong number).

Rubric bands:
    good : 50-90 words
    ok   : 40-49 or 91-110 words
    bad  : <40 or >110 words
"""

import pandas as pd


def word_count(text: str) -> int:
    """Number of whitespace-separated words — matches the rubric's len(text.split())."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def length_band(text: str) -> str:
    """Return the rubric Length rating ('good' / 'ok' / 'bad') for a description."""
    n = word_count(text)
    if 50 <= n <= 90:
        return "good"
    if 40 <= n <= 49 or 91 <= n <= 110:
        return "ok"
    return "bad"


def score_length_column(input_xlsx: str, output_xlsx: str = None,
                        text_col: str = "generated_description") -> pd.DataFrame:
    """Fill word_count and the Length column for every row in a results file.

    Reads the spreadsheet, computes word_count and Length from `text_col`, writes
    them back, and saves. Returns the DataFrame so you can inspect it.
    """
    df = pd.read_excel(input_xlsx)
    df["word_count"] = df[text_col].apply(word_count)
    df["Length"] = df[text_col].apply(length_band)
    df.to_excel(output_xlsx or input_xlsx, index=False)
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python length_scorer.py <results.xlsx> [output.xlsx]")
        print("Fills the Length column (good/ok/bad) from generated_description.")
        raise SystemExit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else infile
    result = score_length_column(infile, outfile)

    counts = result["Length"].value_counts().to_dict()
    print(f"Scored {len(result)} rows -> {outfile}")
    print(f"Length bands: {counts}")