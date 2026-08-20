# Corpus — insurance policy documents

Seven real, publicly published insurance documents from six different insurers
across five lines of business. Nothing here is synthetic. Re-fetch everything
with `../fetch_corpus.sh`.

| File | Line | Insurer / issuer | Format | Pages | Chars | Notes |
|---|---|---|---|---|---|---|
| `allstate_auto_policy.pdf` | Personal auto | Allstate (Virginia form AU141756) | PDF | 16 | 67 k | Parts A–F, uninsured motorist, state-specific statutory references |
| `embrace_pet_policy.pdf` | Pet | Embrace (form PET50 01/23) | PDF | 17 | 45 k | dense numbered exclusion lists, waiting periods |
| `nationwide_pet_medical_plan.pdf` | Pet | Nationwide (form VS-G-2 02-17) | PDF | 9 | 64 k | sample coverage form, two-column layout |
| `mhbp_health_evidence_of_coverage.pdf` | Health (Medicare Advantage) | Aetna, for MHBP Standard Option, 2024 | PDF | 155 | 418 k | the long, messy one: 12 chapters, big benefits-chart tables |
| `allianz_travel_basic_fl.pdf` | Travel | Allianz Global Assistance (US, form 101-P-FL-02-302) | PDF | 30 | 57 k | Florida certificate wording |
| `allianz_travel_policy_wording.pdf` | Travel | Allianz (UK policy wording, 2020) | PDF | 29 | 126 k | same insurer, different jurisdiction and wording — a deliberate near-duplicate trap for retrieval |
| `nfip_flood_policy_44cfr61.md` | Flood | FEMA / NFIP, 44 CFR Part 61 (eCFR) | **Markdown** | — | 241 k | Standard Flood Insurance Policy: Dwelling Form, General Property Form, RCBAP |

**Total ≈ 1.02 M characters ≈ 254 k tokens** — an order of magnitude past what
belongs in a single prompt, which is the reason retrieval exists.

## Why this corpus

- **Two formats.** Six PDFs plus one Markdown document. The flood policy is
  pulled from the eCFR HTML renderer and converted by `../html_to_md.py`, so the
  second format is a genuinely different parse path, not a converted copy of a
  PDF already in the corpus.
- **Messy by construction.** The 155-page Aetna/MHBP Evidence of Coverage carries
  multi-page benefit tables and running headers; the Nationwide form is a
  two-column sample document. Both are exactly the parsing problem the assignment
  wants felt rather than read about.
- **Exclusions everywhere.** Every policy has a "what is not covered" section,
  which is where the negation question lives — "X is not covered" versus "X is
  covered" is a one-word difference with a large consequence.
- **Exact identifiers.** Form numbers (`PET50 (01/23)`, `VS-G-2 (02-17)`,
  `101-P-FL-02-302`, `AU141756`), CFR section numbers, and chapter numbers —
  the kind of token embeddings smear.
- **Two travel policies from the same insurer.** The Allianz US certificate and
  the Allianz UK wording cover the same line with different terms. A retriever
  that matches on topic alone will confuse them, which is precisely the failure
  mode worth measuring.

## Provenance

All documents are published openly by their issuers or by regulators. URLs are in
`../fetch_corpus.sh`. Documents are committed to the repo as fetched.
`allstate_auto_policy.pdf` is an AES-encrypted PDF (owner password only), so
`pypdf` needs the `cryptography` package installed to read it.
