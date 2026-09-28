
# LLM Code Interpretation Evaluation (Gemini)

# Project Overview

This project evaluates how well an LLM can interpret source-code functions and whether an LLM judge can reliably evaluate those interpretations.

## Workflow

```
Code
  ↓
LLM
  ↓
Interpretation
  ↓
┌──────────────┬──────────────┐
│              │              │
Human A      Human B       LLM Judge
│              │              │
└──────────────┴──────────────┘
               ↓
        Compare the scores
```

## Dataset

The dataset contains **160 code functions** from different programming languages.

Each item contains:

* `id`
* `language`
* `code`
* `LLM_interpretation`
* `human_A_score`
* `human_B_score`
* `judge_score`

## Scoring

Interpretations are scored from **0 to 3**:

| Score | Meaning                                        |
| ----- | ---------------------------------------------- |
| 0     | Incorrect or irrelevant                        |
| 1     | Partially correct, but has major errors        |
| 2     | Mostly correct, with minor errors or omissions |
| 3     | Correct and sufficiently complete              |

Two humans independently score every interpretation.

The LLM judge then scores the same interpretations.

## Evaluation

We compare:

* **Human A vs. Human B** → human agreement
* **Humans vs. LLM Judge** → judge reliability

We will also test the judge for:

* Position bias
* Verbosity bias


>>>>>>> 84f0715aded81cdb8d1c3733a56fff047ff9534b


## Setup
    pip install -r requirements.txt
    set GEMINI_API_KEY and MODEL="gemini-3.5-flash" in .env file

## Data
Put your 160 items in `data/dataset.jsonl` (see `dataset.example.jsonl`).
Fields: id, language, code, LLM_interpretation, human_A_score, human_B_score, judge_score.
Leave `LLM_interpretation` null to have Gemini generate it.

## Run everything with one command (from project root)
    python run_all.py                  # add --skip-bias, --bias-limit 30, --workers 8 as needed

## Or run step by step
    python -m harness.interpret     # -> results/interpretations.jsonl
    python -m harness.judge         # -> results/judged.jsonl
    python -m harness.score         # -> results/metrics.json
    python -m harness.bias          # -> results/bias_report.json


### Project Structure

```text
Project1/
├── run_all.py              ← single command that runs everything
├── README.md
├── requirements.txt
├── data/
│   ├── dataset.jsonl       ← your 160 items
│   ├── dataset.example.jsonl
│   └── scoring_guide.md    ← the 0–3 rubric the judge uses
├── prompts/
│   ├── interpret.txt       ← prompt for the interpreting LLM
│   ├── judge.txt           ← prompt for the pointwise judge
│   ├── judge_pairwise.txt  ← A/B comparison (position bias test)
│   ├── alt_interpret.txt   ← generates the second interpretation for pairs
│   └── pad_verbose.txt     ← pads an interpretation (verbosity bias test)
├── harness/
│   ├── __init__.py
│   ├── common.py           ← Gemini client, prompt and JSONL helpers
│   ├── interpret.py        ← step 1: LLM interpretation
│   ├── judge.py            ← step 2: LLM judge
│   ├── score.py            ← step 3: agreement metrics
│   └── bias.py             ← step 4: position and verbosity tests
└── results/                ← generated outputs
    ├── interpretations.jsonl
    ├── judged.jsonl
    ├── metrics.json
    ├── bias_report.json
    ├── position_raw.jsonl
    └── verbosity_raw.jsonl
```

## Notes
- Judge runs at temperature 0 and never sees human scores.
- Judge=gemini-3.5-flash 
- Interpreter= gemini-3.5-flash-lite
- Different prompt for the judge than for the interpreter to limit self-preference bias.

