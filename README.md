# LLM Code Interpretation Evaluation (Gemini)

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


## Project structure

Project1/
├── run_all.py                  ← single command that runs everything
├── README.md
├── requirements.txt
├── data/
│   ├── dataset.jsonl           ← your 160 items 
│   ├── dataset.example.jsonl
│   └── scoring_guide.md        ← the 0–3 rubric the judge uses
├── prompts/
│   ├── interpret.txt           ← prompt for the interpreting LLM
│   ├── judge.txt               ← prompt for the pointwise judge
│   ├── judge_pairwise.txt      ← A/B comparison (position bias test)
│   ├── alt_interpret.txt       ← generates the second interpretation for pairs
│   └── pad_verbose.txt         ← pads an interpretation (verbosity bias test)
├── harness/
│   ├── __init__.py
│   ├── common.py               ← Gemini client, prompt and JSONL helpers
│   ├── interpret.py            ← step 1: LLM interpretation
│   ├── judge.py                ← step 2: LLM judge
│   ├── score.py                ← step 3: agreement metrics
│   └── bias.py                 ← step 4: position and verbosity tests
└── results/                    ← generated outputs
    ├── interpretations.jsonl
    ├── judged.jsonl
    ├── metrics.json
    ├── bias_report.json
    ├── position_raw.jsonl
    └── verbosity_raw.jsonl

## Notes
- Judge runs at temperature 0 and never sees human scores.
- Judge=gemini-3.5-flash 
- Interpreter= gemini-3.5-flash-lite
- Different prompt for the judge than for the interpreter to limit self-preference bias.