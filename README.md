# LLM Code Interpretation Evaluation

## Project Overview

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

## Project Structure

```
data/       → dataset and scoring guide
prompts/    → LLM prompts
harness/    → evaluation code
results/    → experiment results
```

## Goal

The goal is to determine whether an LLM can be used as a **reliable automated judge** for code interpretations while measuring its agreement with human evaluators and possible biases.
