"""Bias tests for the LLM judge.

Position bias : pairwise comparison of two interpretations of the same code, run in both
                orders (A,B) and (B,A). A fair judge picks the same *interpretation* both times.
Verbosity bias: rescore a padded version of each interpretation (same claims, more words).
                A fair judge's score should not change.

Usage: python -m harness.bias --data results/judged.jsonl [--test position|verbosity|both]
"""
import argparse
import json

import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .common import (ROOT, generate, generate_json, load_prompt, parallel_map,
                     read_jsonl, write_jsonl)
from .judge import judge_one


# ---------- Position bias ----------
def pairwise(item, a, b):
    prompt = load_prompt("judge_pairwise", language=item["language"], code=item["code"], a=a, b=b)
    try:
        return generate_json(prompt).get("winner", "tie")
    except Exception:
        return "tie"


def position_one(item):
    orig = item["LLM_interpretation"]
    alt = generate(load_prompt("alt_interpret", language=item["language"], code=item["code"]),
                   temperature=1.0)
    r1 = pairwise(item, orig, alt)  # orig in slot A
    r2 = pairwise(item, alt, orig)  # orig in slot B
    w1 = {"A": "orig", "B": "alt", "tie": "tie"}.get(r1, "tie")
    w2 = {"A": "alt", "B": "orig", "tie": "tie"}.get(r2, "tie")
    return {"id": item["id"], "alt": alt, "slot_run1": r1, "slot_run2": r2,
            "winner_run1": w1, "winner_run2": w2}


def position_report(rows):
    df = pd.DataFrame(rows)
    slots = pd.concat([df["slot_run1"], df["slot_run2"]])
    non_tie = slots[slots != "tie"]
    consistent = df["winner_run1"] == df["winner_run2"]
    return {
        "n_pairs": len(df),
        "consistency_rate": round(float(consistent.mean()), 4),  # 1.0 = no position effect
        "first_slot_win_rate_excl_ties": round(float((non_tie == "A").mean()), 4) if len(non_tie) else None,  # 0.5 = fair
        "second_slot_win_rate_excl_ties": round(float((non_tie == "B").mean()), 4) if len(non_tie) else None,
        "tie_rate": round(float((slots == "tie").mean()), 4),
    }


# ---------- Verbosity bias ----------
def verbosity_one(item):
    orig = item["LLM_interpretation"]
    padded = generate(load_prompt("pad_verbose", interpretation=orig), temperature=0.7)
    s_orig, _ = judge_one(item["language"], item["code"], orig)
    s_pad, _ = judge_one(item["language"], item["code"], padded)
    return {"id": item["id"], "score_original": s_orig, "score_padded": s_pad,
            "len_original": len(orig.split()), "len_padded": len(padded.split()),
            "padded_text": padded}


def verbosity_report(rows, judged_df=None):
    df = pd.DataFrame(rows).dropna(subset=["score_original", "score_padded"])
    diff = df["score_padded"] - df["score_original"]
    try:
        p = float(wilcoxon(diff)[1]) if (diff != 0).any() else 1.0
    except ValueError:
        p = 1.0
    rep = {
        "n": len(df),
        "mean_length_ratio": round(float((df["len_padded"] / df["len_original"]).mean()), 2),
        "mean_score_original": round(float(df["score_original"].mean()), 4),
        "mean_score_padded": round(float(df["score_padded"].mean()), 4),
        "mean_score_change": round(float(diff.mean()), 4),  # >0 => verbosity bias
        "pct_scored_higher": round(float((diff > 0).mean()), 4),
        "pct_scored_lower": round(float((diff < 0).mean()), 4),
        "pct_unchanged": round(float((diff == 0).mean()), 4),
        "wilcoxon_p": round(p, 4),
    }
    if judged_df is not None and {"human_A_score", "human_B_score", "judge_score"} <= set(judged_df.columns):
        j = judged_df.copy()
        j["len"] = j["LLM_interpretation"].str.split().str.len()
        j["excess"] = j["judge_score"] - j[["human_A_score", "human_B_score"]].mean(axis=1)
        j = j.dropna(subset=["excess"])
        if j["excess"].nunique() > 1:
            rep["spearman_length_vs_judge_minus_human"] = round(float(spearmanr(j["len"], j["excess"])[0]), 4)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/judged.jsonl"))
    ap.add_argument("--test", choices=["position", "verbosity", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="use first N items (cheaper)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    items = read_jsonl(args.data)[: args.limit]
    report = {}
    if args.test in ("position", "both"):
        rows = parallel_map(position_one, items, args.workers)
        write_jsonl(ROOT / "results/position_raw.jsonl", rows)
        report["position_bias"] = position_report(rows)
    if args.test in ("verbosity", "both"):
        rows = parallel_map(verbosity_one, items, args.workers)
        write_jsonl(ROOT / "results/verbosity_raw.jsonl", rows)
        report["verbosity_bias"] = verbosity_report(rows, pd.DataFrame(items))

    with open(ROOT / "results/bias_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
