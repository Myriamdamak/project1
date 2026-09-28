"""Step 3: compare scores. Human A vs B (agreement) and Humans vs Judge (reliability).
Usage: python -m harness.score [--data results/judged.jsonl]
Expects human_A_score, human_B_score, judge_score in each row.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .common import ROOT, read_jsonl

LABELS = [0, 1, 2, 3]


def compare(x, y):
    """Agreement metrics between two score series (ints 0-3)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    x, y = d["x"].astype(int), d["y"].astype(int)
    if len(d) < 2:
        return {"n": len(d)}

    def kappa(**kw):
        try:
            k = cohen_kappa_score(x, y, labels=LABELS, **kw)
            return None if np.isnan(k) else round(float(k), 4)
        except Exception:
            return None

    rho = spearmanr(x, y)[0] if x.nunique() > 1 and y.nunique() > 1 else np.nan
    return {
        "n": int(len(d)),
        "exact_agreement": round(float((x == y).mean()), 4),
        "within_1": round(float(((x - y).abs() <= 1).mean()), 4),
        "mae": round(float((x - y).abs().mean()), 4),
        "cohen_kappa": kappa(),
        "weighted_kappa_linear": kappa(weights="linear"),
        "weighted_kappa_quadratic": kappa(weights="quadratic"),
        "spearman": None if np.isnan(rho) else round(float(rho), 4),
        "mean_diff_x_minus_y": round(float((x - y).mean()), 4),  # >0: x more lenient
        "confusion_matrix_rows_x_cols_y": confusion_matrix(x, y, labels=LABELS).tolist(),
    }


def run(df):
    df = df.copy()
    df["human_mean"] = df[["human_A_score", "human_B_score"]].mean(axis=1)
    agree = df["human_A_score"] == df["human_B_score"]
    consensus = df[agree]

    res = {
        "human_A_vs_human_B": compare(df["human_A_score"], df["human_B_score"]),
        "judge_vs_human_A": compare(df["judge_score"], df["human_A_score"]),
        "judge_vs_human_B": compare(df["judge_score"], df["human_B_score"]),
        "judge_vs_consensus (A==B only)": compare(consensus["judge_score"], consensus["human_A_score"]),
        "n_consensus_items": int(agree.sum()),
    }
    hm = df.dropna(subset=["judge_score", "human_mean"])
    if hm["judge_score"].nunique() > 1 and hm["human_mean"].nunique() > 1:
        res["spearman_judge_vs_human_mean"] = round(float(spearmanr(hm["judge_score"], hm["human_mean"])[0]), 4)

    per_lang = {}
    for lang, g in df.groupby("language"):
        per_lang[lang] = {
            "n": int(len(g)),
            "human_A_vs_B_exact": compare(g["human_A_score"], g["human_B_score"]).get("exact_agreement"),
            "judge_vs_A_exact": compare(g["judge_score"], g["human_A_score"]).get("exact_agreement"),
            "judge_vs_B_exact": compare(g["judge_score"], g["human_B_score"]).get("exact_agreement"),
        }
    res["per_language"] = per_lang
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/judged.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/metrics.json"))
    args = ap.parse_args()

    df = pd.DataFrame(read_jsonl(args.data))
    res = run(df)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nSaved -> {args.out}")
