"""Step 2: LLM judge scores each interpretation 0-3 (never sees human scores).
Usage: python -m harness.judge [--data results/interpretations.jsonl] [--out results/judged.jsonl]
"""
import argparse

from .common import (ROOT, generate_json, load_prompt, parallel_map, read_jsonl,
                     scoring_guide, write_jsonl)


def judge_one(language, code, interpretation, temperature=0.0):
    """Return (score, reasoning). score is int 0-3 or None if unparseable."""
    prompt = load_prompt("judge", scoring_guide=scoring_guide(), language=language,
                         code=code, interpretation=interpretation)
    try:
        out = generate_json(prompt, temperature=temperature)
        score = int(out["score"])
        return (score if 0 <= score <= 3 else None), out.get("reasoning", "")
    except Exception as e:
        return None, f"parse_error: {e}"


def judge_item(item):
    item = dict(item)
    item["judge_score"], item["judge_reasoning"] = judge_one(
        item["language"], item["code"], item["LLM_interpretation"])
    return item


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results/interpretations.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/judged.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = parallel_map(judge_item, read_jsonl(args.data), args.workers)
    write_jsonl(args.out, rows)
    failed = sum(r["judge_score"] is None for r in rows)
    print(f"Judged {len(rows)} items ({failed} failed) -> {args.out}")
