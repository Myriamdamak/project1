"""Grade code explanations, saving after each item so a run can resume."""
import argparse

from .common import (
    ROOT, generate_json, load_prompt, read_jsonl,
    scoring_guide, write_jsonl,
)


def judge_one(language, code, interpretation):
    prompt = load_prompt(
        "judge",
        scoring_guide=scoring_guide(),
        language=language,
        code=code,
        interpretation=interpretation,
    )
    result = generate_json(prompt, temperature=0.0)
    score = int(result["score"])
    if score not in (0, 1, 2, 3):
        raise ValueError(f"Invalid score: {score}")
    return score, result.get("reasoning", "")


def save(out, rows):
    temporary = out.with_suffix(".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default=str(ROOT / "results/interpretations.jsonl")
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results/judged.jsonl")
    )
    ap.add_argument(
        "--max-new", type=int, default=15,
        help="Maximum new scores this run (default: 15)"
    )
    args = ap.parse_args()

    from pathlib import Path
    out = Path(args.out)
    rows = read_jsonl(args.data)

    # Keep scores from earlier runs, matching items by their ID.
    if out.exists():
        previous = {str(r["id"]): r for r in read_jsonl(out)}
        rows = [
            {**row, **previous.get(str(row["id"]), {})}
            for row in rows
        ]

    completed_now = 0
    for row in rows:
        if row.get("judge_score") is not None:
            continue
        if completed_now >= args.max_new:
            break

        try:
            score, reasoning = judge_one(
                row["language"], row["code"], row["LLM_interpretation"]
            )
        except Exception as error:
            print(f"Stopped at item {row['id']}: {error}")
            break

        row["judge_score"] = score
        row["judge_reasoning"] = reasoning
        save(out, rows)
        completed_now += 1
        print(f"Saved item {row['id']}: score {score}")

    total = sum(r.get("judge_score") is not None for r in rows)
    print(f"Saved {completed_now} new scores; {total}/{len(rows)} scored in {out}")