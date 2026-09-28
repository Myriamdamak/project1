"""Step 1: LLM interprets each code function.

By default the dataset's existing LLM_interpretation / judge_score are IGNORED:
every interpretation is generated fresh from `language` + `code` only.
Pass --keep-existing to reuse interpretations already in the file.

Usage: python -m harness.interpret [--data data/dataset.jsonl] [--out results/interpretations.jsonl]
"""
import argparse
from functools import partial

from .common import ROOT, generate, load_prompt, parallel_map, read_jsonl, write_jsonl


def interpret_one(item, keep_existing=False):
    if keep_existing and item.get("LLM_interpretation"):
        return item
    prompt = load_prompt("interpret", language=item["language"], code=item["code"])
    item = dict(item)
    item["dataset_interpretation"] = item.get("LLM_interpretation")  # old one, kept for reference only
    item["LLM_interpretation"] = generate(prompt, temperature=0.0)
    item["judge_score"] = None  # never carry over a stale judge score
    return item


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/dataset.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/interpretations.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-existing", action="store_true",
                    help="reuse LLM_interpretation already present in the dataset")
    args = ap.parse_args()

    fn = partial(interpret_one, keep_existing=args.keep_existing)
    rows = parallel_map(fn, read_jsonl(args.data), args.workers)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} interpretations -> {args.out}")