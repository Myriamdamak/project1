"""Run the whole pipeline with one command:

    python run_all.py                       # uses data/dataset.jsonl
    python run_all.py --data my.jsonl --workers 8
    python run_all.py --skip-bias           # skip the (extra-cost) bias tests
    python run_all.py --bias-limit 30       # run bias tests on the first 30 items only

Steps: interpret (regenerates; --keep-existing reuses dataset interpretations) -> judge -> score -> bias
"""
import argparse
import json
import time
from functools import partial

import pandas as pd

from harness.bias import position_one, position_report, verbosity_one, verbosity_report
from harness.common import ROOT, parallel_map, read_jsonl, write_jsonl
from harness.interpret import interpret_one
from harness.judge import judge_item
from harness.score import run as score_run

REQUIRED = ["id", "language", "code"]


def banner(msg):
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/dataset.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-bias", action="store_true")
    ap.add_argument("--keep-existing", action="store_true",
                    help="reuse LLM_interpretation already in the dataset (default: regenerate)")
    ap.add_argument("--bias-limit", type=int, default=None, help="only first N items for bias tests")
    args = ap.parse_args()
    res_dir = ROOT / "results"
    t0 = time.time()

    rows = read_jsonl(args.data)
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise SystemExit(f"Row {i} is missing fields: {missing}")
    print(f"Loaded {len(rows)} items from {args.data}")

    banner("1/4  Interpreting code (dataset interpretations/scores ignored)")
    rows = parallel_map(partial(interpret_one, keep_existing=args.keep_existing), rows, args.workers)
    write_jsonl(res_dir / "interpretations.jsonl", rows)

    banner("2/4  LLM judge scoring")
    rows = parallel_map(judge_item, rows, args.workers)
    write_jsonl(res_dir / "judged.jsonl", rows)
    failed = sum(r["judge_score"] is None for r in rows)
    print(f"Judged {len(rows)} items ({failed} failed to parse)")

    banner("3/4  Agreement metrics (human A/B vs. judge)")
    metrics = score_run(pd.DataFrame(rows))
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for key in ["human_A_vs_human_B", "judge_vs_human_A", "judge_vs_human_B"]:
        m = metrics[key]
        print(f"{key:20s} n={m.get('n')}  exact={m.get('exact_agreement')}  "
              f"QWK={m.get('weighted_kappa_quadratic')}  spearman={m.get('spearman')}")

    if args.skip_bias:
        print("\n4/4  Bias tests skipped")
    else:
        banner("4/4  Bias tests (position + verbosity)")
        items = rows[: args.bias_limit]
        pos = parallel_map(position_one, items, args.workers)
        write_jsonl(res_dir / "position_raw.jsonl", pos)
        ver = parallel_map(verbosity_one, items, args.workers)
        write_jsonl(res_dir / "verbosity_raw.jsonl", ver)
        bias = {"position_bias": position_report(pos),
                "verbosity_bias": verbosity_report(ver, pd.DataFrame(items))}
        (res_dir / "bias_report.json").write_text(json.dumps(bias, indent=2))
        print(json.dumps(bias, indent=2))

    banner(f"Done in {time.time() - t0:.0f}s. Outputs in {res_dir}/")


if __name__ == "__main__":
    main()