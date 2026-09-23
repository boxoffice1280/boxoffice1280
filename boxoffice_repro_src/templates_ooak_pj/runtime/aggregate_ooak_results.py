#!/usr/bin/env python3
"""Aggregate OOAK (or PJ) method-matrix results into a paper-style table.

Reads output_<tag>/boxoffice_s<seed>_<model>.json files produced by
run_boxoffice.py and emits, per R: norm-F1 by warmup configuration
(cell) x method x model, pooled over seeds, as markdown + json.

norm-F1 here = mean method F1 over eval rows in the cell, divided by the
mean baseline F1 over the same rows (the joint filter makes baseline F1
== 1.0 on eval rows for the three paper models, so the denominator is a
safety normalisation, not a re-weighting).

Usage:
  python aggregate_ooak_results.py --out-dir output_ooak_v4_m4 \
      --seeds 7 11 --ratio 0.15 --md-out ooak_results_R015.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

METHOD_LABELS = [
    ("cb_k0", "CB"),
    ("cb_k0q", "CB+Q"),
    ("cb_k5", "FR"),
    ("cb_k5q", "FR+Q"),
    ("ccv3_m1_diffkv", "LM"),
    ("ccv3_m1_q", "LM+Q"),
    ("ccv3_m2_diffkv", "CC-M2"),
    ("ccv3_m2_q", "CC-M2+Q"),
    ("ccv3_m4_diffkv", "CC-M4"),
    ("ccv3_m4_q", "CC-M4+Q"),
]
CELLS = ["<<", "<>", "><", ">>"]
# Model names are the basename of the model path given to run_boxoffice.py
# (the release convention is /data/weights/{llama3.1-8BI,mistral-7B,qwen3-8B}).
MODELS = ["llama3.1-8BI", "mistral-7B", "qwen3-8B"]


def fmt(x):
    return "  . " if x is None else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 11])
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rtag = f"R{args.ratio:.2f}"
    acc = defaultdict(lambda: [0.0, 0.0, 0])  # (model, method, cell) -> [sum_f1, sum_bl, n]
    missing = []
    for model in args.models:
        for seed in args.seeds:
            path = args.out_dir / f"boxoffice_s{seed}_{model}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            data = json.loads(path.read_text())
            for row in data.get("per_query", []):
                if not row.get("is_eval"):
                    continue
                cell = row.get("cell") or ""
                if cell not in CELLS:
                    continue
                bl = (row.get("baseline") or {}).get("f1")
                if bl is None:
                    continue
                for mkey, _ in METHOD_LABELS:
                    rec = row.get(f"{mkey}_{rtag}")
                    if rec is None:
                        continue
                    for ckey in (cell, "AVG"):
                        slot = acc[(model, mkey, ckey)]
                        slot[0] += float(rec.get("f1") or 0.0)
                        slot[1] += float(bl)
                        slot[2] += 1

    def norm(model, mkey, cell):
        s_f1, s_bl, n = acc.get((model, mkey, cell), (0.0, 0.0, 0))
        if n == 0 or s_bl == 0:
            return None
        return s_f1 / s_bl

    lines = []
    lines.append(f"# norm-F1 at R={args.ratio} — seeds {args.seeds}, eval rows only")
    if missing:
        lines.append(f"\nWARNING missing result files: {missing}")
    for model in args.models:
        lines.append(f"\n## {model}")
        header = "| cell | " + " | ".join(label for _, label in METHOD_LABELS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(METHOD_LABELS) + 1))
        for cell in CELLS + ["AVG"]:
            row = [fmt(norm(model, mkey, cell)) for mkey, _ in METHOD_LABELS]
            lines.append(f"| `{cell}` | " + " | ".join(row) + " |")
    md = "\n".join(lines)
    print(md)
    if args.md_out:
        args.md_out.write_text(md + "\n")
    if args.json_out:
        blob = {
            f"{model}|{mkey}|{cell}": norm(model, mkey, cell)
            for model in args.models
            for mkey, _ in METHOD_LABELS
            for cell in CELLS + ["AVG"]
        }
        args.json_out.write_text(json.dumps(blob, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
