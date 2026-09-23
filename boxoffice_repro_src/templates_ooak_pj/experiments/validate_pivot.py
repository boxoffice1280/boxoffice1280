#!/usr/bin/env python3
"""Invariant checker for Pivot-Join (shared-DIRECTOR) datasets.

Usage:
  python validate_pivot.py --full <full.jsonl> [--eval <eval.jsonl>]
      [--expect-eval-per-cell 40] [--check-m4]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

CELLS = ["<<", "<>", "><", ">>"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_row(row: Dict[str, Any], errors: List[str]) -> None:
    rid = row.get("trace_id", "?")
    md = row["metadata"]
    directors = {str(k): str(v) for k, v in md["director_labels"].items()}
    ctx_ids = [str(x) for x in md["context_entity_ids"]]
    if len(ctx_ids) != 10 or len(set(ctx_ids)) != 10:
        errors.append(f"{rid}: context is not 10 distinct chunks")
        return
    pivot = str(md["pivot_entity_id"])
    matcher = str(md["matcher_entity_id"])
    if pivot not in ctx_ids or matcher not in ctx_ids or pivot == matcher:
        errors.append(f"{rid}: pivot/matcher not distinct in-context chunks")
        return
    d = directors[pivot]
    sharers = [eid for eid in ctx_ids if eid != pivot and directors[eid] == d]
    if sharers != [matcher]:
        errors.append(f"{rid}: pivot's director sharers {sharers} != [{matcher}]")
        return
    if row.get("answer") != matcher:
        errors.append(f"{rid}: answer={row.get('answer')} but matcher={matcher}")
    m = re.search(r"(FILM-\d+) is one of the valid candidates", row["question"])
    if not m or m.group(1) != pivot:
        errors.append(f"{rid}: question does not name pivot {pivot}")
    ids_line = row["question"].splitlines()[0]
    for eid in ctx_ids:
        if eid not in ids_line:
            errors.append(f"{rid}: {eid} missing from question candidate list")
            break
    distractors = [eid for eid in ctx_ids if eid not in {pivot, matcher}]
    ddirs = [directors[eid] for eid in distractors]
    if len(set(ddirs)) != len(ddirs):
        dupes = [x for x, c in Counter(ddirs).items() if c > 1]
        errors.append(f"{rid}: distractors share director(s) {dupes}")
    if d in ddirs:
        errors.append(f"{rid}: a distractor shares the pivot's director")


def first_seen_prefixes(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[str, List[str]]]:
    out: Dict[str, Tuple[str, List[str]]] = {}
    for row in rows:
        directors = {str(k): str(v) for k, v in row["metadata"]["director_labels"].items()}
        ctx_ids = [str(x) for x in row["metadata"]["context_entity_ids"]]
        for idx, eid in enumerate(ctx_ids):
            if eid not in out:
                out[eid] = (directors[eid], ctx_ids[:idx])
    return out


def status_of(eid: str, fs: Dict[str, Tuple[str, List[str]]]) -> Optional[str]:
    director, preds = fs[eid]
    if not preds:
        return None
    return ">" if any(fs[p][0] == director for p in preds) else "<"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--eval", dest="eval_path", type=Path, default=None)
    parser.add_argument("--expect-eval-per-cell", type=int, default=None)
    parser.add_argument("--check-m4", action="store_true")
    args = parser.parse_args()

    full_rows = load_jsonl(args.full)
    errors: List[str] = []
    warmup_rows = [r for r in full_rows if not r["metadata"].get("is_full_reuse_failure_eval")]
    eval_rows = [r for r in full_rows if r["metadata"].get("is_full_reuse_failure_eval")]

    for row in full_rows:
        check_row(row, errors)

    fs = first_seen_prefixes(warmup_rows)
    cell_counter: Counter = Counter()
    mate_is_pivot_counter: Counter = Counter()
    for row in eval_rows:
        rid = row["trace_id"]
        md = row["metadata"]
        ctx_ids = [str(x) for x in md["context_entity_ids"]]
        pivot, matcher = str(md["pivot_entity_id"]), str(md["matcher_entity_id"])
        unseen = [eid for eid in ctx_ids if eid not in fs]
        if unseen:
            errors.append(f"{rid}: eval chunks never seen in warmup: {unseen}")
            continue
        shallow = [eid for eid in ctx_ids if len(fs[eid][1]) < 9]
        if shallow:
            errors.append(f"{rid}: eval chunks with <9 caching predecessors: {shallow}")
        pair_statuses = {status_of(pivot, fs), status_of(matcher, fs)}
        if len(pair_statuses) != 1:
            errors.append(f"{rid}: pivot/matcher statuses differ: {pair_statuses}")
            continue
        d_statuses = {status_of(eid, fs) for eid in ctx_ids if eid not in {pivot, matcher}}
        if len(d_statuses) != 1:
            errors.append(f"{rid}: distractors have mixed statuses {d_statuses}")
            continue
        recomputed = f"{next(iter(pair_statuses))}{next(iter(d_statuses))}"
        if recomputed != md.get("computed_matrix_cell"):
            errors.append(f"{rid}: recomputed cell {recomputed} != metadata {md.get('computed_matrix_cell')}")
        if md.get("scheduled_matrix_cell") and md["scheduled_matrix_cell"] != md.get("computed_matrix_cell"):
            errors.append(f"{rid}: scheduled != computed cell")
        cell_counter[md.get("computed_matrix_cell")] += 1
        mate_pred = fs[matcher][1]
        mate_is_pivot = pivot in [p for p in mate_pred if fs[p][0] == fs[matcher][0]]
        if bool(md.get("matcher_mate_is_pivot")) != mate_is_pivot:
            errors.append(f"{rid}: matcher_mate_is_pivot metadata mismatch")
        mate_is_pivot_counter[(md.get("computed_matrix_cell"), mate_is_pivot)] += 1

    if args.expect_eval_per_cell is not None:
        for cell in CELLS:
            if cell_counter.get(cell, 0) != args.expect_eval_per_cell:
                errors.append(f"cell {cell}: {cell_counter.get(cell, 0)} eval rows (want {args.expect_eval_per_cell})")

    if args.eval_path is not None:
        eval_file_rows = load_jsonl(args.eval_path)
        if [r["query_id"] for r in eval_file_rows] != [r["query_id"] for r in eval_rows]:
            errors.append("eval file query_ids do not match the eval suffix of full")

    if args.check_m4:
        eval_chunk_ids: Set[str] = {str(eid) for row in eval_rows for eid in row["metadata"]["context_entity_ids"]}
        variants: Dict[str, Dict[str, Set[Tuple[str, ...]]]] = {eid: {"<": set(), ">": set()} for eid in eval_chunk_ids}
        for row in warmup_rows:
            directors = {str(k): str(v) for k, v in row["metadata"]["director_labels"].items()}
            ctx_ids = [str(x) for x in row["metadata"]["context_entity_ids"]]
            for idx, eid in enumerate(ctx_ids):
                if eid not in eval_chunk_ids or idx == 0:
                    continue
                prefix = tuple(ctx_ids[:idx])
                mated = any(directors[p] == directors[eid] for p in prefix)
                variants[eid][">" if mated else "<"].add(prefix)
        for eid, sides in sorted(variants.items()):
            if len(sides["<"]) < 2 or len(sides[">"]) < 2:
                errors.append(f"M4: {eid} has {len(sides['<'])} fresh / {len(sides['>'])} mated variants (need >=2/>=2)")

    print(f"rows: {len(full_rows)} ({len(warmup_rows)} warmup + {len(eval_rows)} eval); cells: {dict(cell_counter)}")
    print("matcher_mate_is_pivot by cell:", {f"{c}|{v}": n for (c, v), n in sorted(mate_is_pivot_counter.items())})
    if errors:
        print(f"\nFAIL — {len(errors)} violations:")
        for err in errors[:60]:
            print("  -", err)
        if len(errors) > 60:
            print(f"  ... and {len(errors) - 60} more")
        raise SystemExit(1)
    print("PASS — all PJ invariants hold")


if __name__ == "__main__":
    main()
