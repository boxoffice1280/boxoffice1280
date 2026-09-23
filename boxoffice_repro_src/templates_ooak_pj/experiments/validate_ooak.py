#!/usr/bin/env python3
"""Invariant checker for One-of-a-Kind (genre-singleton) datasets.

Usage:
  python validate_ooak.py --full <full.jsonl> [--eval <eval.jsonl>]
      [--expect-eval-per-cell 40] [--check-m4]

Pure python; runs on any host. Exits non-zero on the first violated
invariant class, printing every violation found.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
# Shared max-template helpers live in the parent BoxOffice bundle under
# <bundle>/experiments/v1_helpers; this file sits in
# <bundle>/templates_ooak_pj/experiments. Override with BOXOFFICE_BUNDLE_ROOT.
_BUNDLE_ROOT = Path(os.environ.get("BOXOFFICE_BUNDLE_ROOT", str(Path(__file__).resolve().parents[2])))
for _p in (_BUNDLE_ROOT / "experiments" / "v1_helpers", HERE / "v1_helpers"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from generate_one_of_a_kind_v1 import GENRE_CANON, genre_class  # type: ignore  # noqa: E402

CELLS = ["<<", "<>", "><", ">>"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_genres(row: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in row["metadata"]["genre_labels"].items()}


def check_row_wellformed(row: Dict[str, Any], errors: List[str]) -> Optional[str]:
    """Exactly one label- and class-singleton; answer matches; no label mixing."""
    rid = row.get("trace_id", "?")
    genres = row_genres(row)
    ctx_ids = [str(x) for x in row["metadata"]["context_entity_ids"]]
    if len(ctx_ids) != 10 or len(set(ctx_ids)) != 10:
        errors.append(f"{rid}: context is not 10 distinct chunks")
        return None
    label_counts = Counter(genres[eid] for eid in ctx_ids)
    class_counts = Counter(genre_class(genres[eid]) for eid in ctx_ids)
    label_singles = [eid for eid in ctx_ids if label_counts[genres[eid]] == 1]
    class_singles = [eid for eid in ctx_ids if class_counts[genre_class(genres[eid])] == 1]
    if len(label_singles) != 1:
        errors.append(f"{rid}: {len(label_singles)} label-singletons (want 1)")
        return None
    if len(class_singles) != 1 or class_singles[0] != label_singles[0]:
        errors.append(f"{rid}: class-singleton mismatch ({class_singles} vs {label_singles})")
        return None
    for cls in class_counts:
        labels_of_cls = {genres[eid] for eid in ctx_ids if genre_class(genres[eid]) == cls}
        if len(labels_of_cls) > 1:
            errors.append(f"{rid}: mixes surface labels {sorted(labels_of_cls)} of class {cls}")
            return None
    singleton = label_singles[0]
    if row.get("answer") != singleton:
        errors.append(f"{rid}: answer={row.get('answer')} but singleton={singleton}")
        return None
    if row["metadata"].get("singleton_entity_id") != singleton:
        errors.append(f"{rid}: metadata singleton_entity_id mismatch")
    ids_line = row["question"].splitlines()[0]
    for eid in ctx_ids:
        if eid not in ids_line:
            errors.append(f"{rid}: {eid} missing from question candidate list")
            break
    return singleton


def first_seen_prefixes(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[str, List[str]]]:
    """entity_id -> (genre_label, first-appearance predecessor ids)."""
    out: Dict[str, Tuple[str, List[str]]] = {}
    for row in rows:
        genres = row_genres(row)
        ctx_ids = [str(x) for x in row["metadata"]["context_entity_ids"]]
        for idx, eid in enumerate(ctx_ids):
            if eid not in out:
                out[eid] = (genres[eid], ctx_ids[:idx])
    return out


def status_of(eid: str, fs: Dict[str, Tuple[str, List[str]]]) -> Optional[str]:
    genre, preds = fs[eid]
    if not preds:
        return None
    cls = genre_class(genre)
    mates = sum(1 for p in preds if genre_class(fs[p][0]) == cls)
    return ">" if mates else "<"


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
    boundary_ok = all(
        not r["metadata"].get("is_full_reuse_failure_eval") for r in full_rows[: len(warmup_rows)]
    ) and all(r["metadata"].get("is_full_reuse_failure_eval") for r in full_rows[len(warmup_rows):])
    if not boundary_ok:
        errors.append("warmup/eval rows are interleaved — boundary is broken")

    # 1. Every row is a well-formed OOAK query.
    for row in full_rows:
        check_row_wellformed(row, errors)

    # 2. Conditioning statuses recomputed from scratch match metadata cells.
    fs = first_seen_prefixes(warmup_rows)
    cell_counter: Counter = Counter()
    for row in eval_rows:
        rid = row["trace_id"]
        md = row["metadata"]
        genres = row_genres(row)
        ctx_ids = [str(x) for x in md["context_entity_ids"]]
        singleton = md["singleton_entity_id"]
        unseen = [eid for eid in ctx_ids if eid not in fs]
        if unseen:
            errors.append(f"{rid}: eval chunks never seen in warmup: {unseen}")
            continue
        shallow = [eid for eid in ctx_ids if len(fs[eid][1]) < 9]
        if shallow:
            errors.append(f"{rid}: eval chunks with <9 caching predecessors: {shallow}")
        s_status = status_of(singleton, fs)
        d_statuses = {status_of(eid, fs) for eid in ctx_ids if eid != singleton}
        if len(d_statuses) != 1:
            errors.append(f"{rid}: duplicates have mixed statuses {d_statuses}")
            continue
        recomputed = f"{s_status}{next(iter(d_statuses))}"
        if recomputed != md.get("computed_matrix_cell"):
            errors.append(f"{rid}: recomputed cell {recomputed} != metadata {md.get('computed_matrix_cell')}")
        if md.get("scheduled_matrix_cell") and md["scheduled_matrix_cell"] != md.get("computed_matrix_cell"):
            errors.append(f"{rid}: scheduled {md['scheduled_matrix_cell']} != computed {md['computed_matrix_cell']}")
        cell_counter[md.get("computed_matrix_cell")] += 1

        dup_partition: Dict[str, List[str]] = defaultdict(list)
        for eid in ctx_ids:
            if eid != singleton:
                dup_partition[genres[eid]].append(eid)
        sizes = sorted((len(v) for v in dup_partition.values()), reverse=True)
        declared = md.get("duplicate_partition_layout")
        if declared is not None and sizes != sorted(declared, reverse=True):
            errors.append(f"{rid}: duplicate partition {sizes} != declared {declared}")
        if any(s < 2 for s in sizes) or sum(sizes) != 9:
            errors.append(f"{rid}: invalid duplicate partition {sizes}")

    if args.expect_eval_per_cell is not None:
        for cell in CELLS:
            if cell_counter.get(cell, 0) != args.expect_eval_per_cell:
                errors.append(f"cell {cell}: {cell_counter.get(cell, 0)} eval rows (want {args.expect_eval_per_cell})")

    # 3. Eval file (if given) matches the eval suffix of full row-for-row.
    if args.eval_path is not None:
        eval_file_rows = load_jsonl(args.eval_path)
        if [r["query_id"] for r in eval_file_rows] != [r["query_id"] for r in eval_rows]:
            errors.append("eval file query_ids do not match the eval suffix of full")

    # 4. M=4 balance: >=2 fresh + >=2 mated distinct prefix variants per eval chunk.
    if args.check_m4:
        eval_chunk_ids: Set[str] = {str(eid) for row in eval_rows for eid in row["metadata"]["context_entity_ids"]}
        variants: Dict[str, Dict[str, Set[Tuple[str, ...]]]] = {
            eid: {"<": set(), ">": set()} for eid in eval_chunk_ids
        }
        for row in warmup_rows:
            genres = row_genres(row)
            ctx_ids = [str(x) for x in row["metadata"]["context_entity_ids"]]
            for idx, eid in enumerate(ctx_ids):
                if eid not in eval_chunk_ids or idx == 0:
                    continue
                prefix = tuple(ctx_ids[:idx])
                cls = genre_class(genres[eid])
                mates = sum(1 for p in prefix if genre_class(genres[p]) == cls)
                variants[eid][">" if mates else "<"].add(prefix)
        for eid, sides in sorted(variants.items()):
            if len(sides["<"]) < 2 or len(sides[">"]) < 2:
                errors.append(f"M4: {eid} has {len(sides['<'])} fresh / {len(sides['>'])} mated variants (need >=2/>=2)")

    n_eval, n_warm = len(eval_rows), len(warmup_rows)
    print(f"rows: {len(full_rows)} ({n_warm} warmup + {n_eval} eval); cells: {dict(cell_counter)}")
    print(f"labels in corpus canon map: {GENRE_CANON}")
    if errors:
        print(f"\nFAIL — {len(errors)} violations:")
        for err in errors[:60]:
            print("  -", err)
        if len(errors) > 60:
            print(f"  ... and {len(errors) - 60} more")
        raise SystemExit(1)
    print("PASS — all OOAK invariants hold")


if __name__ == "__main__":
    main()
