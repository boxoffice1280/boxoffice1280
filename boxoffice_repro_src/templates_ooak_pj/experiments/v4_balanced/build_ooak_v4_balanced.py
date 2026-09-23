#!/usr/bin/env python3
"""Balance OOAK warmup so every eval chunk has >=2 fresh and >=2 mated variants (M=4).

Mirror of build_filtered_v4_balanced.py for the one-of-a-kind template:
direction `<` = fresh prefix variant (no class-mate among the 9
predecessors), direction `>` = mated prefix variant (k same-label mates,
default 1). Every appended warmup row is itself a valid OOAK query.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

V1_SCRIPTS = Path(__file__).resolve().parents[1] / "v1_helpers"
# Shared max-template helpers (generate_one_vs_all_v1.py) live in the parent
# BoxOffice bundle under <bundle>/experiments/v1_helpers. This file sits in
# <bundle>/templates_ooak_pj/experiments/<stage>; override the bundle root
# with BOXOFFICE_BUNDLE_ROOT if the subtree is relocated.
_BUNDLE_ROOT = Path(os.environ.get("BOXOFFICE_BUNDLE_ROOT", str(Path(__file__).resolve().parents[3])))
for _p in (_BUNDLE_ROOT / "experiments" / "v1_helpers", V1_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from generate_one_vs_all_v1 import write_jsonl  # type: ignore  # noqa: E402
from generate_one_of_a_kind_v1 import (  # type: ignore  # noqa: E402
    DEFAULT_OOAK_PROMPT_VARIANT,
    build_fresh_prefix,
    build_mated_prefix,
    genre_class,
    load_chunk_map,
    load_film_map,
    make_ooak_row,
    update_first_seen_ooak,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prefix_sets_by_direction(
    rows: Sequence[Dict[str, Any]], target_ids: Set[str]
) -> Dict[str, Dict[str, Set[Tuple[str, ...]]]]:
    """Per target chunk, the distinct warmup prefixes seen, split fresh/mated."""
    out: Dict[str, Dict[str, Set[Tuple[str, ...]]]] = {eid: {"<": set(), ">": set()} for eid in target_ids}
    for row in rows:
        ctx = [str(x) for x in row["metadata"]["context_entity_ids"]]
        genres = row["metadata"]["genre_labels"]
        for idx, eid in enumerate(ctx):
            if eid not in target_ids:
                continue
            prefix = tuple(ctx[:idx])
            if not prefix:
                continue
            target_cls = genre_class(str(genres[eid]))
            mates = sum(1 for x in prefix if genre_class(str(genres[x])) == target_cls)
            out[eid][">" if mates else "<"].add(prefix)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-full", type=Path, required=True)
    parser.add_argument("--source-eval", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--canonical-corpus", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k-mates", type=int, default=1)
    parser.add_argument("--source-tag", type=str, default="ooak_filtered_v2_reference")
    parser.add_argument("--out-full", type=Path, required=True)
    parser.add_argument("--out-eval", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-validation", type=Path, required=True)
    args = parser.parse_args()

    source_full = load_jsonl(args.source_full)
    source_eval = load_jsonl(args.source_eval)
    source_manifest = json.loads(args.source_manifest.read_text())
    warmup_count_orig = len(source_full) - len(source_eval)
    original_warmup = copy.deepcopy(source_full[:warmup_count_orig])
    eval_rows = copy.deepcopy(source_eval)
    target_ids = sorted({str(eid) for row in source_eval for eid in row["metadata"]["context_entity_ids"]})
    target_set = set(target_ids)

    film_by_id = load_film_map(args.source_corpus)
    chunk_text_by_id, chunk_tokens_by_id = load_chunk_map(args.canonical_corpus)
    all_films = [film_by_id[eid] for eid in sorted(film_by_id)]

    prompt_variant = str(source_full[0].get("metadata", {}).get("prompt_variant") or DEFAULT_OOAK_PROMPT_VARIANT)
    dataset_variant = "ooak_filtered_v4_balanced_m4"

    first_seen: Dict[str, Dict[str, Any]] = {}
    for row in original_warmup:
        update_first_seen_ooak(row, first_seen)
    existing = prefix_sets_by_direction(original_warmup, target_set)

    added_rows: List[Dict[str, Any]] = []
    for target_id in target_ids:
        target = film_by_id[target_id]
        plan = {
            "<": max(0, 2 - len(existing[target_id]["<"])),
            ">": max(0, 2 - len(existing[target_id][">"])),
        }
        for direction in ("<", ">"):
            for variant_index in range(plan[direction]):
                prefix_films = None
                designated = None
                for attempt in range(200):
                    rng = random.Random(f"{args.seed}:{target_id}:{direction}:{variant_index}:{attempt}")
                    if direction == "<":
                        built = build_fresh_prefix(target=target, pool=all_films, rng=rng)
                        if built is None:
                            continue
                        candidate_prefix = built
                    else:
                        built = build_mated_prefix(target=target, pool=all_films, rng=rng, k_mates=args.k_mates)
                        if built is None:
                            continue
                        candidate_prefix, designated = built
                    tup = tuple(f.entity_id for f in candidate_prefix)
                    if tup in existing[target_id][direction]:
                        continue
                    prefix_films = candidate_prefix
                    break
                if prefix_films is None:
                    raise RuntimeError(f"Could not build {direction} variant {variant_index} for {target_id}")
                context = prefix_films + [target]
                rebuilt = make_ooak_row(
                    trace_index=warmup_count_orig + len(added_rows),
                    context=context,
                    chunks_by_id=chunk_text_by_id,
                    chunk_tokens_by_id=chunk_tokens_by_id,
                    first_seen=first_seen,
                    stage=f"warmup_ooak_v4_balanced_m4_{direction}_{variant_index:02d}",
                    scheduled_cell=None,
                    guaranteed_start=None,
                    prompt_variant=prompt_variant,
                )
                md = rebuilt["metadata"]
                md["balanced_variant_target_entity_id"] = target_id
                md["balanced_variant_direction"] = direction
                md["balanced_variant_direction_variant_index"] = variant_index + 1
                md["balanced_variant_prefix_entity_ids"] = [f.entity_id for f in prefix_films]
                md["balanced_variant_prefix_size"] = len(prefix_films)
                md["balanced_variant_designated_singleton"] = designated.entity_id if designated else target_id
                md["balanced_variant_builder"] = "build_ooak_v4_balanced"
                added_rows.append(rebuilt)
                existing[target_id][direction].add(tuple(f.entity_id for f in prefix_films))
                update_first_seen_ooak(rebuilt, first_seen)

    full_rows = copy.deepcopy(original_warmup) + added_rows + copy.deepcopy(eval_rows)
    warmup_count = warmup_count_orig + len(added_rows)
    guaranteed_start = warmup_count + 1
    for idx, row in enumerate(full_rows):
        qid = f"ooak_query_{idx:04d}"
        row["query_id"] = qid
        row["trace_id"] = qid
        row["longbench_id"] = f"one_of_a_kind_v1_{idx:04d}"
        md = row.setdefault("metadata", {})
        md["dataset_variant"] = dataset_variant
        md["source_tag"] = args.source_tag
        md["balanced_variant_requirement"] = {"<": 2, ">": 2}
        is_eval = idx >= warmup_count
        md["row_index_0based"] = idx
        md["row_index_1based"] = idx + 1
        md["full_reuse_eval_start_index_0based"] = warmup_count
        md["full_reuse_eval_start_query_1based"] = guaranteed_start
        md["full_reuse_eval_row_index_0based"] = idx - warmup_count if is_eval else None
        md["full_reuse_eval_row_index_1based"] = idx - warmup_count + 1 if is_eval else None
        md["is_full_reuse_failure_eval"] = bool(is_eval)
        md["guaranteed_pollution_start_query"] = guaranteed_start
        md["pollution_mode"] = "guaranteed" if is_eval else "warmup"

    eval_out = copy.deepcopy(full_rows[warmup_count:])

    counts_sets = prefix_sets_by_direction(full_rows[:warmup_count], target_set)
    counts = {eid: {"<": len(counts_sets[eid]["<"]), ">": len(counts_sets[eid][">"])} for eid in target_ids}
    bad = {eid: cnt for eid, cnt in counts.items() if cnt["<"] < 2 or cnt[">"] < 2}
    if bad:
        raise RuntimeError(f"Balanced variant requirement failed (need >=2 per direction): {bad}")

    hist = Counter((cnt["<"], cnt[">"]) for cnt in counts.values())
    validation = {
        "dataset_variant": dataset_variant,
        "source_tag": args.source_tag,
        "seed": args.seed,
        "k_mates": args.k_mates,
        "eval_unique_chunks": len(target_ids),
        "warmup_rows_original": warmup_count_orig,
        "warmup_rows_added": len(added_rows),
        "warmup_rows_total": warmup_count,
        "exact_direction_count_histogram": {f"{k[0]}/{k[1]}": v for k, v in sorted(hist.items())},
        "per_chunk_counts": counts,
    }

    manifest = copy.deepcopy(source_manifest)
    manifest["dataset"] = dataset_variant
    manifest["source_tag"] = args.source_tag
    manifest["num_rows"] = len(full_rows)
    manifest["warmup_rows"] = warmup_count
    manifest["eval_rows"] = len(eval_out)
    manifest["balanced_variant_requirement"] = {"<": 2, ">": 2}
    manifest["balanced_variant_eval_unique_chunks"] = len(target_ids)
    manifest["balanced_variant_warmup_rows_original"] = warmup_count_orig
    manifest["balanced_variant_warmup_rows_added"] = len(added_rows)
    manifest["balanced_variant_exact_direction_count_histogram"] = validation["exact_direction_count_histogram"]

    write_jsonl(args.out_full, full_rows)
    write_jsonl(args.out_eval, eval_out)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.out_validation.parent.mkdir(parents=True, exist_ok=True)
    args.out_validation.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(full_rows)} full rows ({warmup_count} warmup, {len(eval_out)} eval) to {args.out_full}")


if __name__ == "__main__":
    main()
