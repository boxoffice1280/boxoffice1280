#!/usr/bin/env python3
"""Generate One-of-a-Kind (genre-singleton) boxoffice datasets.

Warmup schedule + per-cell eval assembly for the OOAK template
(../../OOAK_TEMPLATE_DESIGN.md). Reads the canonical corpus directly so
chunk texts are byte-identical to the max-template release datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

V1_SCRIPTS = Path(__file__).resolve().parents[1] / "v1_helpers"
# Shared max-template helpers (generate_one_vs_all_v1.py) live in the parent
# BoxOffice bundle under <bundle>/experiments/v1_helpers. This file sits in
# <bundle>/templates_ooak_pj/experiments/<stage>; override the bundle root
# with BOXOFFICE_BUNDLE_ROOT if the subtree is relocated.
_BUNDLE_ROOT = Path(os.environ.get("BOXOFFICE_BUNDLE_ROOT", str(Path(__file__).resolve().parents[3])))
for _p in (_BUNDLE_ROOT / "experiments" / "v1_helpers", V1_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from generate_one_vs_all_v1 import CELL_CODES, Film, write_jsonl  # type: ignore  # noqa: E402
from generate_one_of_a_kind_v1 import (  # type: ignore  # noqa: E402
    DEFAULT_OOAK_PROMPT_VARIANT,
    build_fresh_prefix,
    build_mated_prefix,
    cached_ooak_maps,
    genre_class,
    load_chunk_map,
    load_film_map,
    make_ooak_row,
    singleton_of,
    update_first_seen_ooak,
)


@dataclass(frozen=True)
class OoakCandidateRow:
    cell: str
    context: Tuple[Film, ...]
    singleton: Film
    duplicate_labels: Tuple[str, ...]
    topk_candidate_overlap_count: int
    nonwinner_candidate_overlap_count: int
    singleton_topk_mate_count: int
    warm_score: float


def load_warm_profile(path: Path) -> Dict[str, List[str]]:
    """entity_id -> top-k neighbour entity ids (embedding KNN, corpus-wide)."""
    import re

    profile: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ref_text = str(row.get("reference_context") or "")
            match = re.search(r"ENTITY_ID:\s*(FILM-\d+)", ref_text)
            if not match:
                continue
            profile[match.group(1)] = [str(ctx.get("entity_id") or "") for ctx in row.get("top_k_contexts", [])]
    return profile


def pick_support_blocks(films: Sequence[Film], rng: random.Random) -> Tuple[List[Film], List[Film]]:
    """Two 9-film fresh-support blocks (three same-label triples each), disjoint classes.

    Blocks are drawn from the largest labels so small labels keep enough
    films (>=3 per conditioning side) for eval duplicate triples.
    """
    by_label: Dict[str, List[Film]] = defaultdict(list)
    for film in films:
        by_label[film.genre].append(film)
    labels_sorted = sorted(by_label, key=lambda label: len(by_label[label]), reverse=True)
    picked_labels: List[str] = []
    used_classes: Set[str] = set()
    for label in labels_sorted:
        if len(by_label[label]) < 6:  # keep >=3 films per side after removing 3
            continue
        cls = genre_class(label)
        if cls in used_classes:
            continue
        picked_labels.append(label)
        used_classes.add(cls)
        if len(picked_labels) == 6:
            break
    if len(picked_labels) < 6:
        raise RuntimeError(f"Need 6 support labels with >=6 films; got {picked_labels}")
    block_a_labels, block_b_labels = picked_labels[:3], picked_labels[3:6]
    block_a: List[Film] = []
    block_b: List[Film] = []
    for label in block_a_labels:
        block_a.extend(rng.sample(by_label[label], k=3))
    for label in block_b_labels:
        block_b.extend(rng.sample(by_label[label], k=3))
    return block_a, block_b


def build_ooak_warmup_schedule(
    films: Sequence[Film],
    *,
    rng: random.Random,
    k_mates: int,
) -> List[Tuple[str, List[Film]]]:
    """Alternating fresh/mated 10-chunk rows; every row a valid OOAK query.

    Every non-support film becomes the 10th chunk of exactly one row, cached
    fresh (`<`) or mated (`>`). Per label, fresh-seeded targets come first so
    mated rows always find already-seen same-label mates. Support-block films
    keep <9 predecessors, which excludes them from eval pools downstream.
    """
    block_a, block_b = pick_support_blocks(films, rng)
    support_ids = {f.entity_id for f in block_a} | {f.entity_id for f in block_b}
    block_a_classes = {genre_class(f.genre) for f in block_a}
    block_b_classes = {genre_class(f.genre) for f in block_b}

    targets_by_label: Dict[str, List[Film]] = defaultdict(list)
    for film in films:
        if film.entity_id not in support_ids:
            targets_by_label[film.genre].append(film)
    for label in targets_by_label:
        rng.shuffle(targets_by_label[label])

    # Round-robin labels; within each label first half fresh, second half mated.
    plan: List[Tuple[Film, str]] = []
    labels = sorted(targets_by_label, key=lambda lb: len(targets_by_label[lb]), reverse=True)
    per_label_plan: Dict[str, List[Tuple[Film, str]]] = {}
    for label in labels:
        members = targets_by_label[label]
        n_fresh = (len(members) + 1) // 2
        per_label_plan[label] = [(f, "<") for f in members[:n_fresh]] + [(f, ">") for f in members[n_fresh:]]
    idx = 0
    while any(per_label_plan[label] for label in labels):
        label = labels[idx % len(labels)]
        if per_label_plan[label]:
            plan.append(per_label_plan[label].pop(0))
        idx += 1

    schedule: List[Tuple[str, List[Film]]] = []
    seen: List[Film] = []
    seen_ids: Set[str] = set()

    def mark_seen(context: Sequence[Film]) -> None:
        for film in context:
            if film.entity_id not in seen_ids:
                seen_ids.add(film.entity_id)
                seen.append(film)

    deferred: List[Tuple[Film, str]] = []
    queue = list(plan)
    while queue or deferred:
        if not queue:
            queue, deferred = deferred, []
            if not queue:
                break
        target, direction = queue.pop(0)
        target_cls = genre_class(target.genre)
        if direction == "<":
            block = block_a if target_cls not in block_a_classes else block_b
            if target_cls in {genre_class(f.genre) for f in block}:
                raise RuntimeError(f"No fresh-support block avoids class {target_cls}")
            context = list(block) + [target]
            stage = f"warmup_ooak_fresh_{target.entity_id}"
        else:
            built = build_mated_prefix(target=target, pool=seen, rng=rng, k_mates=k_mates)
            if built is None:
                deferred.append((target, direction))
                if not queue and all(d == ">" for _, d in deferred):
                    raise RuntimeError(
                        f"Cannot build mated prefix for {target.entity_id} ({target.genre}); "
                        f"seen labels: {Counter(f.genre for f in seen)}"
                    )
                continue
            prefix, _designated = built
            context = prefix + [target]
            stage = f"warmup_ooak_mated_{target.entity_id}"
        singleton_of(context)  # raises if ill-formed
        schedule.append((stage, context))
        mark_seen(context)
    return schedule


def make_ooak_candidate_pool(
    *,
    films: Sequence[Film],
    first_seen: Dict[str, Dict[str, Any]],
    warm_profile: Dict[str, List[str]],
    per_cell_attempts: int,
    rng: random.Random,
    duplicate_layout: Tuple[int, ...] = (3, 3, 3),
    min_predecessors: int = 9,
) -> Dict[str, List[OoakCandidateRow]]:
    status_by_id, predecessor_count_by_id, _ = cached_ooak_maps(first_seen)
    films_by_id = {film.entity_id: film for film in films}
    pools: Dict[str, List[Film]] = {
        status: [
            films_by_id[eid]
            for eid, value in status_by_id.items()
            if value == status and predecessor_count_by_id.get(eid, 0) >= min_predecessors and eid in films_by_id
        ]
        for status in ["<", ">"]
    }
    by_cell: Dict[str, List[OoakCandidateRow]] = {cell: [] for cell in CELL_CODES}
    seen_keys: Dict[str, Set[Tuple[str, ...]]] = {cell: set() for cell in CELL_CODES}

    for cell in CELL_CODES:
        singleton_status, duplicate_status = cell[0], cell[1]
        singleton_pool = pools[singleton_status]
        duplicate_pool = pools[duplicate_status]
        dup_by_label: Dict[str, List[Film]] = defaultdict(list)
        for film in duplicate_pool:
            dup_by_label[film.genre].append(film)
        attempts = 0
        while attempts < per_cell_attempts:
            attempts += 1
            if not singleton_pool:
                break
            singleton = rng.choice(singleton_pool)
            singleton_cls = genre_class(singleton.genre)
            layout = list(duplicate_layout)
            need_labels = len(layout)
            eligible_labels = [
                label
                for label, members in dup_by_label.items()
                if genre_class(label) != singleton_cls
                and len([f for f in members if f.entity_id != singleton.entity_id]) >= min(layout)
            ]
            chosen_labels: List[str] = []
            used_classes = {singleton_cls}
            rng.shuffle(eligible_labels)
            # Greedy fill largest group first so big labels take big slots.
            layout_sorted = sorted(layout, reverse=True)
            for count in layout_sorted:
                pick = None
                for label in eligible_labels:
                    cls = genre_class(label)
                    if cls in used_classes:
                        continue
                    members = [f for f in dup_by_label[label] if f.entity_id != singleton.entity_id]
                    if len(members) >= count:
                        pick = label
                        break
                if pick is None:
                    break
                chosen_labels.append(pick)
                used_classes.add(genre_class(pick))
            if len(chosen_labels) < need_labels:
                continue
            duplicates: List[Film] = []
            for label, count in zip(chosen_labels, layout_sorted):
                members = [f for f in dup_by_label[label] if f.entity_id != singleton.entity_id]
                duplicates.extend(rng.sample(members, k=count))
            context = [singleton] + duplicates
            rng.shuffle(context)
            key = tuple(sorted(film.entity_id for film in context))
            if key in seen_keys[cell]:
                continue
            seen_keys[cell].add(key)
            try:
                row_singleton = singleton_of(context)
            except ValueError:
                continue
            if row_singleton.entity_id != singleton.entity_id:
                continue

            candidate_ids = {film.entity_id for film in context}
            overlaps = 0
            nonwinner_overlaps = 0
            singleton_mates_topk = 0
            for film in context:
                neighbours = warm_profile.get(film.entity_id, [])
                for nid in neighbours:
                    if nid in candidate_ids:
                        overlaps += 1
                        if film.entity_id != singleton.entity_id:
                            nonwinner_overlaps += 1
                if film.entity_id == singleton.entity_id:
                    singleton_mates_topk = sum(
                        1
                        for nid in neighbours
                        if nid in films_by_id and genre_class(films_by_id[nid].genre) == singleton_cls
                    )
            warm_score = 0.45 * (overlaps / max(1, 5 * len(context))) + 0.55 * (singleton_mates_topk / 5.0)
            by_cell[cell].append(
                OoakCandidateRow(
                    cell=cell,
                    context=tuple(context),
                    singleton=singleton,
                    duplicate_labels=tuple(chosen_labels),
                    topk_candidate_overlap_count=overlaps,
                    nonwinner_candidate_overlap_count=nonwinner_overlaps,
                    singleton_topk_mate_count=singleton_mates_topk,
                    warm_score=float(warm_score),
                )
            )
    return by_cell


def select_balanced_ooak_rows(
    by_cell: Dict[str, List[OoakCandidateRow]],
    *,
    eval_per_cell: int,
    rng: random.Random,
) -> List[OoakCandidateRow]:
    """Bucket-match cells on (nonwinner overlap, singleton top-k mates), then fill."""
    buckets_by_cell: Dict[str, Dict[Tuple[int, int], List[OoakCandidateRow]]] = {}
    for cell, rows in by_cell.items():
        buckets: Dict[Tuple[int, int], List[OoakCandidateRow]] = defaultdict(list)
        for row in rows:
            buckets[(row.nonwinner_candidate_overlap_count, row.singleton_topk_mate_count)].append(row)
        for bucket_rows in buckets.values():
            rng.shuffle(bucket_rows)
            bucket_rows.sort(key=lambda row: (row.warm_score, row.topk_candidate_overlap_count))
        buckets_by_cell[cell] = buckets

    common = set.intersection(*(set(b) for b in buckets_by_cell.values())) if buckets_by_cell else set()
    bucket_order = sorted(common, key=lambda b: (b[0], b[1]))
    selected: Dict[str, List[OoakCandidateRow]] = {cell: [] for cell in CELL_CODES}
    used: Dict[str, Set[Tuple[str, ...]]] = {cell: set() for cell in CELL_CODES}

    changed = True
    while changed and any(len(selected[cell]) < eval_per_cell for cell in CELL_CODES):
        changed = False
        for bucket in bucket_order:
            for cell in CELL_CODES:
                if len(selected[cell]) >= eval_per_cell:
                    continue
                rows = buckets_by_cell[cell].get(bucket, [])
                while rows:
                    row = rows.pop(0)
                    key = tuple(sorted(film.entity_id for film in row.context))
                    if key in used[cell]:
                        continue
                    selected[cell].append(row)
                    used[cell].add(key)
                    changed = True
                    break

    for cell in CELL_CODES:
        if len(selected[cell]) >= eval_per_cell:
            continue
        fallback = sorted(by_cell[cell], key=lambda row: (row.warm_score, row.nonwinner_candidate_overlap_count))
        for row in fallback:
            key = tuple(sorted(film.entity_id for film in row.context))
            if key in used[cell]:
                continue
            selected[cell].append(row)
            used[cell].add(key)
            if len(selected[cell]) >= eval_per_cell:
                break
        if len(selected[cell]) < eval_per_cell:
            raise RuntimeError(f"Could only select {len(selected[cell])}/{eval_per_cell} rows for cell {cell}")

    out = [row for cell in CELL_CODES for row in selected[cell][:eval_per_cell]]
    rng.shuffle(out)
    return out


def add_ooak_v2_metadata(row: Dict[str, Any], cand: OoakCandidateRow, warm_profile: Dict[str, List[str]]) -> None:
    md = row["metadata"]
    md["template"] = "one_of_a_kind_v2"
    md["matrix_scheduler_mode"] = "one_of_a_kind_v2_balanced_warm_topk"
    md["v2_balance_metrics"] = {
        "warm_topk_candidate_overlap_count": cand.topk_candidate_overlap_count,
        "warm_topk_nonwinner_candidate_overlap_count": cand.nonwinner_candidate_overlap_count,
        "singleton_topk_mate_count": cand.singleton_topk_mate_count,
        "duplicate_labels": list(cand.duplicate_labels),
        "warm_score": cand.warm_score,
    }
    md["duplicate_partition_layout"] = sorted(
        (len(v) for k, v in md["genre_label_partition"].items() if k != md["singleton_genre_label"]),
        reverse=True,
    )
    md["warm_topk_profile"] = {
        "retrieval_mode": "embedding",
        "top_k": 5,
        "profile_source": "qwen3_8B warm fusion_docs_topk5",
        "per_entity": {
            film.entity_id: {"topk_entity_ids": warm_profile.get(film.entity_id, [])} for film in cand.context
        },
    }


def generate(
    *,
    source_corpus: Path,
    canonical_corpus: Path,
    warm_fusion_docs: Path,
    seed: int,
    eval_per_cell: int,
    per_cell_attempts: int,
    k_mates: int,
    prompt_variant: str,
    duplicate_layout: Tuple[int, ...] = (3, 3, 3),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    film_by_id = load_film_map(source_corpus)
    chunk_text_by_id, chunk_tokens_by_id = load_chunk_map(canonical_corpus)
    missing = sorted(set(film_by_id) - set(chunk_text_by_id))
    if missing:
        raise RuntimeError(f"{len(missing)} films lack canonical chunks, e.g. {missing[:3]}")
    films = [film_by_id[eid] for eid in sorted(film_by_id)]
    warm_profile = load_warm_profile(warm_fusion_docs)

    first_seen: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    schedule = build_ooak_warmup_schedule(films, rng=rng, k_mates=k_mates)
    for stage, context in schedule:
        row = make_ooak_row(
            trace_index=len(rows),
            context=context,
            chunks_by_id=chunk_text_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            first_seen=first_seen,
            stage=stage,
            scheduled_cell=None,
            guaranteed_start=None,
            prompt_variant=prompt_variant,
        )
        rows.append(row)
        update_first_seen_ooak(row, first_seen)

    by_cell = make_ooak_candidate_pool(
        films=films,
        first_seen=first_seen,
        warm_profile=warm_profile,
        per_cell_attempts=per_cell_attempts,
        rng=random.Random(seed * 17 + 3),
        duplicate_layout=duplicate_layout,
    )
    selected = select_balanced_ooak_rows(
        by_cell,
        eval_per_cell=eval_per_cell,
        rng=random.Random(seed * 19 + 5),
    )
    guaranteed_start = len(rows) + 1
    for row in rows:
        row["metadata"]["guaranteed_pollution_start_query"] = guaranteed_start

    eval_rows: List[Dict[str, Any]] = []
    for cand in selected:
        row = make_ooak_row(
            trace_index=len(rows) + len(eval_rows),
            context=cand.context,
            chunks_by_id=chunk_text_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            first_seen=first_seen,
            stage="guaranteed_eval",
            scheduled_cell=cand.cell,
            guaranteed_start=guaranteed_start,
            prompt_variant=prompt_variant,
        )
        if row["metadata"]["computed_matrix_cell"] != cand.cell:
            raise RuntimeError(
                f"Computed cell mismatch: expected {cand.cell}, got {row['metadata']['computed_matrix_cell']}"
            )
        add_ooak_v2_metadata(row, cand, warm_profile)
        eval_rows.append(row)

    rows.extend(eval_rows)
    start_index_0based = guaranteed_start - 1
    for idx, row in enumerate(rows):
        md = row.setdefault("metadata", {})
        is_eval = idx >= start_index_0based
        md["row_index_0based"] = idx
        md["row_index_1based"] = idx + 1
        md["full_reuse_eval_start_index_0based"] = start_index_0based
        md["full_reuse_eval_start_query_1based"] = guaranteed_start
        md["full_reuse_eval_row_index_0based"] = idx - start_index_0based if is_eval else None
        md["full_reuse_eval_row_index_1based"] = idx - start_index_0based + 1 if is_eval else None
        md["is_full_reuse_failure_eval"] = bool(is_eval)
        if is_eval:
            md["pollution_mode"] = "guaranteed"

    status_by_id, predecessor_count_by_id, mate_count_by_id = cached_ooak_maps(first_seen)
    eval_by_cell = Counter(row["metadata"]["computed_matrix_cell"] for row in eval_rows)
    singleton_labels = Counter(row["metadata"]["singleton_genre_label"] for row in eval_rows)
    manifest = {
        "seed": seed,
        "dataset": "one_of_a_kind_v2",
        "template_family": "film_attribute_one_of_a_kind_singleton",
        "conditioning_semantics": "genre_mate_presence",
        "k_mates": k_mates,
        "duplicate_layout": list(duplicate_layout),
        "num_rows": len(rows),
        "warmup_rows": guaranteed_start - 1,
        "eval_rows": len(eval_rows),
        "eval_per_cell": eval_per_cell,
        "cell_counts": dict(eval_by_cell),
        "status_counts": {
            "<": sum(1 for v in status_by_id.values() if v == "<"),
            ">": sum(1 for v in status_by_id.values() if v == ">"),
        },
        "pool_sizes": {
            status: sum(
                1
                for eid, v in status_by_id.items()
                if v == status and predecessor_count_by_id.get(eid, 0) >= 9
            )
            for status in ["<", ">"]
        },
        "mate_count_histogram": dict(Counter(mate_count_by_id.values())),
        "eval_singleton_label_histogram": dict(singleton_labels),
        "pool_candidates_by_cell": {cell: len(by_cell[cell]) for cell in CELL_CODES},
        "warm_fusion_docs": str(warm_fusion_docs),
        "source_corpus": str(source_corpus),
        "canonical_corpus": str(canonical_corpus),
        "per_cell_attempts": per_cell_attempts,
        "prompt_variant": prompt_variant,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--canonical-corpus", type=Path, required=True)
    parser.add_argument("--warm-fusion-docs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-per-cell", type=int, default=40)
    parser.add_argument("--per-cell-attempts", type=int, default=20000)
    parser.add_argument("--k-mates", type=int, default=1, help="mates preceding a mated-seeded chunk (strength dial)")
    parser.add_argument(
        "--duplicate-partition",
        default="3,3,3",
        help="comma-separated duplicate group sizes over the 9 non-singleton chunks (e.g. 3,3,3 or 5,4)",
    )
    parser.add_argument("--prompt-variant", default=DEFAULT_OOAK_PROMPT_VARIANT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    layout = tuple(int(x) for x in args.duplicate_partition.split(",") if x.strip())
    if sum(layout) != 9 or any(x < 2 for x in layout):
        raise SystemExit(f"--duplicate-partition must sum to 9 with all groups >=2; got {layout}")
    rows, manifest = generate(
        source_corpus=args.source_corpus,
        canonical_corpus=args.canonical_corpus,
        warm_fusion_docs=args.warm_fusion_docs,
        seed=args.seed,
        eval_per_cell=args.eval_per_cell,
        per_cell_attempts=args.per_cell_attempts,
        k_mates=args.k_mates,
        prompt_variant=args.prompt_variant,
        duplicate_layout=layout,
    )
    write_jsonl(args.out, rows)
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
