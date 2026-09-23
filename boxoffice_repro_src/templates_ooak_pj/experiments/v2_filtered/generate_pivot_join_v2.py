#!/usr/bin/env python3
"""Generate Pivot-Join (shared-DIRECTOR multi-hop) boxoffice datasets.

Warmup schedule + per-cell eval assembly for the PJ template
(../../PIVOT_TEMPLATE_DESIGN.md). Reads the canonical corpus directly so
chunk texts are byte-identical to the other templates' datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
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
from generate_one_of_a_kind_v1 import load_chunk_map, load_film_map  # type: ignore  # noqa: E402
from generate_pivot_join_v1 import (  # type: ignore  # noqa: E402
    DEFAULT_PJ_PROMPT_VARIANT,
    build_pj_fresh_prefix,
    build_pj_mated_prefix,
    cached_pj_maps,
    make_pj_row,
    pj_matcher_of,
    update_first_seen_pj,
)


@dataclass(frozen=True)
class PjCandidateRow:
    cell: str
    context: Tuple[Film, ...]
    pivot: Film
    matcher: Film
    matcher_mate_is_pivot: bool
    topk_candidate_overlap_count: int
    matcher_topk_has_pivot: bool
    warm_score: float


def load_warm_topk_ids(path: Path) -> Dict[str, List[str]]:
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


def build_pj_intro_row(films: Sequence[Film], rng: random.Random) -> Tuple[List[Film], Film]:
    """One bootstrap row: a pair from the largest director group + 8 distinct fillers."""
    by_dir: Dict[str, List[Film]] = defaultdict(list)
    for film in films:
        by_dir[film.director].append(film)
    dirs_sorted = sorted(by_dir, key=lambda d: len(by_dir[d]), reverse=True)
    pair_dir = dirs_sorted[0]
    pair = rng.sample(by_dir[pair_dir], k=2)
    filler_dirs = [d for d in dirs_sorted[1:] if by_dir[d]]
    if len(filler_dirs) < 8:
        raise RuntimeError("Need at least 9 director groups for the intro row")
    fillers = [rng.choice(by_dir[d]) for d in filler_dirs[:8]]
    context = pair + fillers
    rng.shuffle(context)
    return context, rng.choice(pair)


def build_pj_warmup_schedule(
    films: Sequence[Film],
    *,
    rng: random.Random,
    k_mates: int,
) -> List[Tuple[str, List[Film], Film]]:
    """(stage, context, pivot) rows; every non-intro film targeted exactly once."""
    intro_context, intro_pivot = build_pj_intro_row(films, rng)
    intro_ids = {f.entity_id for f in intro_context}

    targets_by_dir: Dict[str, List[Film]] = defaultdict(list)
    for film in films:
        if film.entity_id not in intro_ids:
            targets_by_dir[film.director].append(film)
    for d in targets_by_dir:
        rng.shuffle(targets_by_dir[d])

    per_dir_plan: Dict[str, List[Tuple[Film, str]]] = {}
    for d, members in targets_by_dir.items():
        n_fresh = (len(members) + 1) // 2
        per_dir_plan[d] = [(f, "<") for f in members[:n_fresh]] + [(f, ">") for f in members[n_fresh:]]
    dirs = sorted(per_dir_plan, key=lambda d: len(per_dir_plan[d]), reverse=True)
    plan: List[Tuple[Film, str]] = []
    idx = 0
    while any(per_dir_plan[d] for d in dirs):
        d = dirs[idx % len(dirs)]
        if per_dir_plan[d]:
            plan.append(per_dir_plan[d].pop(0))
        idx += 1

    schedule: List[Tuple[str, List[Film], Film]] = [("warmup_pj_intro", intro_context, intro_pivot)]
    seen: List[Film] = list(intro_context)
    seen_ids: Set[str] = set(intro_ids)

    def mark_seen(context: Sequence[Film]) -> None:
        for film in context:
            if film.entity_id not in seen_ids:
                seen_ids.add(film.entity_id)
                seen.append(film)

    queue = list(plan)
    deferred: List[Tuple[Film, str]] = []
    stall = 0
    while queue or deferred:
        if not queue:
            queue, deferred = deferred, []
            stall += 1
            if stall > 3:
                raise RuntimeError(f"Warmup scheduling stalled; remaining: {[(f.entity_id, s) for f, s in queue]}")
        target, direction = queue.pop(0)
        if direction == "<":
            built = build_pj_fresh_prefix(target=target, pool=seen, rng=rng)
            stage = f"warmup_pj_fresh_{target.entity_id}"
        else:
            built = build_pj_mated_prefix(target=target, pool=seen, rng=rng, k_mates=k_mates)
            stage = f"warmup_pj_mated_{target.entity_id}"
        if built is None:
            deferred.append((target, direction))
            continue
        prefix, pivot = built
        context = prefix + [target]
        pj_matcher_of(context, pivot)  # raises if the named pivot lacks a unique sharer
        schedule.append((stage, context, pivot))
        mark_seen(context)
        stall = 0
    return schedule


def make_pj_candidate_pool(
    *,
    films: Sequence[Film],
    first_seen: Dict[str, Dict[str, Any]],
    warm_topk: Dict[str, List[str]],
    per_cell_attempts: int,
    rng: random.Random,
    min_predecessors: int = 9,
) -> Dict[str, List[PjCandidateRow]]:
    status_by_id, predecessor_count_by_id, mate_ids_by_id = cached_pj_maps(first_seen)
    films_by_id = {film.entity_id: film for film in films}
    eligible = {
        eid: status_by_id[eid]
        for eid in status_by_id
        if predecessor_count_by_id.get(eid, 0) >= min_predecessors and eid in films_by_id
    }
    pools: Dict[str, Dict[str, List[Film]]] = {"<": defaultdict(list), ">": defaultdict(list)}
    for eid, status in eligible.items():
        film = films_by_id[eid]
        pools[status][film.director].append(film)

    by_cell: Dict[str, List[PjCandidateRow]] = {cell: [] for cell in CELL_CODES}
    seen_keys: Dict[str, Set[Tuple[str, ...]]] = {cell: set() for cell in CELL_CODES}
    for cell in CELL_CODES:
        pair_status, distractor_status = cell[0], cell[1]
        pair_dirs = [d for d, members in pools[pair_status].items() if len(members) >= 2]
        attempts = 0
        while attempts < per_cell_attempts:
            attempts += 1
            if not pair_dirs:
                break
            d = rng.choice(pair_dirs)
            pivot, matcher = rng.sample(pools[pair_status][d], k=2)
            distractor_dirs = [
                dd for dd, members in pools[distractor_status].items() if dd != d and members
            ]
            if len(distractor_dirs) < 8:
                continue
            distractors = [rng.choice(pools[distractor_status][dd]) for dd in rng.sample(distractor_dirs, k=8)]
            context = [pivot, matcher] + distractors
            rng.shuffle(context)
            key = tuple(sorted(f.entity_id for f in context)) + (pivot.entity_id,)
            if key in seen_keys[cell]:
                continue
            seen_keys[cell].add(key)
            candidate_ids = {f.entity_id for f in context}
            overlaps = sum(
                1 for f in context for nid in warm_topk.get(f.entity_id, []) if nid in candidate_ids
            )
            matcher_topk_has_pivot = pivot.entity_id in warm_topk.get(matcher.entity_id, [])
            mate_is_pivot = pivot.entity_id in mate_ids_by_id.get(matcher.entity_id, [])
            warm_score = 0.5 * (overlaps / max(1, 5 * len(context))) + 0.5 * float(matcher_topk_has_pivot)
            by_cell[cell].append(
                PjCandidateRow(
                    cell=cell,
                    context=tuple(context),
                    pivot=pivot,
                    matcher=matcher,
                    matcher_mate_is_pivot=mate_is_pivot,
                    topk_candidate_overlap_count=overlaps,
                    matcher_topk_has_pivot=matcher_topk_has_pivot,
                    warm_score=float(warm_score),
                )
            )
    return by_cell


def select_balanced_pj_rows(
    by_cell: Dict[str, List[PjCandidateRow]],
    *,
    eval_per_cell: int,
    rng: random.Random,
) -> List[PjCandidateRow]:
    """Round-robin buckets; for mated-pair cells alternate matcher_mate_is_pivot."""
    selected: Dict[str, List[PjCandidateRow]] = {cell: [] for cell in CELL_CODES}
    for cell in CELL_CODES:
        rows = list(by_cell[cell])
        rng.shuffle(rows)
        rows.sort(key=lambda r: (r.warm_score, r.topk_candidate_overlap_count))
        used: Set[Tuple[str, ...]] = set()
        if cell[0] == ">":
            with_pivot = [r for r in rows if r.matcher_mate_is_pivot]
            without = [r for r in rows if not r.matcher_mate_is_pivot]
            queues = [with_pivot, without]
            qi = 0
            while len(selected[cell]) < eval_per_cell and any(queues):
                queue = queues[qi % 2] or queues[(qi + 1) % 2]
                qi += 1
                while queue:
                    row = queue.pop(0)
                    key = tuple(sorted(f.entity_id for f in row.context)) + (row.pivot.entity_id,)
                    if key in used:
                        continue
                    selected[cell].append(row)
                    used.add(key)
                    break
        else:
            for row in rows:
                key = tuple(sorted(f.entity_id for f in row.context)) + (row.pivot.entity_id,)
                if key in used:
                    continue
                selected[cell].append(row)
                used.add(key)
                if len(selected[cell]) >= eval_per_cell:
                    break
        if len(selected[cell]) < eval_per_cell:
            raise RuntimeError(f"Could only select {len(selected[cell])}/{eval_per_cell} rows for cell {cell}")
    out = [row for cell in CELL_CODES for row in selected[cell][:eval_per_cell]]
    rng.shuffle(out)
    return out


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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    film_by_id = load_film_map(source_corpus)
    chunk_text_by_id, chunk_tokens_by_id = load_chunk_map(canonical_corpus)
    films = [film_by_id[eid] for eid in sorted(film_by_id)]
    warm_topk = load_warm_topk_ids(warm_fusion_docs)

    first_seen: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    schedule = build_pj_warmup_schedule(films, rng=rng, k_mates=k_mates)
    for stage, context, pivot in schedule:
        row = make_pj_row(
            trace_index=len(rows),
            context=context,
            pivot=pivot,
            chunks_by_id=chunk_text_by_id,
            chunk_tokens_by_id=chunk_tokens_by_id,
            first_seen=first_seen,
            stage=stage,
            scheduled_cell=None,
            guaranteed_start=None,
            prompt_variant=prompt_variant,
        )
        rows.append(row)
        update_first_seen_pj(row, first_seen)

    by_cell = make_pj_candidate_pool(
        films=films,
        first_seen=first_seen,
        warm_topk=warm_topk,
        per_cell_attempts=per_cell_attempts,
        rng=random.Random(seed * 17 + 3),
    )
    selected = select_balanced_pj_rows(
        by_cell,
        eval_per_cell=eval_per_cell,
        rng=random.Random(seed * 19 + 5),
    )
    guaranteed_start = len(rows) + 1
    for row in rows:
        row["metadata"]["guaranteed_pollution_start_query"] = guaranteed_start

    eval_rows: List[Dict[str, Any]] = []
    for cand in selected:
        row = make_pj_row(
            trace_index=len(rows) + len(eval_rows),
            context=cand.context,
            pivot=cand.pivot,
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
        md = row["metadata"]
        md["template"] = "pivot_join_v2"
        md["matrix_scheduler_mode"] = "pivot_join_v2_balanced"
        md["v2_balance_metrics"] = {
            "warm_topk_candidate_overlap_count": cand.topk_candidate_overlap_count,
            "matcher_topk_has_pivot": cand.matcher_topk_has_pivot,
            "matcher_mate_is_pivot": cand.matcher_mate_is_pivot,
            "warm_score": cand.warm_score,
        }
        md["warm_topk_profile"] = {
            "retrieval_mode": "embedding",
            "top_k": 5,
            "profile_source": "qwen3_8B warm fusion_docs_topk5",
            "per_entity": {f.entity_id: {"topk_entity_ids": warm_topk.get(f.entity_id, [])} for f in cand.context},
        }
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

    status_by_id, predecessor_count_by_id, mate_ids = cached_pj_maps(first_seen)
    eval_by_cell = Counter(row["metadata"]["computed_matrix_cell"] for row in eval_rows)
    mate_is_pivot_by_cell = Counter(
        (row["metadata"]["computed_matrix_cell"], bool(row["metadata"]["matcher_mate_is_pivot"]))
        for row in eval_rows
    )
    manifest = {
        "seed": seed,
        "dataset": "pivot_join_v2",
        "template_family": "film_attribute_pivot_join_match",
        "conditioning_semantics": "director_mate_presence",
        "aligned_cell": "><",
        "k_mates": k_mates,
        "num_rows": len(rows),
        "warmup_rows": guaranteed_start - 1,
        "eval_rows": len(eval_rows),
        "eval_per_cell": eval_per_cell,
        "cell_counts": dict(eval_by_cell),
        "matcher_mate_is_pivot_by_cell": {f"{c}|{v}": n for (c, v), n in sorted(mate_is_pivot_by_cell.items())},
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
    parser.add_argument("--k-mates", type=int, default=1)
    parser.add_argument("--prompt-variant", default=DEFAULT_PJ_PROMPT_VARIANT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    rows, manifest = generate(
        source_corpus=args.source_corpus,
        canonical_corpus=args.canonical_corpus,
        warm_fusion_docs=args.warm_fusion_docs,
        seed=args.seed,
        eval_per_cell=args.eval_per_cell,
        per_cell_attempts=args.per_cell_attempts,
        k_mates=args.k_mates,
        prompt_variant=args.prompt_variant,
    )
    write_jsonl(args.out, rows)
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
