#!/usr/bin/env python3
"""Pivot-Join (PJ) template core: multi-hop shared-DIRECTOR match.

Third Boxoffice query template (see ../../PIVOT_TEMPLATE_DESIGN.md).
Roles: pivot (named in question), matcher (answer, shares DIRECTOR with
pivot), 8 distractors (pairwise-distinct directors).

Conditioning codes (same alphabet as OOAK/max so downstream grouping
works unchanged): `<` = fresh (no director-mate among caching
predecessors), `>` = mated. Cell = f"{pair_status}{distractor_status}"
where pair = {pivot, matcher}. Role-consistent (aligned) cell is `><`
for this template — recorded as metadata `aligned_cell`.
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

V1_SCRIPTS = Path(__file__).resolve().parent
# Shared max-template helpers (generate_one_vs_all_v1.py) live in the parent
# BoxOffice bundle under <bundle>/experiments/v1_helpers. This file sits in
# <bundle>/templates_ooak_pj/experiments/v1_helpers; override the bundle root
# with BOXOFFICE_BUNDLE_ROOT if the subtree is relocated.
_BUNDLE_ROOT = Path(os.environ.get("BOXOFFICE_BUNDLE_ROOT", str(Path(__file__).resolve().parents[3])))
for _p in (_BUNDLE_ROOT / "experiments" / "v1_helpers", V1_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from generate_one_vs_all_v1 import Film  # type: ignore  # noqa: E402
from generate_one_of_a_kind_v1 import load_chunk_map, load_film_map  # type: ignore  # noqa: E402

PJ_PROMPT_VARIANTS: Dict[str, str] = {
    "closed_world_pairwise": (
        "Use only the synthetic movie dossiers below. Ignore outside/world knowledge. "
        "Answer by comparing only the DIRECTOR fields across the named candidates. "
        "Return exactly one FILM-ID and no other text."
    ),
}
DEFAULT_PJ_PROMPT_VARIANT = "closed_world_pairwise"
PJ_ALIGNED_CELL = "><"


def make_pj_question(context: Sequence[Film], pivot: Film, prompt_variant: str) -> str:
    ids_text = ", ".join(film.entity_id for film in context)
    return (
        f"Valid candidates: {ids_text}\n"
        f"{pivot.entity_id} is one of the valid candidates. Exactly one OTHER valid candidate "
        f"has the same DIRECTOR as {pivot.entity_id}.\n"
        f"Read the DIRECTOR field of {pivot.entity_id}'s dossier, then find the other candidate "
        "whose DIRECTOR field matches it; do not infer directors from titles.\n"
        f"Return exactly one valid FILM-ID (not {pivot.entity_id})."
    )


def pj_matcher_of(context: Sequence[Film], pivot: Film) -> Film:
    """Return the unique non-pivot sharer of the pivot's director; raise if not unique."""
    sharers = [
        film
        for film in context
        if film.entity_id != pivot.entity_id and film.director == pivot.director
    ]
    if len(sharers) != 1:
        raise ValueError(
            f"Pivot {pivot.entity_id} ({pivot.director}) has {len(sharers)} sharers, want 1"
        )
    return sharers[0]


def pj_status(target: Film, predecessors: Sequence[Film]) -> Optional[str]:
    if not predecessors:
        return None
    mates = [p for p in predecessors if p.director == target.director]
    return ">" if mates else "<"


def update_first_seen_pj(row: Dict[str, Any], first_seen: Dict[str, Dict[str, Any]]) -> None:
    md = row["metadata"]
    context_ids = [str(x) for x in md["context_entity_ids"]]
    directors = md["director_labels"]
    query_index = int(row["trace_id"].rsplit("_", 1)[-1]) + 1
    for idx, entity_id in enumerate(context_ids):
        if entity_id not in first_seen:
            first_seen[entity_id] = {
                "trace_id": row["trace_id"],
                "query_index_1based": query_index,
                "preceding_ids": context_ids[:idx],
                "director": directors[entity_id],
            }


def cached_pj_maps(first_seen: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, List[str]]]:
    """(status_by_id, predecessor_count_by_id, mate_ids_by_id) from first appearances."""
    status_by_id: Dict[str, str] = {}
    predecessor_count_by_id: Dict[str, int] = {}
    mate_ids_by_id: Dict[str, List[str]] = {}
    for entity_id, cached in first_seen.items():
        preds = [str(x) for x in cached.get("preceding_ids", [])]
        predecessor_count_by_id[entity_id] = len(preds)
        if not preds:
            continue
        director = str(cached["director"])
        mates = [p for p in preds if p in first_seen and str(first_seen[p]["director"]) == director]
        mate_ids_by_id[entity_id] = mates
        status_by_id[entity_id] = ">" if mates else "<"
    return status_by_id, predecessor_count_by_id, mate_ids_by_id


def make_pj_row(
    *,
    trace_index: int,
    context: Sequence[Film],
    pivot: Film,
    chunks_by_id: Dict[str, str],
    chunk_tokens_by_id: Dict[str, Any],
    first_seen: Dict[str, Dict[str, Any]],
    stage: str,
    scheduled_cell: Optional[str],
    guaranteed_start: Optional[int],
    prompt_variant: str,
) -> Dict[str, Any]:
    context = list(context)
    context_ids = [film.entity_id for film in context]
    films_in_ctx = {film.entity_id: film for film in context}
    if pivot.entity_id not in films_in_ctx:
        raise ValueError("pivot must be in context")
    matcher = pj_matcher_of(context, pivot)
    pair_ids = {pivot.entity_id, matcher.entity_id}
    distractors = [f for f in context if f.entity_id not in pair_ids]

    instruction = PJ_PROMPT_VARIANTS[prompt_variant]
    question = make_pj_question(context, pivot, prompt_variant)
    answer = matcher.entity_id
    passages = [{"title": film.entity_id, "text": chunks_by_id[film.entity_id]} for film in context]
    ctxs = [chunks_by_id[film.entity_id] for film in context]
    prompt_segments = [instruction] + ctxs + [f"Question: {question}"]

    current_predecessors = {entity_id: context_ids[:idx] for idx, entity_id in enumerate(context_ids)}
    histories: List[Dict[str, Any]] = []
    statuses: Dict[str, Optional[str]] = {}
    orig_mates: Dict[str, List[str]] = {}
    for film in context:
        cached = first_seen.get(film.entity_id)
        predecessor_ids = [str(x) for x in cached.get("preceding_ids", [])] if cached else []
        predecessor_directors = {
            pid: (first_seen.get(pid, {}).get("director") or films_in_ctx.get(pid, film).director)
            for pid in predecessor_ids
        }
        predecessor_films = [
            Film(
                entity_id=pid,
                title="",
                director=str(d),
                release_year=0,
                starlight_awards=0,
                box_office_musd=0,
                genre="",
                studio="",
                country="",
                runtime_min=0,
            )
            for pid, d in predecessor_directors.items()
            if d is not None
        ]
        status = pj_status(film, predecessor_films)
        statuses[film.entity_id] = status
        mates = [p.entity_id for p in predecessor_films if p.director == film.director]
        orig_mates[film.entity_id] = mates
        role = (
            "pivot"
            if film.entity_id == pivot.entity_id
            else "matcher"
            if film.entity_id == matcher.entity_id
            else "distractor"
        )
        histories.append(
            {
                "entity_id": film.entity_id,
                "role": role,
                "director": film.director,
                "conditioning_status": status,
                "current_predecessor_ids": list(current_predecessors.get(film.entity_id, [])),
                "original_caching_trace_id": cached.get("trace_id") if cached else None,
                "original_caching_query_index_1based": cached.get("query_index_1based") if cached else None,
                "original_caching_predecessor_ids": predecessor_ids,
                "original_caching_predecessor_count": len(predecessor_ids),
                "original_caching_mate_ids": mates,
            }
        )

    pair_statuses = {statuses[pivot.entity_id], statuses[matcher.entity_id]}
    pair_status = next(iter(pair_statuses)) if len(pair_statuses) == 1 else None
    d_statuses = [statuses[f.entity_id] for f in distractors]
    distractor_status: Optional[str] = None
    if d_statuses and all(s == d_statuses[0] for s in d_statuses):
        distractor_status = d_statuses[0]
    computed_cell = None
    if pair_status in {"<", ">"} and distractor_status in {"<", ">"}:
        computed_cell = f"{pair_status}{distractor_status}"

    matcher_mate_is_pivot = pivot.entity_id in orig_mates.get(matcher.entity_id, [])
    before_seen = set(first_seen)
    all_seen = all(entity_id in before_seen for entity_id in context_ids)
    distractor_counts = {
        "<": d_statuses.count("<"),
        ">": d_statuses.count(">"),
        "none": d_statuses.count(None),
    }

    golden_indices = list(range(len(context)))
    row = {
        "trace_id": f"pj_query_{trace_index:04d}",
        "query_id": f"pj_query_{trace_index:04d}",
        "dataset": "pivot_join_v1",
        "longbench_id": f"pivot_join_v1_{trace_index:04d}",
        "question": question,
        "answer": answer,
        "answers": [answer, f"Answer={answer}"],
        "answers_all": [answer, f"Answer={answer}"],
        "type": "synthetic_boxoffice_pivot_join_director",
        "n_hops": len(context),
        "num_passages": len(context),
        "passages": passages,
        "ctxs": ctxs,
        "golden_chunk_indices": golden_indices,
        "golden_chunk_titles": list(context_ids),
        "golden_chunk_count": len(golden_indices),
        "non_golden_chunk_indices": [],
        "prompt_segments": prompt_segments,
        "prompt_text": "\n\n".join(prompt_segments),
        "turn_1_poison_prompt": "\n\n".join([instruction] + ctxs),
        "turn_2_eval_prompt": f"Question: {question}",
        "gold_answer": f"Answer={answer}",
        "metadata": {
            "template": "pivot_join_v1",
            "template_family": "film_attribute_pivot_join_match",
            "comparison_attribute_key": "director",
            "comparison_attribute_label": "DIRECTOR",
            "comparison_attribute_type": "categorical",
            "comparison_direction": "match",
            "question_type": "pivot_join_shared_attribute",
            "conditioning_semantics": "director_mate_presence",
            "aligned_cell": PJ_ALIGNED_CELL,
            "prompt_variant": prompt_variant,
            "instruction_chunk": instruction,
            "context_entity_ids": context_ids,
            "appears_entity_ids": context_ids,
            "candidate_entity_ids": context_ids,
            "winner_entity_id": matcher.entity_id,
            "winner_attribute_value": matcher.director,
            "runner_up_entity_id": pivot.entity_id,
            "runner_up_attribute_value": pivot.director,
            "answer_entity_id": matcher.entity_id,
            "answer_chunk_ids": [matcher.entity_id],
            "runner_up_chunk_ids": [pivot.entity_id],
            "gold_chunk_ids": list(context_ids),
            "gold_chunk_roles": {
                film.entity_id: (
                    "pivot"
                    if film.entity_id == pivot.entity_id
                    else "matcher"
                    if film.entity_id == matcher.entity_id
                    else "distractor"
                )
                for film in context
            },
            "pivot_entity_id": pivot.entity_id,
            "matcher_entity_id": matcher.entity_id,
            "shared_director": pivot.director,
            "matcher_mate_is_pivot": matcher_mate_is_pivot,
            "director_labels": {film.entity_id: film.director for film in context},
            "num_context_chunks": len(context),
            "num_candidates": len(context),
            "chance_level": round(1.0 / (len(context) - 1), 4),
            "pollution_mode": "guaranteed" if guaranteed_start is not None and trace_index + 1 >= guaranteed_start else "warmup",
            "matrix_scheduler_mode": "pivot_join_v1",
            "matrix_scheduler_stage": stage,
            "scheduled_matrix_cell": scheduled_cell,
            "computed_matrix_cell": computed_cell,
            "winner_conditioning_status": statuses.get(matcher.entity_id),
            "pair_conditioning_status": pair_status,
            "runner_up_conditioning_status": statuses.get(pivot.entity_id),
            "nonwinner_conditioning_status": distractor_status,
            "distractor_conditioning_status": distractor_status,
            "nonwinner_conditioning_status_counts": distractor_counts,
            "other_conditioning_status_counts": distractor_counts,
            "original_caching_mate_ids": orig_mates,
            "matrix_threshold": 0.5,
            "guaranteed_pollution_start_query": guaranteed_start,
            "all_chunks_seen_before_query": all_seen,
            "comparison_attribute_values": {film.entity_id: film.director for film in context},
            "chunk_token_counts": {entity_id: chunk_tokens_by_id.get(entity_id) for entity_id in context_ids},
            "pollution_guarantee": {
                "holds": bool(all_seen and scheduled_cell is not None),
                "gold_chunk_ids": list(context_ids),
                "answer_chunk_ids": [matcher.entity_id],
                "gold_chunk_histories": histories,
                "all_chunk_histories": histories,
            },
        },
    }
    return row


def build_pj_fresh_prefix(
    *,
    target: Film,
    pool: Sequence[Film],
    rng: random.Random,
) -> Optional[Tuple[List[Film], Film]]:
    """9 predecessors: one director-pair + 7 distinct-director fillers.

    Returns (ordered_prefix, row_pivot). No prefix film shares the target's
    director, so the appended target is cached fresh. The row's question
    names row_pivot (one of the pair); the answer is the other pair member.
    """
    by_dir: Dict[str, List[Film]] = {}
    for film in pool:
        if film.entity_id == target.entity_id or film.director == target.director:
            continue
        by_dir.setdefault(film.director, []).append(film)
    pair_dirs = [d for d, films in by_dir.items() if len(films) >= 2]
    if not pair_dirs:
        return None
    rng.shuffle(pair_dirs)
    pair_dir = pair_dirs[0]
    pair = rng.sample(by_dir[pair_dir], k=2)
    filler_dirs = [d for d in by_dir if d != pair_dir]
    if len(filler_dirs) < 7:
        return None
    fillers = [rng.choice(by_dir[d]) for d in rng.sample(filler_dirs, k=7)]
    prefix = pair + fillers
    rng.shuffle(prefix)
    pivot = rng.choice(pair)
    return prefix, pivot


def build_pj_mated_prefix(
    *,
    target: Film,
    pool: Sequence[Film],
    rng: random.Random,
    k_mates: int = 1,
) -> Optional[Tuple[List[Film], Film]]:
    """9 predecessors: k mates of the target + (9-k) distinct-director fillers.

    Returns (ordered_prefix, row_pivot). Row pivot = one mate; the appended
    target is the row's matcher (cached mated). With k_mates=1 the named
    pivot's director appears exactly twice (mate + target). Fillers have
    pairwise-distinct directors, none equal to the target's or each other's.
    """
    if not 1 <= k_mates <= 2:
        return None
    by_dir: Dict[str, List[Film]] = {}
    mates_avail: List[Film] = []
    for film in pool:
        if film.entity_id == target.entity_id:
            continue
        if film.director == target.director:
            mates_avail.append(film)
        else:
            by_dir.setdefault(film.director, []).append(film)
    if len(mates_avail) < k_mates:
        return None
    mates = rng.sample(mates_avail, k=k_mates)
    n_fillers = 9 - k_mates
    filler_dirs = list(by_dir)
    if len(filler_dirs) < n_fillers:
        return None
    fillers = [rng.choice(by_dir[d]) for d in rng.sample(filler_dirs, k=n_fillers)]
    prefix = mates + fillers
    rng.shuffle(prefix)
    # With k_mates=2 the pivot's director appears 3x in the full row, which
    # breaks answer uniqueness — only k_mates=1 rows are valid PJ queries.
    if k_mates != 1:
        return None
    pivot = mates[0]
    return prefix, pivot
