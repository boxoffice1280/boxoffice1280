#!/usr/bin/env python3
"""One-of-a-Kind (OOAK) template core: genre-singleton odd-one-out.

Second Boxoffice query template (see ../../OOAK_TEMPLATE_DESIGN.md).
Same corpus, same 10-candidate context shape and row schema as the
one-vs-all max template; the question asks for the FILM-ID of the only
film whose GENRE appears exactly once among the candidates.

Conditioning codes reuse the max-template alphabet so every downstream
consumer (run_boxoffice.py, joint filter, plotting) works unchanged:

  `<` = fresh  (no genre class-mate among caching predecessors)
  `>` = mated  (>=1 same-label genre mate among caching predecessors)

Cell = f"{singleton_status}{duplicates_status}":
  `<>` aligned, `<<` all-fresh, `>>` all-mated, `><` flipped.
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

from generate_one_vs_all_v1 import (  # type: ignore  # noqa: E402
    CELL_CODES,
    Film,
    load_jsonl,
    write_jsonl,
)

# Two surface labels for one semantic genre. All *class* logic canonicalises;
# duplicate triples and warmup mates are always same-*label*.
GENRE_CANON = {"Science Fiction": "Sci-Fi"}

OOAK_PROMPT_VARIANTS: Dict[str, str] = {
    "closed_world": (
        "Use only the synthetic movie dossiers below. Ignore outside/world knowledge. "
        "For uniqueness, compare only the GENRE fields across all named candidates. "
        "Return exactly one FILM-ID and no other text."
    ),
    "closed_world_pairwise": (
        "Use only the synthetic movie dossiers below. Ignore outside/world knowledge. "
        "For each named candidate, compare its GENRE field with every other candidate's "
        "GENRE field; exactly one candidate's GENRE is shared by no other candidate. "
        "Return exactly one FILM-ID and no other text."
    ),
}
DEFAULT_OOAK_PROMPT_VARIANT = "closed_world"


def genre_class(label: str) -> str:
    return GENRE_CANON.get(label, label)


def load_film_map(path: Path) -> Dict[str, Film]:
    out: Dict[str, Film] = {}
    for row in load_jsonl(path):
        film = Film(
            entity_id=str(row["entity_id"]),
            title=str(row["title"]),
            director=str(row["director"]),
            release_year=int(row["release_year"]),
            starlight_awards=int(row["starlight_awards"]),
            box_office_musd=int(row["box_office_musd"]),
            genre=str(row["genre"]),
            studio=str(row["studio"]),
            country=str(row["country"]),
            runtime_min=int(row["runtime_min"]),
            cast=str(row.get("cast", "")),
            setting_city=str(row.get("setting_city", "")),
            setting_city_population_mil=str(row.get("setting_city_population_mil", "")),
        )
        out[film.entity_id] = film
    return out


def load_chunk_map(path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    texts: Dict[str, str] = {}
    toks: Dict[str, Any] = {}
    for row in load_jsonl(path):
        eid = str(row["entity_id"])
        texts[eid] = str(row["text"])
        toks[eid] = row.get("rendered_token_count_no_tokenizer")
    return texts, toks


def ooak_instruction(prompt_variant: str) -> str:
    return OOAK_PROMPT_VARIANTS[prompt_variant]


def make_ooak_question(context: Sequence[Film], prompt_variant: str) -> str:
    ids_text = ", ".join(film.entity_id for film in context)
    if prompt_variant == "closed_world_pairwise":
        return (
            f"Valid candidates: {ids_text}\n"
            "Exactly one valid candidate has a GENRE that no other listed candidate shares.\n"
            "Check the GENRE field of every dossier against the others; do not infer genres from titles.\n"
            "Which FILM-ID is that unique-genre candidate? Return exactly one valid FILM-ID."
        )
    return (
        f"Valid candidates: {ids_text}\n"
        "Which valid FILM-ID is the only film whose GENRE appears exactly once among all listed candidates?\n"
        "Read the GENRE field from the dossiers; do not infer genres from titles.\n"
        "Return exactly one valid FILM-ID."
    )


def singleton_of(context: Sequence[Film]) -> Film:
    """Return the unique genre singleton; raise if the row is ill-formed."""
    label_counts = Counter(film.genre for film in context)
    class_counts = Counter(genre_class(film.genre) for film in context)
    label_singles = [film for film in context if label_counts[film.genre] == 1]
    class_singles = [film for film in context if class_counts[genre_class(film.genre)] == 1]
    if len(label_singles) != 1 or len(class_singles) != 1:
        raise ValueError(
            f"Row is not a valid OOAK query: label_singletons={[f.entity_id for f in label_singles]} "
            f"class_singletons={[f.entity_id for f in class_singles]}"
        )
    if label_singles[0].entity_id != class_singles[0].entity_id:
        raise ValueError("Label singleton and class singleton disagree")
    mixed = {
        cls
        for cls in class_counts
        if len({film.genre for film in context if genre_class(film.genre) == cls}) > 1
    }
    if mixed:
        raise ValueError(f"Query mixes surface labels of canonical class(es) {sorted(mixed)}")
    return label_singles[0]


def ooak_status(target: Film, predecessors: Sequence[Film]) -> Optional[str]:
    """`<` fresh / `>` mated from the (class-strict) predecessors; None if no predecessors."""
    if not predecessors:
        return None
    mates = [p for p in predecessors if genre_class(p.genre) == genre_class(target.genre)]
    return ">" if mates else "<"


def mate_count(target: Film, predecessors: Sequence[Film]) -> int:
    return sum(1 for p in predecessors if genre_class(p.genre) == genre_class(target.genre))


def update_first_seen_ooak(row: Dict[str, Any], first_seen: Dict[str, Dict[str, Any]]) -> None:
    md = row["metadata"]
    context_ids = [str(x) for x in md["context_entity_ids"]]
    genres = md["genre_labels"]
    query_index = int(row["trace_id"].rsplit("_", 1)[-1]) + 1
    for idx, entity_id in enumerate(context_ids):
        if entity_id not in first_seen:
            first_seen[entity_id] = {
                "trace_id": row["trace_id"],
                "query_index_1based": query_index,
                "preceding_ids": context_ids[:idx],
                "genre": genres[entity_id],
            }


def cached_ooak_maps(first_seen: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]:
    """(status_by_id, predecessor_count_by_id, mate_count_by_id) from first appearances."""
    status_by_id: Dict[str, str] = {}
    predecessor_count_by_id: Dict[str, int] = {}
    mate_count_by_id: Dict[str, int] = {}
    for entity_id, cached in first_seen.items():
        preds = [str(x) for x in cached.get("preceding_ids", [])]
        predecessor_count_by_id[entity_id] = len(preds)
        if not preds:
            continue
        target_cls = genre_class(str(cached["genre"]))
        pred_genres = [str(first_seen[p]["genre"]) for p in preds if p in first_seen]
        if len(pred_genres) != len(preds):
            continue
        mates = sum(1 for g in pred_genres if genre_class(g) == target_cls)
        mate_count_by_id[entity_id] = mates
        status_by_id[entity_id] = ">" if mates else "<"
    return status_by_id, predecessor_count_by_id, mate_count_by_id


def make_ooak_row(
    *,
    trace_index: int,
    context: Sequence[Film],
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
    singleton = singleton_of(context)
    duplicates = [film for film in context if film.entity_id != singleton.entity_id]

    instruction = ooak_instruction(prompt_variant)
    question = make_ooak_question(context, prompt_variant)
    answer = singleton.entity_id
    passages = [{"title": film.entity_id, "text": chunks_by_id[film.entity_id]} for film in context]
    ctxs = [chunks_by_id[film.entity_id] for film in context]
    prompt_segments = [instruction] + ctxs + [f"Question: {question}"]

    current_predecessors = {entity_id: context_ids[:idx] for idx, entity_id in enumerate(context_ids)}
    histories: List[Dict[str, Any]] = []
    statuses: Dict[str, Optional[str]] = {}
    orig_mate_counts: Dict[str, int] = {}
    for film in context:
        cached = first_seen.get(film.entity_id)
        predecessor_ids = [str(x) for x in cached.get("preceding_ids", [])] if cached else []
        predecessor_genres = {
            pred_id: (first_seen.get(pred_id, {}).get("genre") or films_in_ctx.get(pred_id, film).genre)
            for pred_id in predecessor_ids
        }
        predecessor_films = [
            Film(
                entity_id=pred_id,
                title="",
                director="",
                release_year=0,
                starlight_awards=0,
                box_office_musd=0,
                genre=str(genre),
                studio="",
                country="",
                runtime_min=0,
            )
            for pred_id, genre in predecessor_genres.items()
            if genre is not None
        ]
        status = ooak_status(film, predecessor_films)
        statuses[film.entity_id] = status
        orig_mates = mate_count(film, predecessor_films)
        orig_mate_counts[film.entity_id] = orig_mates
        current_pred_ids = current_predecessors.get(film.entity_id, [])
        histories.append(
            {
                "entity_id": film.entity_id,
                "role": "singleton" if film.entity_id == singleton.entity_id else "duplicate_candidate",
                "genre_label": film.genre,
                "genre_class": genre_class(film.genre),
                "conditioning_status": status,
                "current_predecessor_ids": list(current_pred_ids),
                "current_predecessor_mate_count": sum(
                    1
                    for pid in current_pred_ids
                    if pid in films_in_ctx and genre_class(films_in_ctx[pid].genre) == genre_class(film.genre)
                ),
                "original_caching_trace_id": cached.get("trace_id") if cached else None,
                "original_caching_query_index_1based": cached.get("query_index_1based") if cached else None,
                "original_caching_predecessor_ids": predecessor_ids,
                "original_caching_predecessor_count": len(predecessor_ids),
                "original_caching_predecessor_genres": predecessor_genres,
                "original_caching_mate_count": orig_mates,
            }
        )

    singleton_status = statuses.get(singleton.entity_id)
    duplicate_statuses = [statuses[f.entity_id] for f in duplicates]
    duplicates_uniform: Optional[str] = None
    if duplicate_statuses and all(s == duplicate_statuses[0] for s in duplicate_statuses):
        duplicates_uniform = duplicate_statuses[0]
    computed_cell = None
    if singleton_status in {"<", ">"} and duplicates_uniform in {"<", ">"}:
        computed_cell = f"{singleton_status}{duplicates_uniform}"

    before_seen = set(first_seen)
    all_seen = all(entity_id in before_seen for entity_id in context_ids)
    duplicate_counts = {
        "<": duplicate_statuses.count("<"),
        ">": duplicate_statuses.count(">"),
        "none": duplicate_statuses.count(None),
    }
    label_partition: Dict[str, List[str]] = {}
    for film in context:
        label_partition.setdefault(film.genre, []).append(film.entity_id)

    golden_indices = list(range(len(context)))
    row = {
        "trace_id": f"ooak_query_{trace_index:04d}",
        "query_id": f"ooak_query_{trace_index:04d}",
        "dataset": "one_of_a_kind_v1",
        "longbench_id": f"one_of_a_kind_v1_{trace_index:04d}",
        "question": question,
        "answer": answer,
        "answers": [answer, f"Answer={answer}"],
        "answers_all": [answer, f"Answer={answer}"],
        "type": "synthetic_boxoffice_one_of_a_kind_genre_singleton",
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
            "template": "one_of_a_kind_v1",
            "template_family": "film_attribute_one_of_a_kind_singleton",
            "comparison_attribute_key": "genre",
            "comparison_attribute_label": "GENRE",
            "comparison_attribute_type": "categorical",
            "comparison_direction": "unique",
            "question_type": "one_of_a_kind_singleton",
            "conditioning_semantics": "genre_mate_presence",
            "prompt_variant": prompt_variant,
            "instruction_chunk": instruction,
            "context_entity_ids": context_ids,
            "appears_entity_ids": context_ids,
            "candidate_entity_ids": context_ids,
            "winner_entity_id": singleton.entity_id,
            "winner_attribute_value": singleton.genre,
            "runner_up_entity_id": None,
            "runner_up_attribute_value": None,
            "answer_entity_id": singleton.entity_id,
            "answer_chunk_ids": [singleton.entity_id],
            "runner_up_chunk_ids": [],
            "gold_chunk_ids": list(context_ids),
            "gold_chunk_roles": {
                film.entity_id: ("singleton" if film.entity_id == singleton.entity_id else "duplicate_candidate")
                for film in context
            },
            "singleton_entity_id": singleton.entity_id,
            "singleton_genre_label": singleton.genre,
            "singleton_genre_class": genre_class(singleton.genre),
            "genre_labels": {film.entity_id: film.genre for film in context},
            "genre_label_partition": label_partition,
            "num_context_chunks": len(context),
            "num_candidates": len(context),
            "pollution_mode": "guaranteed" if guaranteed_start is not None and trace_index + 1 >= guaranteed_start else "warmup",
            "matrix_scheduler_mode": "one_of_a_kind_v1",
            "matrix_scheduler_stage": stage,
            "scheduled_matrix_cell": scheduled_cell,
            "computed_matrix_cell": computed_cell,
            "winner_conditioning_status": singleton_status,
            "singleton_conditioning_status": singleton_status,
            "runner_up_conditioning_status": None,
            "nonwinner_conditioning_status": duplicates_uniform,
            "duplicates_conditioning_status": duplicates_uniform,
            "nonwinner_conditioning_status_counts": duplicate_counts,
            "other_conditioning_status_counts": duplicate_counts,
            "original_caching_mate_counts": orig_mate_counts,
            "matrix_threshold": 0.5,
            "guaranteed_pollution_start_query": guaranteed_start,
            "all_chunks_seen_before_query": all_seen,
            "comparison_attribute_values": {film.entity_id: film.genre for film in context},
            "chunk_token_counts": {entity_id: chunk_tokens_by_id.get(entity_id) for entity_id in context_ids},
            "pollution_guarantee": {
                "holds": bool(all_seen and scheduled_cell is not None),
                "gold_chunk_ids": list(context_ids),
                "answer_chunk_ids": [singleton.entity_id],
                "gold_chunk_histories": histories,
                "all_chunk_histories": histories,
            },
        },
    }
    return row


def _distinct_class_labels(
    labels_by_count: Dict[str, List[str]],
    *,
    exclude_classes: Set[str],
    need: int,
    min_films: int,
    rng: random.Random,
) -> Optional[List[str]]:
    """Pick `need` labels with >=min_films available films, pairwise-distinct classes."""
    eligible = [label for label, ids in labels_by_count.items() if len(ids) >= min_films and genre_class(label) not in exclude_classes]
    rng.shuffle(eligible)
    chosen: List[str] = []
    used_classes: Set[str] = set(exclude_classes)
    for label in eligible:
        cls = genre_class(label)
        if cls in used_classes:
            continue
        chosen.append(label)
        used_classes.add(cls)
        if len(chosen) == need:
            return chosen
    return None


def build_fresh_prefix(
    *,
    target: Film,
    pool: Sequence[Film],
    rng: random.Random,
    num_predecessors: int = 9,
) -> Optional[List[Film]]:
    """9 predecessors as three same-label triples, no class-mate of target.

    The resulting row (prefix + [target]) has exactly one singleton: the target.
    """
    if num_predecessors != 9:
        raise ValueError("OOAK prefixes are 3+3+3 over 9 predecessors")
    by_label: Dict[str, List[Film]] = {}
    for film in pool:
        if film.entity_id == target.entity_id:
            continue
        by_label.setdefault(film.genre, []).append(film)
    labels = _distinct_class_labels(
        {label: [f.entity_id for f in films] for label, films in by_label.items()},
        exclude_classes={genre_class(target.genre)},
        need=3,
        min_films=3,
        rng=rng,
    )
    if labels is None:
        return None
    prefix: List[Film] = []
    for label in labels:
        prefix.extend(rng.sample(by_label[label], k=3))
    rng.shuffle(prefix)
    return prefix


def build_mated_prefix(
    *,
    target: Film,
    pool: Sequence[Film],
    rng: random.Random,
    k_mates: int = 1,
    num_predecessors: int = 9,
) -> Optional[Tuple[List[Film], Film]]:
    """9 predecessors: k same-label mates of target + fillers + one designated singleton.

    Returns (ordered_prefix, designated_singleton). The resulting row
    (prefix + [target]) has exactly one singleton (the designated one, whose
    class appears once); the target is mated (k same-label mates precede it).
    Filler groups keep every non-singleton class at >=2 occurrences.
    """
    if num_predecessors != 9:
        raise ValueError("OOAK prefixes use 9 predecessors")
    if not 1 <= k_mates <= 3:
        raise ValueError("k_mates must be in 1..3")
    by_label: Dict[str, List[Film]] = {}
    for film in pool:
        if film.entity_id == target.entity_id:
            continue
        by_label.setdefault(film.genre, []).append(film)
    mates_avail = by_label.get(target.genre, [])
    if len(mates_avail) < k_mates:
        return None
    mates = rng.sample(mates_avail, k=k_mates)
    target_cls = genre_class(target.genre)

    # Filler layout over the remaining 9 - k - 1 slots, all groups >= 2:
    filler_slots = num_predecessors - k_mates - 1
    layouts = {7: (3, 2, 2), 6: (2, 2, 2), 5: (3, 2)}
    layout = layouts.get(filler_slots)
    if layout is None:
        return None
    label_ids = {label: [f.entity_id for f in films] for label, films in by_label.items()}
    filler_labels = _distinct_class_labels(
        label_ids,
        exclude_classes={target_cls},
        need=len(layout),
        min_films=max(layout),
        rng=rng,
    )
    if filler_labels is None:
        return None
    used_classes = {target_cls} | {genre_class(label) for label in filler_labels}
    singleton_label_choices = [
        label
        for label, films in by_label.items()
        if genre_class(label) not in used_classes and len(films) >= 1
    ]
    if not singleton_label_choices:
        return None
    singleton_label = rng.choice(singleton_label_choices)
    designated = rng.choice(by_label[singleton_label])

    fillers: List[Film] = []
    for label, count in zip(filler_labels, layout):
        candidates = [f for f in by_label[label] if f.entity_id != designated.entity_id]
        if len(candidates) < count:
            return None
        fillers.extend(rng.sample(candidates, k=count))
    prefix = mates + fillers + [designated]
    rng.shuffle(prefix)
    return prefix, designated
