# One-of-a-Kind (OOAK): the second Boxoffice query template

Genre-singleton odd-one-out — a second query template over the same corpus,
showing that Boxoffice is not tied to a single (ordinal max) query template.

## Task

Same corpus (the canonical 100-film catalogue, byte-identical chunks), same
10-candidate context shape, new question:

> Valid candidates: FILM-a, ..., FILM-j
> Which valid FILM-ID is the only film whose GENRE appears exactly once
> among all listed candidates?
> Read the GENRE field from the dossiers; do not infer genres from titles.
> Return exactly one valid FILM-ID.

By construction the 10 candidates partition by genre as 3+3+3+1: three
genres appear three times each, one genre appears once. The answer is the
FILM-ID of the genre singleton.

Properties preserved from the max template:
- answer space = the 10 listed FILM-IDs → chance = 1/10 (LI filter empty);
- fictional films → NC filter empty; joint 3-model full-prefill filter → B0 empty;
- every chunk is golden (uniqueness of the singleton can only be verified by
  reading every candidate's GENRE);
- same instruction/dossiers/question prompt layout (12 segments).

What changes: the reasoning primitive. Max = ordinal comparison over a
numeric field; OOAK = equality/membership matching over a categorical field
(set reasoning, induction-head style). Not an argmax/argmin.

## Roles and conditioning

Within a query, a chunk's role is either **singleton** (its genre appears
once — the answer) or **duplicate** (its genre appears ≥2 times).

At warmup (first appearance), a chunk is cached under one of two
conditionings, determined by its predecessors in the caching context:

- **fresh** (code `<`): no predecessor shares its genre → the cached KV
  encodes "I am the only one of my kind so far" (singleton-consistent);
- **mated** (code `>`): ≥1 predecessor shares its genre (k mates, default
  k=1; k is the strength dial) → the cached KV encodes "my kind repeats"
  (duplicate-consistent).

The 2-char cell code `{singleton_status}{duplicates_status}` reuses the
max-template alphabet so `run_boxoffice.py`, the joint filter, and all
plotting group-bys work unchanged:

| cell | singleton cached | duplicates cached | analog of |
|------|------------------|-------------------|-----------|
| `<>` | fresh            | mated             | aligned (easy) |
| `<<` | fresh            | fresh             | all-W (medium) |
| `>>` | mated            | mated             | all-N (medium) |
| `><` | mated            | fresh             | flipped (hard) |

Metadata records `conditioning_semantics: "genre_mate_presence"` plus
per-chunk mate counts so the reinterpretation is explicit in the data.

## The Sci-Fi / Science Fiction hazard

The corpus has two surface labels for one semantic genre: `Sci-Fi` (10
films) and `Science Fiction` (6). Guards:

1. genre logic uses surface labels; a duplicate triple is always
   same-label;
2. a query never contains both labels of the same canonical class
   (otherwise a literal reader finds two "singletons");
3. the singleton's canonical class differs from every duplicate class
   (semantic uniqueness, not just literal).

Warmup mates are same-label; freshness is canonical-class-strict (a
`Science Fiction` predecessor breaks freshness of a `Sci-Fi` target).

Effective classes (canonical): Thriller 18, Drama 15, Action 15, Neo-Noir
12, Mystery 11, Sci-Fi 10+6, Comedy 7, Adventure 6 — every label has ≥3
films on each conditioning side under interleaved scheduling, enough for
3-per-class duplicate groups and out-of-query mates.

## Warmup schedule (analog of v2's interleaved support blocks)

Every warmup row is itself a well-formed OOAK query (same instruction —
required, since the instruction segment precedes the chunks and is part of
every cached KV prefix).

- Two fresh-support blocks FS_A, FS_B: 9 films each, three same-label
  triples (3+3+3); FS_A and FS_B use disjoint class sets. Introduced by two
  intro rows (their films' own statuses are not used for eval pools — the
  ≥9-predecessor requirement excludes them, as in v2).
- Fresh-seeding row for target t: `FS_x + [t]` where FS_x avoids t's class
  → t is the singleton (answer = t), cached fresh with 9 predecessors.
- Mated-seeding row for target t: `[k mates of t] + fillers + [designated
  singleton s] + [t]` with the 9 predecessors arranged so the row has
  exactly one singleton = s (answer = s ≠ t); t is cached mated (k
  same-label mate predecessors, default k=1).
- Targets are scheduled per label, alternating fresh/mated so both pools
  stay balanced per label; a label's first targets are fresh-seeded so
  later mated-seedings have already-seen mates.

## Eval assembly (analog of v2's per-cell pools + balanced selection)

For each cell (a,b): singleton from the status-a pool, 9 duplicates =
three same-label triples from the status-b pool, all classes distinct from
each other and from the singleton's class, subject to guards (1)-(3).
Oversample per cell (default 100), dedup by candidate set, then select a
balanced subset via warm-top-k metrics (per-chunk e5 top-5 neighbours from
the same corpus-wide profile the max template uses):

- `warm_topk_candidate_overlap_count` (same as v2),
- `singleton_topk_mate_count` — how many of the singleton's top-5
  neighbours are class-mates (the FR-fairness analog of numeric delta
  direction).

The joint filter (unchanged scripts) then keeps rows where all 3 models
answer correctly with full context and none answers correctly
question-only; trim to 40/cell → 160 eval rows per seed.

## M>1 (CacheCraft) balance — v4 analog

`build_ooak_v4_balanced.py` appends warmup rows until every eval-used chunk
has ≥2 fresh and ≥2 mated cached variants (M=4), mirroring
`build_filtered_v4_balanced.py` (variant prefixes are 9-predecessor rows of
the same two shapes above, each still a valid OOAK query).

## Pipeline (all downstream stages unchanged)

1. `generate_one_of_a_kind_v2.py` → oversampled `ooak_v2_s<seed>_full/eval.jsonl`
2. `run_ooak_joint_filtered_probe.sh` → joint 3-model filter → 40/cell
3. `build_ooak_v4_balanced.py` → `ooak_filtered_v4_balanced_m4_s<seed>_{full,eval}.jsonl`
4. `runtime/run_boxoffice.py` with `--boxoffice-dir/--eval-stem`
   pointed at the ooak seeds — no code changes (it reads only
   `prompt_segments`, `question`, `answers*`, `golden_chunk_indices`, and
   the `<<//<>/></>>` cell codes).
5. Same plotting, with cells relabeled aligned/all-fresh/all-mated/flipped.

Row schema is identical to the max template (all keys present;
`winner_*` = singleton, `runner_up_*` = null), so every existing loader
works.

## Invariants (checked by `validate_ooak.py`, pure python, runs anywhere)

- exactly one label-singleton AND one class-singleton per row; answer = it;
- no query mixes Sci-Fi and Science Fiction;
- eval: `computed_matrix_cell == scheduled_matrix_cell`, 40/cell (post-filter),
  every eval chunk first-seen in warmup with ≥9 predecessors, singleton
  status = cell[0], all nine duplicates = cell[1];
- v4: every eval chunk has ≥2 fresh + ≥2 mated warmup variants;
- warmup rows are valid OOAK queries (answer verifiable from the context).
