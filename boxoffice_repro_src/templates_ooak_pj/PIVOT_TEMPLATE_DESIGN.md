# Pivot-Join (PJ): the third Boxoffice query template — multi-hop attribute match

This template adds multi-hop semantic reasoning to Boxoffice in a
controlled form: find the named
pivot's dossier → extract its DIRECTOR → find the one other candidate
with the same DIRECTOR.

## Task

Same corpus, same 10-candidate context shape:

> Valid candidates: FILM-a, ..., FILM-j
> FILM-P is one of the valid candidates. Exactly one OTHER valid candidate
> has the same DIRECTOR as FILM-P.
> Read the DIRECTOR field of FILM-P's dossier, then find the other
> candidate whose DIRECTOR field matches it; do not infer directors from
> titles.
> Return exactly one valid FILM-ID (not FILM-P).

Roles: **pivot** P (named in the question), **matcher** M (the answer —
shares director d with P), 8 **distractors** with pairwise-distinct
directors ∉ {d}. Every chunk is golden (uniqueness of M requires checking
every candidate). Chance = 1/9 (uniform over the non-pivot candidates).

Corpus support: 20 directors over 100 films; every film has 1–7
director-mates (group sizes 2–8, median 5), so pivot pairs, out-of-context
mates for stale seeding, and 8-distinct-director distractor sets all exist
in depth.

## Conditioning

Same fresh/mated axis as OOAK, over DIRECTOR instead of GENRE:
`<` = no director-mate among the chunk's caching predecessors,
`>` = ≥1 mate (k dial, default 1).

Roles at eval: P and M "share" (duplicate-like); distractors are
directorially unique in-context (singleton-like). The two cell axes are
the **pair** {P, M} (uniform status) and the **distractors** (uniform):

| cell | pair cached | distractors cached | role-consistency |
|------|-------------|--------------------|------------------|
| `><` | mated       | fresh              | **aligned** (every cache matches its eval role) |
| `>>` | mated       | mated              | distractors stale |
| `<<` | fresh       | fresh              | pair stale |
| `<>` | fresh       | mated              | **flipped** (every cache contradicts its role) |

NOTE the mapping differs from OOAK/max (`aligned_cell = "><"` is recorded
in metadata; plots group by literal cell then relabel).

**Association dial (new, unique to PJ):** for a mated matcher, the mate in
its caching prefix may be the pivot itself (`matcher_mate_is_pivot=true` —
the exact P–M association is pre-baked in the cache) or a third
same-director film (association stale even though the "I share my
director" signal is fresh). Recorded per row; selection balances the two
when the pool allows. This probes association-specific staleness, which
neither the max template nor OOAK can express.

## Warmup

Every warmup row is a valid PJ query (same instruction; the named pivot has
exactly one sharer). Target = 10th chunk:
- fresh-seeding: prefix = one director-pair (the row's P and M, drawn from
  already-seen films, director ≠ target's) + 7 pairwise-distinct-director
  fillers; target's director absent → target cached `<`.
- mated-seeding: prefix = k mates of the target + 8 pairwise-distinct
  fillers; the row's question names the mate as pivot → answer = target
  (the target is its seeding row's matcher); target cached `>`.
- Two intro rows bootstrap the seen pool; their films keep <9 caching
  predecessors and are excluded from eval pools (as in OOAK/v2-max).
- Per director group, fresh-seeded members precede mated ones.

## Pipeline

Identical to OOAK: generate oversampled → joint 3-model filter
(full-context F1=1, question-only EM=0) → 40/cell × 4 cells × 2 seeds →
M=4 balance (≥2 fresh + ≥2 mated variants per eval chunk) →
`run_boxoffice.py` unchanged.

## Invariants (validate_pivot.py)

- named pivot's director appears exactly twice in context (P + M); answer
  = M; distractors pairwise-distinct directors ∉ {d};
- eval: recomputed statuses match `computed_matrix_cell`; scheduled ==
  computed; pair uniform, distractors uniform; every chunk ≥9 caching
  predecessors, first-seen in warmup;
- warmup rows valid PJ queries; question names an in-context pivot.
