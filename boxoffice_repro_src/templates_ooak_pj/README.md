# BoxOffice query templates: OOAK and PJ

This subtree adds two further BoxOffice query templates on top of the
one-vs-all *max* template shipped in the parent bundle:

- **OOAK** — *one-of-a-kind* / categorical (genre) uniqueness
- **PJ** — *pivot join* / multi-hop shared-DIRECTOR match

Both reuse the same 100-film corpus (byte-identical dossier chunks), the same
10-candidate context shape, the same joint 3-model filter, the same M=4
warm-balance step, and the **unchanged** BoxOffice runtime
(`../runtime/run_boxoffice.py`, `../runtime/run.sh`). Nothing in the parent
bundle was modified; everything template-specific lives here.

Seeds used: **7, 11, 13, 17**. Models: **`qwen3-8B`, `llama3.1-8BI`,
`mistral-7B`** (the three default BoxOffice models).

## 1. What the templates are

### OOAK — one-of-a-kind (genre singleton)

Same corpus and context shape as the max template, new question: *which of
the ten listed FILM-IDs is the only film whose GENRE appears exactly once among
the candidates?* By construction the ten candidates partition by genre into
duplicate groups plus one singleton (the released seeds use a 5 + 4 + 1
layout, `duplicate_partition_layout = [5, 4]`); the answer is the singleton.
Chance stays 1/10, every chunk is golden (uniqueness can only be verified by
reading every candidate's GENRE), and the prompt layout is the same 12-segment
instruction / dossiers / question layout. What changes is the reasoning
primitive: max is an ordinal comparison over a numeric field, OOAK is
equality/membership matching over a categorical field. The corpus has two
surface labels for one genre (`Sci-Fi` / `Science Fiction`); the generator
never mixes them in one query and treats freshness at the canonical-class
level. Full design: [`OOAK_TEMPLATE_DESIGN.md`](OOAK_TEMPLATE_DESIGN.md).

### PJ — pivot join (shared director, multi-hop)

Controlled multi-hop reasoning: the question names a **pivot** film P that is
one of the ten candidates and states that exactly one *other* candidate shares
P's DIRECTOR. The model must find P's dossier, read its DIRECTOR, and return
the one **matcher** M with the same DIRECTOR; the other eight candidates
(**distractors**) have pairwise-distinct directors different from P's. Chance
is 1/9. PJ also exposes an *association* dial that neither max nor OOAK can:
for a mated matcher, the same-director film in its caching prefix may be the
pivot itself (`matcher_mate_is_pivot = true`, the exact P-M association is
pre-baked in the cache) or a third film. Full design:
[`PIVOT_TEMPLATE_DESIGN.md`](PIVOT_TEMPLATE_DESIGN.md).

## 2. Cell semantics

Every chunk is cached at its first appearance in the warmup stream under one
of two conditionings, determined by its predecessors in that caching context:

- `<` = **fresh**: no predecessor shares the chunk's attribute
- `>` = **mated**: at least one predecessor shares it (`k_mates = 1`)

The attribute is **genre** for OOAK and **director** for PJ. An eval row's
cell code is `<special-role status><other-chunks status>`, where the
special role is the singleton (OOAK) or the pivot/matcher pair (PJ) and the
other chunks are the duplicates (OOAK) or the distractors (PJ). The four cells
are balanced to **40 eval rows each** (160 eval rows per seed).

| code | physical meaning              | OOAK name   | PJ name     |
|------|-------------------------------|-------------|-------------|
| `<<` | all chunks fresh              | all-fresh   | all-fresh   |
| `>>` | all chunks mated              | all-mated   | all-mated   |
| `<>` | special fresh, others mated   | **aligned** | flipped     |
| `><` | special mated, others fresh   | flipped     | **aligned** |

Note the polarity mirror: OOAK's special chunk is *special-when-unique*
(fresh), so its role-consistent ("aligned") cell is `<>`; PJ's special pair is
*special-when-paired* (mated), so its aligned cell is `><` (recorded per row
as `metadata.aligned_cell`). Reporting the two uniform corners as
*all-fresh* / *all-mated* avoids naming ambiguity between templates. The cell
is stored in `metadata.scheduled_matrix_cell` / `metadata.computed_matrix_cell`,
which is exactly what the unchanged runtime reads.

## 3. Datasets

The per-seed datasets are large (about 595 MB for the 16 files) and are
**not** in this repository. They are hosted in the same Hugging Face dataset
as the main BoxOffice release, `Boxoffice1280/boxoffice1280`, in the
folders `templates/ooak/` and `templates/pj/` (each with `seeds/`, `manifests/`, `validation/`) — see the dataset card.

Files (per template `<tpl>` in `{ooak, pj}` and seed `<s>` in `{7, 11, 13, 17}`):

- `<tpl>_filtered_v4_balanced_m4_s<s>_full.jsonl` — warmup rows followed by
  the 160 eval rows (the runtime replays this order)
- `<tpl>_filtered_v4_balanced_m4_s<s>_eval.jsonl` — the 160 eval rows

Row counts of the released files (`full` / `eval`): OOAK 338/160, 339/160,
341/160, 343/160; PJ 401/160, 400/160, 401/160, 391/160 for seeds 7, 11, 13,
17. The row schema is identical to the max template, so every existing loader
works.

Place the files under `templates_ooak_pj/seeds/` (git-ignored here) or point
`BOXOFFICE_DIR` at wherever you put them. The matching provenance files are
included in this subtree:

- `manifests/<tpl>_filtered_v4_balanced_m4_s<s>.manifest.json` — generator
  settings, filter survivors per cell, balance histograms. Absolute paths
  recorded at generation time were replaced by the placeholder
  `<BOXOFFICE_BUNDLE_ROOT>`. The filter-model label `mistral` in these files
  refers to the `mistral-7B` weights.
- `validation/<tpl>_filtered_v4_balanced_m4_s<s>_direction_counts.json` —
  per-chunk fresh/mated variant counts after M=4 balancing.

Check a downloaded seed (pure Python, no models):

```bash
cd boxoffice_repro_src/templates_ooak_pj
python experiments/validate_ooak.py  --full seeds/ooak_filtered_v4_balanced_m4_s7_full.jsonl \
  --eval seeds/ooak_filtered_v4_balanced_m4_s7_eval.jsonl --expect-eval-per-cell 40 --check-m4
python experiments/validate_pivot.py --full seeds/pj_filtered_v4_balanced_m4_s7_full.jsonl \
  --eval seeds/pj_filtered_v4_balanced_m4_s7_eval.jsonl   --expect-eval-per-cell 40 --check-m4
```

Both print `PASS` on all released seeds.

## 4. Layout of this subtree

```text
templates_ooak_pj/
  README.md
  OOAK_TEMPLATE_DESIGN.md          template design notes (OOAK)
  PIVOT_TEMPLATE_DESIGN.md         template design notes (PJ)
  experiments/
    v1_helpers/
      generate_one_of_a_kind_v1.py   OOAK core: roles, prefixes, row builder
      generate_pivot_join_v1.py      PJ core (imports the OOAK loaders)
    v2_filtered/
      generate_one_of_a_kind_v2.py   OOAK oversampled warmup + eval generator
      generate_pivot_join_v2.py      PJ oversampled warmup + eval generator
      run_ooak_joint_filtered_probe.sh   phase-2 driver (generate + joint filter)
      run_pj_joint_filtered_probe.sh     phase-2 driver (generate + joint filter)
    v4_balanced/
      build_ooak_v4_balanced.py      phase-3 M=4 warm balance (OOAK)
      build_pj_v4_balanced.py        phase-3 M=4 warm balance (PJ)
    validate_ooak.py                 invariant checker (OOAK)
    validate_pivot.py                invariant checker (PJ)
  scripts/
    02b_regenerate_ooak_filtered_v2.sh   phase 2 wrapper (OOAK)
    02c_regenerate_pj_filtered_v2.sh     phase 2 wrapper (PJ)
    03b_regenerate_ooak_balanced_v4.sh   phase 3 wrapper + validator (OOAK)
    03c_regenerate_pj_balanced_v4.sh     phase 3 wrapper + validator (PJ)
    run_template_matrix.sh               method matrix + aggregation launcher
  runtime/
    aggregate_ooak_results.py        per-cell norm-F1 tables (handles both)
  manifests/                         8 manifests (released seeds)
  validation/                        8 direction-count files (released seeds)
  seeds/                             (git-ignored) put the datasets here
```

Shared code that is **reused from the parent bundle, unmodified**: the
corpus (`../corpus/`), the warm-retrieval metadata
(`../artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl`), the max-template
helpers `../experiments/v1_helpers/generate_one_vs_all_v1.py` (`Film`,
`CELL_CODES`, JSONL I/O), the joint-filter helpers
`../experiments/v2_filtered/{build_question_only_inputs,build_joint_filtered_dataset}.py`,
the filter runner `../src/pilot_eval.py`, and the evaluation runtime
`../runtime/`.

**Import paths.** Every Python file here adds both its local directory and
`<bundle>/experiments/v1_helpers` to `sys.path` (resolved relative to its own
location), so the scripts run from this subtree with no `PYTHONPATH` setup.
If you relocate the subtree, set `BOXOFFICE_BUNDLE_ROOT=/path/to/boxoffice_repro_src`.
The shell drivers export `BOXOFFICE_BUNDLE_ROOT` and a full `PYTHONPATH`
themselves.

## 5. Regenerating the seeds

```text
corpus (parent bundle)
   |  phase 2  [needs the 3 models]   generate oversampled candidates, run each
   v           model on full-context and question-only prompts, keep rows that
 <tpl>_v2_joint_s<seed>_{full,eval}   every model answers with context and none
   |                                  answers without, trim to 40 / cell
   |  phase 3  [no models]            append warmup rows until every eval chunk
   v                                  has >= 2 fresh and >= 2 mated cached variants
 seeds/<tpl>_filtered_v4_balanced_m4_s<seed>_{full,eval}.jsonl
```

Phase 2 re-runs the models, so a regenerated seed reproduces the released one
in distribution, not byte-for-byte. For exact numbers use the released files.

Model weights default to `/data/weights/{qwen3-8B,llama3.1-8BI,mistral-7B}`;
override with `MODEL_WEIGHTS_ROOT=/path/to/weights` or per model with
`QWEN3_8B_PATH`, `LLAMA31_8BI_PATH`, `MISTRAL_7B_PATH`. Accelerator ids go in
`DEVICE_POOL_CSV`; the filter runner's device string is `DEVICE` (default
`cuda:0`, use `npu:0` on an NPU stack). One filter process runs per device id;
`MAX_PARALLEL_JOBS` defaults to the number of ids.

### Phase 2 — generate + joint filter (needs the 3 models)

```bash
cd boxoffice_repro_src/templates_ooak_pj

# OOAK (seeds 7,11,13,17)
SEEDS_CSV="7,11,13,17" DEVICE_POOL_CSV="0,1" \
MODELS_CSV="qwen3_8B,llama3.1-8BI,mistral-7B" \
MODEL_WEIGHTS_ROOT=/data/weights \
bash scripts/02b_regenerate_ooak_filtered_v2.sh
#   -> regenerated/ooak_filtered_v2_joint_probe_<TS>/shared/ooak_v2_joint_s<seed>_{full,eval}.jsonl

# PJ (seeds 7,11,13,17)
SEEDS_CSV="7,11,13,17" DEVICE_POOL_CSV="0,1" \
MODELS_CSV="qwen3_8B,llama3.1-8BI,mistral-7B" \
MODEL_WEIGHTS_ROOT=/data/weights \
bash scripts/02c_regenerate_pj_filtered_v2.sh
#   -> regenerated/pj_filtered_v2_joint_probe_<TS>/shared/pj_v2_joint_s<seed>_{full,eval}.jsonl
```

Settings encoded by the wrappers (these are the values used for the released
seeds): `KEEP_PER_CELL=40`, `PROMPT_VARIANT=closed_world_pairwise`,
`K_MATES=1`, `PER_CELL_ATTEMPTS=50000`, `CHUNK_TARGET_TOKENS=512`,
`EVAL_PER_CELL_OVERSAMPLED=150`, and for OOAK `DUPLICATE_PARTITION=5,4`.
The PJ seeds 13 and 17 were produced with `EVAL_PER_CELL_OVERSAMPLED=100`
(pass it explicitly to match). All knobs are environment variables of the
wrappers and of the underlying `experiments/v2_filtered/run_<tpl>_joint_filtered_probe.sh`.

### Phase 3 — balance warmup to M=4 (no models)

```bash
cd boxoffice_repro_src/templates_ooak_pj

PROBE_ROOT=regenerated/ooak_filtered_v2_joint_probe_<TS> SEEDS_CSV="7,11,13,17" K_MATES=1 \
bash scripts/03b_regenerate_ooak_balanced_v4.sh

PROBE_ROOT=regenerated/pj_filtered_v2_joint_probe_<TS> SEEDS_CSV="7,11,13,17" K_MATES=1 \
bash scripts/03c_regenerate_pj_balanced_v4.sh
```

Each wrapper writes `seeds/`, `manifests/` and `validation/` files for every
seed and then runs the matching validator with
`--expect-eval-per-cell 40 --check-m4`. Source tags default to the values in
the released manifests (`regenerated_ooak_filtered_v2_joint_probe` for OOAK,
`pj_filtered_v2_reference` for PJ); override with `SOURCE_TAG=...`.

The exact per-seed build call the wrappers execute (PJ shown; OOAK is the same
with `build_ooak_v4_balanced.py`, `ooak_v2_joint_...` inputs and `ooak_...`
outputs):

```bash
python experiments/v4_balanced/build_pj_v4_balanced.py \
  --source-full     $PROBE_ROOT/shared/pj_v2_joint_s${seed}_full.jsonl \
  --source-eval     $PROBE_ROOT/shared/pj_v2_joint_s${seed}_eval.jsonl \
  --source-manifest $PROBE_ROOT/shared/pj_v2_joint_s${seed}.manifest.json \
  --source-corpus    ../corpus/source_corpus_100_films.jsonl \
  --canonical-corpus ../corpus/canonical_corpus_100_chunks.jsonl \
  --seed ${seed} --k-mates 1 \
  --out-full   seeds/pj_filtered_v4_balanced_m4_s${seed}_full.jsonl \
  --out-eval   seeds/pj_filtered_v4_balanced_m4_s${seed}_eval.jsonl \
  --out-manifest   manifests/pj_filtered_v4_balanced_m4_s${seed}.manifest.json \
  --out-validation validation/pj_filtered_v4_balanced_m4_s${seed}_direction_counts.json
```

## 6. Running the method matrix

The evaluation runtime is the parent bundle's `runtime/run.sh` +
`run_boxoffice.py`, unchanged: it reads only `prompt_segments`, `question`,
`answers*`, `golden_chunk_indices` and the cell code from `metadata`, all of
which the new templates provide. The corpus-wide passage pool used by the
`cb_k5*` methods is built from the canonical catalogue and is
template-independent (one file set per seed, because the runtime indexes by
seed). It expects the methods bundle at `../../cache_methods_src` (override
with `METHODS_BUNDLE`).

```bash
cd boxoffice_repro_src/templates_ooak_pj

# OOAK: pool build + 10-method matrix at R in {0, 0.15} + aggregation
TEMPLATE=ooak GPUS="0 1 2 3" SEEDS="7 11 13 17" \
MODELS="/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B" \
bash scripts/run_template_matrix.sh

# PJ
TEMPLATE=pj GPUS="0 1 2 3" SEEDS="7 11 13 17" \
MODELS="/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B" \
bash scripts/run_template_matrix.sh
```

Defaults: `BOXOFFICE_DIR=templates_ooak_pj/seeds`,
`OUT_DIR=../runtime/output_<tpl>_v4_m4`, `POOL_DIR=../runtime/pool`,
`METHODS="cb_k0 cb_k0q cb_k5 cb_k5q ccv3_m1_diffkv ccv3_m1_q ccv3_m2_diffkv ccv3_m2_q ccv3_m4_diffkv ccv3_m4_q"`,
`RATIOS="0 0.15"`. Set `BUILD_POOL=0` once the pool exists, `AGGREGATE=0` to
skip the table. `SHARDS` is pinned to 1: the `ccv3_*` (warm) caches accumulate
over the warmup-to-eval order of the `_full` file and sharding breaks the warm
benefit. One Python process runs per accelerator id (`PACK=1`).

The launcher is equivalent to calling the runtime directly:

```bash
cd boxoffice_repro_src/runtime
CACHE_METHODS_SRC=../../cache_methods_src \
BOXOFFICE_DIR=../templates_ooak_pj/seeds \
DATASET_STEM="pj_filtered_v4_balanced_m4_s{seed}_full.jsonl" \
EVAL_STEM="pj_filtered_v4_balanced_m4_s{seed}_eval.jsonl" \
POOL_DIR=pool OUT_DIR=output_pj_v4_m4 \
GPUS="0 1 2 3" SEEDS="7 11 13 17" SHARDS=1 \
METHODS="cb_k0 cb_k0q cb_k5 cb_k5q ccv3_m1_diffkv ccv3_m1_q ccv3_m2_diffkv ccv3_m2_q ccv3_m4_diffkv ccv3_m4_q" \
RATIOS="0 0.15" \
bash run.sh
```

`run.sh` writes one file per (seed, model, method group) and folds them with
`aggregate_groups.py` into the canonical `boxoffice_s<seed>_<model>.json` files
(`<model>` is the basename of the model directory) that the aggregator reads.

## 7. Aggregating to the per-cell tables

```bash
cd boxoffice_repro_src/templates_ooak_pj
python runtime/aggregate_ooak_results.py \
  --out-dir ../runtime/output_pj_v4_m4 \
  --seeds 7 11 13 17 --ratio 0.15 \
  --models llama3.1-8BI mistral-7B qwen3-8B \
  --md-out pj_normf1_R015.md --json-out pj_normf1_R015.json
# OOAK: --out-dir ../runtime/output_ooak_v4_m4, ooak_* outputs
```

Output: one table per model with rows `<<`, `<>`, `><`, `>>`, `AVG` and one
column per method. Per (model, method, cell), **norm-F1** is
`sum(method F1) / sum(baseline F1)` over the eval rows in the cell, pooled over
seeds. The joint filter makes baseline full-prefill F1 close to 1 on every eval
row, so this is essentially the mean method F1. The paper-level number is the
mean of the three per-model tables.

Method key to label: `cb_k0` CB (CacheBlend, cold), `cb_k0q` CB+Q, `cb_k5`
FR (FusionRAG, cold, offline pool k=5), `cb_k5q` FR+Q, `ccv3_m1_diffkv` LM
(LMCache-style, warm), `ccv3_m1_q` LM+Q, `ccv3_m2_diffkv` CC-M2 (CacheCraft,
2 cached versions), `ccv3_m2_q` CC-M2+Q, `ccv3_m4_diffkv` CC-M4, `ccv3_m4_q`
CC-M4+Q. `+Q` = query-driven token selection. `ccv3_m3_*` exists in the runtime
but is not reported.

## 8. Notes

- `OOAK_TEMPLATE_DESIGN.md` describes the duplicate layout as three genre
  triples (3+3+3+1). The released seeds use `DUPLICATE_PARTITION=5,4`
  (5+4+1); the generator and validator accept any partition with groups of
  size >= 2 summing to 9, and the manifests record the layout used.
- The shell drivers export `ASCEND_RT_VISIBLE_DEVICES`, `ASCEND_VISIBLE_DEVICES`
  and `CUDA_VISIBLE_DEVICES` for the chosen device id, like the parent bundle's
  `run_joint_filtered_probe.sh`; pass `DEVICE=npu:0` on an NPU stack.
- Phase 3 and aggregation are pure Python post-processing and run anywhere;
  only phase 2 and the matrix need accelerators and the model weights.
