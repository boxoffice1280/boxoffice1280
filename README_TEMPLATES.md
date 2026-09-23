# Additional BoxOffice query templates (OOAK and PJ)

The BoxOffice bundle now also covers two further query templates over the
same 100-film corpus, in addition to the one-vs-all *max* template:

- **OOAK** (one-of-a-kind / genre uniqueness): which candidate is the only
  film whose GENRE appears exactly once among the ten listed dossiers?
- **PJ** (pivot join / multi-hop): which other candidate shares the DIRECTOR
  of the named pivot film?

Everything template-specific is self-contained under

- [`boxoffice_repro_src/templates_ooak_pj/`](boxoffice_repro_src/templates_ooak_pj/README.md)

which holds the generators, joint-filter drivers, M=4 balance builders,
validators, stage wrappers, a matrix launcher, the results aggregator, the
design notes, and the manifests/validation files of the released seeds
(**7, 11, 13, 17**; models `qwen3-8B`, `llama3.1-8BI`, `mistral-7B`). The
existing bundles are unchanged: the new templates run on the unmodified
BoxOffice runtime (`boxoffice_repro_src/runtime/`) and reuse the corpus,
warm-retrieval metadata and filter runner of `boxoffice_repro_src/`.

The per-seed datasets (`{ooak,pj}_filtered_v4_balanced_m4_s{7,11,13,17}_{full,eval}.jsonl`,
about 595 MB) are distributed with the BoxOffice dataset release on Hugging
Face (`Boxoffice1280/boxoffice1280`, under `templates/ooak/seeds/` and `templates/pj/seeds/`,
with manifests and validation counts alongside; see the dataset card) rather than in this repository.

Quick start (after placing the datasets under `boxoffice_repro_src/templates_ooak_pj/seeds/`):

```bash
cd boxoffice_repro_src/templates_ooak_pj
TEMPLATE=ooak GPUS="0 1" MODELS="/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B" \
  bash scripts/run_template_matrix.sh
TEMPLATE=pj   GPUS="0 1" MODELS="/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B" \
  bash scripts/run_template_matrix.sh
```

See the subtree README for the cell semantics, the regeneration commands
(phase 2 joint filter, phase 3 M=4 balance) and the aggregation step.
