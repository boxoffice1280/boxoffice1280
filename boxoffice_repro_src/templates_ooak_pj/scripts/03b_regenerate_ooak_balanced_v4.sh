#!/usr/bin/env bash
# Phase 3 (OOAK) — derive balanced-M=4 one-of-a-kind seeds from a phase-2
# ooak joint probe run. Pure post-process; no model invocations.
#
# Layout: TPL_ROOT = <bundle>/templates_ooak_pj (this subtree; seeds/manifests/
# validation are written here), GEN_ROOT = the parent BoxOffice bundle (corpus).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL_ROOT="${TPL_ROOT:-$(cd "${HERE}/.." && pwd)}"
GEN_ROOT="${GEN_ROOT:-$(cd "${TPL_ROOT}/.." && pwd)}"
export BOXOFFICE_BUNDLE_ROOT="${GEN_ROOT}"

PROBE_ROOT="${PROBE_ROOT:?PROBE_ROOT is required (ooak_filtered_v2_joint_probe_<TS> dir from phase 2)}"
SEEDS_CSV="${SEEDS_CSV:-7,11,13,17}"
K_MATES="${K_MATES:-1}"
SOURCE_TAG="${SOURCE_TAG:-regenerated_ooak_filtered_v2_joint_probe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${TPL_ROOT}/seeds" "${TPL_ROOT}/manifests" "${TPL_ROOT}/validation"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for seed in "${SEEDS[@]}"; do
  "${PYTHON_BIN}" "${TPL_ROOT}/experiments/v4_balanced/build_ooak_v4_balanced.py" \
    --source-full "${PROBE_ROOT}/shared/ooak_v2_joint_s${seed}_full.jsonl" \
    --source-eval "${PROBE_ROOT}/shared/ooak_v2_joint_s${seed}_eval.jsonl" \
    --source-manifest "${PROBE_ROOT}/shared/ooak_v2_joint_s${seed}.manifest.json" \
    --source-corpus "${GEN_ROOT}/corpus/source_corpus_100_films.jsonl" \
    --canonical-corpus "${GEN_ROOT}/corpus/canonical_corpus_100_chunks.jsonl" \
    --seed "${seed}" \
    --k-mates "${K_MATES}" \
    --source-tag "${SOURCE_TAG}" \
    --out-full "${TPL_ROOT}/seeds/ooak_filtered_v4_balanced_m4_s${seed}_full.jsonl" \
    --out-eval "${TPL_ROOT}/seeds/ooak_filtered_v4_balanced_m4_s${seed}_eval.jsonl" \
    --out-manifest "${TPL_ROOT}/manifests/ooak_filtered_v4_balanced_m4_s${seed}.manifest.json" \
    --out-validation "${TPL_ROOT}/validation/ooak_filtered_v4_balanced_m4_s${seed}_direction_counts.json"

  "${PYTHON_BIN}" "${TPL_ROOT}/experiments/validate_ooak.py" \
    --full "${TPL_ROOT}/seeds/ooak_filtered_v4_balanced_m4_s${seed}_full.jsonl" \
    --eval "${TPL_ROOT}/seeds/ooak_filtered_v4_balanced_m4_s${seed}_eval.jsonl" \
    --expect-eval-per-cell 40 \
    --check-m4
done
