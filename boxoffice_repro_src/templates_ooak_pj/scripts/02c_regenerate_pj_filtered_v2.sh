#!/usr/bin/env bash
# Phase 2 (PJ) — build the pivot-join filtered-v2 joint shared seeds.
# Calls experiments/v2_filtered/run_pj_joint_filtered_probe.sh, which uses
# pilot_eval.py --method baseline (= BaselineNosep) for the joint filter.
# PJ mirror of 02b_regenerate_ooak_filtered_v2.sh.
#
# Layout: TPL_ROOT = <bundle>/templates_ooak_pj (this subtree; outputs go under
# TPL_ROOT/regenerated), GEN_ROOT = the parent BoxOffice bundle (corpus,
# warm-retrieval artifacts, src/pilot_eval.py).
# Devices: DEVICE_POOL_CSV lists the accelerator ids to use (default 0,1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL_ROOT="${TPL_ROOT:-$(cd "${HERE}/.." && pwd)}"
GEN_ROOT="${GEN_ROOT:-$(cd "${TPL_ROOT}/.." && pwd)}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${TPL_ROOT}/regenerated/pj_filtered_v2_joint_probe_${TS}}"
WARM_FUSION_DOCS="${WARM_FUSION_DOCS:-${GEN_ROOT}/artifacts/rebuilt_warm_k5/fusion_docs_topk5.jsonl}"

if [[ ! -f "${WARM_FUSION_DOCS}" ]]; then
  bash "${GEN_ROOT}/scripts/01_rebuild_warm_fusion_docs.sh"
fi

GEN_ROOT="${GEN_ROOT}" \
TPL_ROOT="${TPL_ROOT}" \
RUN_ROOT="${OUT_ROOT}" \
WARM_FUSION_DOCS="${WARM_FUSION_DOCS}" \
DEVICE_POOL_CSV="${DEVICE_POOL_CSV:-0,1}" \
MODELS_CSV="${MODELS_CSV:-qwen3_8B,llama3.1-8BI,mistral-7B}" \
SEEDS_CSV="${SEEDS_CSV:-7,11,13,17}" \
K_MATES="${K_MATES:-1}" \
PROMPT_VARIANT="${PROMPT_VARIANT:-closed_world_pairwise}" \
EVAL_PER_CELL_OVERSAMPLED="${EVAL_PER_CELL_OVERSAMPLED:-150}" \
KEEP_PER_CELL="${KEEP_PER_CELL:-40}" \
bash "${TPL_ROOT}/experiments/v2_filtered/run_pj_joint_filtered_probe.sh"

echo "${OUT_ROOT}"
