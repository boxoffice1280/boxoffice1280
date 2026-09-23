#!/usr/bin/env bash
# Run the cache-reuse method matrix on the OOAK or PJ seeds with the unchanged
# BoxOffice runtime (../../runtime/run.sh + run_boxoffice.py), then aggregate
# the per-cell norm-F1 tables. Mirrors ../../scripts/run_boxoffice_pipeline.sh.
#
# Required:
#   GPUS        space-separated accelerator ids, e.g. GPUS="0 1 2 3"
# Knobs (env):
#   TEMPLATE    ooak | pj                         (default: ooak)
#   SEEDS       space-separated seeds             (default: "7 11 13 17")
#   MODELS      space-separated model dirs        (default: the three paper models)
#   METHODS     space-separated run keys          (default: the 10 reported methods)
#   RATIOS      recompute ratios                  (default: "0 0.15")
#   BOXOFFICE_DIR  where the <tpl>_filtered_v4_balanced_m4_s<seed>_{full,eval}.jsonl
#               files live                        (default: templates_ooak_pj/seeds)
#   OUT_DIR     per-seed result dir               (default: runtime/output_<tpl>_v4_m4)
#   POOL_DIR    corpus-wide pool dir              (default: runtime/pool)
#   BUILD_POOL=1 RUN_EVAL=1 AGGREGATE=1           stage switches
#   PACK / POOL_PACK                              processes per device (default 1)
#   METHODS_BUNDLE                                path to cache_methods_src
#   AGG_RATIO   ratio to tabulate                 (default 0.15)
#
# SHARDS is pinned to 1: the ccv3_* (warm) methods keep state across the
# warmup->eval order of the _full file and sharding breaks the warm benefit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL_ROOT="${TPL_ROOT:-$(cd "${HERE}/.." && pwd)}"
GEN_ROOT="${GEN_ROOT:-$(cd "${TPL_ROOT}/.." && pwd)}"
RUNTIME="${GEN_ROOT}/runtime"
METHODS_BUNDLE="${METHODS_BUNDLE:-$(cd "${GEN_ROOT}/../cache_methods_src" && pwd)}"

TEMPLATE="${TEMPLATE:-ooak}"
case "${TEMPLATE}" in
  ooak|pj) ;;
  *) echo "TEMPLATE must be 'ooak' or 'pj' (got: ${TEMPLATE})" >&2; exit 2 ;;
esac

GPUS="${GPUS:-}"
SEEDS="${SEEDS:-7 11 13 17}"
MODELS="${MODELS:-/data/weights/llama3.1-8BI /data/weights/mistral-7B /data/weights/qwen3-8B}"
METHODS="${METHODS:-cb_k0 cb_k0q cb_k5 cb_k5q ccv3_m1_diffkv ccv3_m1_q ccv3_m2_diffkv ccv3_m2_q ccv3_m4_diffkv ccv3_m4_q}"
RATIOS="${RATIOS:-0 0.15}"
PACK="${PACK:-1}"
POOL_PACK="${POOL_PACK:-1}"
BUILD_POOL="${BUILD_POOL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
AGGREGATE="${AGGREGATE:-1}"
AGG_RATIO="${AGG_RATIO:-0.15}"
BOXOFFICE_DIR="${BOXOFFICE_DIR:-${TPL_ROOT}/seeds}"
OUT_DIR="${OUT_DIR:-${RUNTIME}/output_${TEMPLATE}_v4_m4}"
POOL_DIR="${POOL_DIR:-${RUNTIME}/pool}"
DATASET_STEM="${TEMPLATE}_filtered_v4_balanced_m4_s{seed}_full.jsonl"
EVAL_STEM="${TEMPLATE}_filtered_v4_balanced_m4_s{seed}_eval.jsonl"

if [ -z "${GPUS}" ] && { [ "${BUILD_POOL}" = "1" ] || [ "${RUN_EVAL}" = "1" ]; }; then
  echo "GPUS is required when BUILD_POOL=1 or RUN_EVAL=1" >&2
  exit 2
fi
if [ "${SHARDS:-1}" != "1" ]; then
  echo "SHARDS must stay 1 for the warm (ccv3_*) methods; ignoring SHARDS=${SHARDS}" >&2
fi

for seed in ${SEEDS}; do
  for stem in "${DATASET_STEM}" "${EVAL_STEM}"; do
    f="${BOXOFFICE_DIR}/${stem//\{seed\}/${seed}}"
    if [ ! -f "${f}" ]; then
      echo "missing seed file: ${f}" >&2
      echo "Download the ${TEMPLATE} seeds from the dataset release into ${BOXOFFICE_DIR}" >&2
      echo "(or regenerate them with scripts/02[bc]_* and scripts/03[bc]_*; see README.md)." >&2
      exit 1
    fi
  done
done

mkdir -p "${OUT_DIR}" "${POOL_DIR}"

if [ "${BUILD_POOL}" = "1" ]; then
  # The pool is built from the canonical catalogue and is template-independent;
  # per-seed files are emitted because the runtime indexes them by seed.
  python "${RUNTIME}/build_pool.py" \
    --corpus-jsonl "${GEN_ROOT}/corpus/canonical_corpus_100_chunks.jsonl" \
    --seeds ${SEEDS} --ks 5 10 --npus ${GPUS} --pack "${POOL_PACK}" \
    --out-dir "${POOL_DIR}"
fi

if [ "${RUN_EVAL}" = "1" ]; then
  (
    cd "${RUNTIME}"
    CACHE_METHODS_SRC="${METHODS_BUNDLE}" \
    GPUS="${GPUS}" PACK="${PACK}" MODELS="${MODELS}" SEEDS="${SEEDS}" SHARDS=1 \
    METHODS="${METHODS}" RATIOS="${RATIOS}" \
    BOXOFFICE_DIR="${BOXOFFICE_DIR}" \
    DATASET_STEM="${DATASET_STEM}" EVAL_STEM="${EVAL_STEM}" \
    POOL_DIR="${POOL_DIR}" OUT_DIR="${OUT_DIR}" \
    bash run.sh
  )
fi

if [ "${AGGREGATE}" = "1" ]; then
  MODEL_NAMES=""
  for m in ${MODELS}; do MODEL_NAMES="${MODEL_NAMES} $(basename "${m}")"; done
  rtag="$(printf 'R%03d' "$(python -c "print(round(${AGG_RATIO}*100))")")"
  python "${TPL_ROOT}/runtime/aggregate_ooak_results.py" \
    --out-dir "${OUT_DIR}" --seeds ${SEEDS} --ratio "${AGG_RATIO}" \
    --models ${MODEL_NAMES} \
    --md-out "${OUT_DIR}/${TEMPLATE}_normf1_${rtag}.md" \
    --json-out "${OUT_DIR}/${TEMPLATE}_normf1_${rtag}.json"
fi
