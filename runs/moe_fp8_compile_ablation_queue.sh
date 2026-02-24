#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

DEFAULT_ROOT="/mnt/stepeval/yangxiaobo/cache/nanochat/tmp_exp"
if [[ -n "${1:-}" ]]; then
  EXP_DIR="${1}"
elif [[ -f /tmp/nanochat_active_exp_dir.txt ]]; then
  EXP_DIR="$(cat /tmp/nanochat_active_exp_dir.txt)"
else
  EXP_DIR="${DEFAULT_ROOT}/$(date +%Y%m%d_%H%M%S)_moe-fp8-compile-ablation-v5"
fi

mkdir -p "${EXP_DIR}" "${EXP_DIR}/ckpt_root"
printf "%s\n" "${EXP_DIR}" > /tmp/nanochat_active_exp_dir.txt

QUEUE_TAG="$(date +%Y%m%d_%H%M%S)"
REPORT_PREFIX="tmp_report/${QUEUE_TAG}_moe_fp8_compile_ablation_progress"
LAUNCH_LOG="${EXP_DIR}/launch.txt"

common_args=(
  --run dummy
  --use-moe
  --n-experts 8
  --expert-topk 2
  --moe-freq 1
  --expert-hidden-mult 2
  --balance-loss-coeff 0.01
  --depth 12
  --aspect-ratio 64
  --head-dim 64
  --max-seq-len 1024
  --device-batch-size 16
  --num-iterations "${NUM_ITERATIONS:-320}"
  --eval-every -1
  --core-metric-every -1
  --sample-every -1
  --save-every "${SAVE_EVERY:-320}"
  --checkpoint-root "${EXP_DIR}/ckpt_root"
)

run_one() {
  local run_name="$1"
  shift
  local extra_args=("$@")
  local log_path="${EXP_DIR}/${run_name}.log"
  local cmd_path="${EXP_DIR}/${run_name}.cmd.txt"

  {
    echo "[queue ${QUEUE_TAG} $(date -Iseconds)] start ${run_name}"
  } | tee -a "${LAUNCH_LOG}"

  {
    echo "torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m scripts.base_train -- \\"
    printf "  %q \\\n" "${common_args[@]}"
    printf "  %q \\\n" --model-tag "${run_name}"
    printf "  %q \\\n" "${extra_args[@]}"
  } > "${cmd_path}"

  set +e
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m scripts.base_train -- \
    "${common_args[@]}" \
    --model-tag "${run_name}" \
    "${extra_args[@]}" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e

  {
    echo "[queue ${QUEUE_TAG} $(date -Iseconds)] finish ${run_name} status=${status}"
  } | tee -a "${LAUNCH_LOG}"

  python tmp_report/generate_moe_fp8_ablation_report.py \
    --exp-dir "${EXP_DIR}" \
    --out-md "${REPORT_PREFIX}.md"
  cp "${REPORT_PREFIX}.md" "tmp_report/$(date +%Y%m%d_%H%M%S)_moe_fp8_compile_ablation_progress.md"

  return "${status}"
}

run_one "moe_fp8_experts_dynamic_long" \
  --compile-mode dynamic \
  --fp8 \
  --fp8-include-moe-experts || true

run_one "moe_fp8_dynamic_noexpertfp8_long" \
  --compile-mode dynamic \
  --fp8 || true

run_one "moe_bf16_experts_dynamic_long" \
  --compile-mode dynamic || true

run_one "moe_fp8_experts_nocompile_long" \
  --compile-mode none \
  --fp8 \
  --fp8-include-moe-experts || true

run_one "moe_bf16_experts_nocompile_long" \
  --compile-mode none || true

echo "[queue ${QUEUE_TAG} $(date -Iseconds)] queue done" | tee -a "${LAUNCH_LOG}"
