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
export AUTO_FALLBACK_ON_COMPILE_FAILURE="${AUTO_FALLBACK_ON_COMPILE_FAILURE:-1}"

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
  local compile_mode="auto"
  local log_path="${EXP_DIR}/${run_name}.log"
  local cmd_path="${EXP_DIR}/${run_name}.cmd.txt"

  for ((i=0; i<${#extra_args[@]}; i++)); do
    if [[ "${extra_args[i]}" == "--compile-mode" ]] && (( i + 1 < ${#extra_args[@]} )); then
      compile_mode="${extra_args[i+1]}"
    fi
  done

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

  if [[ "${AUTO_FALLBACK_ON_COMPILE_FAILURE}" == "1" ]] && [[ "${status}" -ne 0 ]] && [[ "${compile_mode}" != "none" ]] && \
    grep -Eq "InductorError|Expected self.size\\(1\\) to be divisible by 16" "${log_path}"; then
    local fallback_run_name="${run_name}_fallback_none"
    local fallback_log_path="${EXP_DIR}/${fallback_run_name}.log"
    local fallback_cmd_path="${EXP_DIR}/${fallback_run_name}.cmd.txt"
    local fallback_args=("${extra_args[@]}" --compile-mode none)

    {
      echo "[queue ${QUEUE_TAG} $(date -Iseconds)] fallback ${fallback_run_name} trigger=compile_failure primary_status=${status}"
      echo "[queue ${QUEUE_TAG} $(date -Iseconds)] start ${fallback_run_name}"
    } | tee -a "${LAUNCH_LOG}"

    {
      echo "torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} -m scripts.base_train -- \\"
      printf "  %q \\\n" "${common_args[@]}"
      printf "  %q \\\n" --model-tag "${fallback_run_name}"
      printf "  %q \\\n" "${fallback_args[@]}"
    } > "${fallback_cmd_path}"

    set +e
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m scripts.base_train -- \
      "${common_args[@]}" \
      --model-tag "${fallback_run_name}" \
      "${fallback_args[@]}" 2>&1 | tee "${fallback_log_path}"
    local fallback_status=${PIPESTATUS[0]}
    set -e

    {
      echo "[queue ${QUEUE_TAG} $(date -Iseconds)] finish ${fallback_run_name} status=${fallback_status}"
    } | tee -a "${LAUNCH_LOG}"

    python tmp_report/generate_moe_fp8_ablation_report.py \
      --exp-dir "${EXP_DIR}" \
      --out-md "${REPORT_PREFIX}.md"
    cp "${REPORT_PREFIX}.md" "tmp_report/$(date +%Y%m%d_%H%M%S)_moe_fp8_compile_ablation_progress.md"
  fi

  return "${status}"
}

run_profile_default() {
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
}

run_profile_compile_sweep_v1() {
  run_one "moe_bf16_experts_static_long" \
    --compile-mode static || true

  run_one "moe_fp8_noexperts_static_long" \
    --compile-mode static \
    --fp8 || true

  run_one "moe_fp8_experts_static_long" \
    --compile-mode static \
    --fp8 \
    --fp8-include-moe-experts || true

  run_one "moe_fp8_experts_dynamic_long_retest" \
    --compile-mode dynamic \
    --fp8 \
    --fp8-include-moe-experts || true

  run_one "moe_fp8_experts_nocompile_long_retest" \
    --compile-mode none \
    --fp8 \
    --fp8-include-moe-experts || true

  run_one "moe_bf16_experts_nocompile_long_retest" \
    --compile-mode none || true
}

QUEUE_PROFILE="${QUEUE_PROFILE:-default}"
case "${QUEUE_PROFILE}" in
  default)
    run_profile_default
    ;;
  compile_sweep_v1)
    run_profile_compile_sweep_v1
    ;;
  *)
    echo "[queue ${QUEUE_TAG} $(date -Iseconds)] unknown QUEUE_PROFILE=${QUEUE_PROFILE}" | tee -a "${LAUNCH_LOG}"
    exit 2
    ;;
esac

echo "[queue ${QUEUE_TAG} $(date -Iseconds)] queue done" | tee -a "${LAUNCH_LOG}"
