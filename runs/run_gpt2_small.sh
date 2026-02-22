export OMP_NUM_THREADS=1
# export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_BASE_DIR="/mnt/stepeval/yangxiaobo/cache/nanochat"

MODEL_TAG=test_gpt2_small

python -m nanochat.report reset

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --aspect-ratio=64 \
    --head-dim=64 \
    --max-seq-len=1024 \
    --window-pattern="L" \
    --device-batch-size=16 \
    --target-param-data-ratio=8.25 \
    --fp8 \
    --model-tag=${MODEL_TAG} \
    --eval-every=0

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
    --model-tag=${MODEL_TAG} \
    --device-batch-size=16

python -m nanochat.report generate