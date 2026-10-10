#!/usr/bin/env bash
set -Eeuo pipefail
# Training and wandb must remain independent of a client-side proxy tunnel.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy WANDB_HTTP_PROXY WANDB_HTTPS_PROXY
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT"

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
    echo "Usage: bash $0 <task|all> [--dry-run] [--config.key value ...]" >&2
    echo "Tasks: classification_the_blocks, insert_the_two_tubes_into_the_rack_one_by_one, place_the_slippers_on_the_shoe_rack" >&2
    exit 2
fi
shift
if [[ "$TASK" == all ]]; then
    if [[ -n "${RUN_ID:-}" ]]; then
        echo "RUN_ID must be unset when using all; each task needs its own run directory." >&2
        exit 2
    fi
    for task in classification_the_blocks insert_the_two_tubes_into_the_rack_one_by_one place_the_slippers_on_the_shoe_rack; do
        bash "$0" "$task" "$@"
    done
    exit 0
fi
case "$TASK" in
    classification_the_blocks) PORT=29641 ;;
    insert_the_two_tubes_into_the_rack_one_by_one) PORT=29642 ;;
    place_the_slippers_on_the_shoe_rack) PORT=29643 ;;
    *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac

DRY_RUN=0
OVERRIDES=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then DRY_RUN=1; else OVERRIDES+=("$arg"); fi
done
PYTHON_BIN="${PYTHON:-python}"
CONFIG="$SCRIPT_DIR/configs/$TASK.yaml"
RUN_NAME="${RUN_ID:-qwen3fast_edl_real_${TASK}_edl_1e-2_state}"
RUN_ROOT="${RUN_ROOT_DIR:-./playground/Checkpoints}"
if [[ "${RESUME:-0}" == 1 ]]; then OVERRIDES+=(--trainer.is_resume True); fi
OVERRIDES+=(--run_id "$RUN_NAME" --run_root_dir "$RUN_ROOT")

COMMAND=("$PYTHON_BIN" -m accelerate.commands.launch
    --config_file "${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
    --num_processes "${NUM_PROCESSES:-4}" --main_process_port "${MAIN_PROCESS_PORT:-$PORT}"
    "$SCRIPT_DIR/train_edl_real.py" --config_yaml "$CONFIG" "${OVERRIDES[@]}")
printf 'Task: %s\nConfig: %s\nRun: %s/%s\n' "$TASK" "$CONFIG" "$RUN_ROOT" "$RUN_NAME"
printf '%q ' "${COMMAND[@]}"; printf '\n'
if (( DRY_RUN )); then exit 0; fi

"$PYTHON_BIN" "$SCRIPT_DIR/preflight_real.py" --config_yaml "$CONFIG" "${OVERRIDES[@]}"
mkdir -p "$RUN_ROOT/$RUN_NAME/wandb"
cp "$0" "$SCRIPT_DIR/train_edl_real.py" "$SCRIPT_DIR/data_config.py" "$CONFIG" "$RUN_ROOT/$RUN_NAME/"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
exec "${COMMAND[@]}"
