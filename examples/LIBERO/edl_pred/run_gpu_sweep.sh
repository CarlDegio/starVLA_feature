#!/usr/bin/env bash
# Launch the fixed eight-way sweep only after an all-or-nothing launch preflight.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_ID="all_suites_seed7_v1"
SWEEP_PARENT="${SCRIPT_DIR}/outputs/sweeps"
SWEEP_ROOT="${SWEEP_PARENT}/${SWEEP_ID}"
LAUNCH_RESERVATION="${SWEEP_ROOT}.launch-reservation"
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/default.yaml"
MANIFEST_PATH="${SWEEP_ROOT}/sweep_manifest.json"

# gpu|config filename|run name|encoder|head|position encoding
CANONICAL_RECORDS=(
  "0|all_mlp_flat_softmax.yaml|all_mlp_flat_softmax_seed7|mlp_flat|softmax|none"
  "1|all_mlp_flat_edl.yaml|all_mlp_flat_edl_seed7|mlp_flat|edl|none"
  "2|all_attention_pool_softmax_sinusoidal.yaml|all_attention_pool_softmax_sinusoidal_seed7|token_attention_pool|softmax|sinusoidal"
  "3|all_attention_pool_edl_sinusoidal.yaml|all_attention_pool_edl_sinusoidal_seed7|token_attention_pool|edl|sinusoidal"
  "4|all_self_attention_softmax_sinusoidal.yaml|all_self_attention_softmax_sinusoidal_seed7|token_self_attention|softmax|sinusoidal"
  "5|all_self_attention_edl_sinusoidal.yaml|all_self_attention_edl_sinusoidal_seed7|token_self_attention|edl|sinusoidal"
  "6|all_attention_pool_edl_none.yaml|all_attention_pool_edl_none_seed7|token_attention_pool|edl|none"
  "7|all_attention_pool_edl_learned.yaml|all_attention_pool_edl_learned_seed7|token_attention_pool|edl|learned"
)

assert_no_symlink_ancestors() {
  local path="$1"
  while :; do
    if [[ -L "${path}" ]]; then
      printf 'Refusing symlink ancestor: %s\n' "${path}" >&2
      return 1
    fi
    [[ "${path}" == "/" ]] && return 0
    path="$(dirname "${path}")"
  done
}

cleanup_reservation() {
  rmdir "${LAUNCH_RESERVATION}" 2>/dev/null || true
}

cd "${REPO_ROOT}"
assert_no_symlink_ancestors "${SWEEP_PARENT}"
mkdir -p "${SWEEP_PARENT}"
assert_no_symlink_ancestors "${SWEEP_PARENT}"
if ! mkdir "${LAUNCH_RESERVATION}"; then
  printf 'Sweep launch reservation already exists: %s\n' "${LAUNCH_RESERVATION}" >&2
  exit 1
fi
trap cleanup_reservation EXIT

assert_no_symlink_ancestors "${SWEEP_ROOT}"
if [[ -e "${SWEEP_ROOT}" || -L "${SWEEP_ROOT}" ]]; then
  printf 'Refusing existing sweep root: %s\n' "${SWEEP_ROOT}" >&2
  exit 1
fi
declare -a RECORDS=()
declare -A SEEN_GPU=()
declare -A SEEN_CONFIG=()
declare -A SEEN_RUN=()
declare -A SEEN_OUTPUT=()
declare -A SEEN_LOG=()
preflight_failed=0

for record in "${CANONICAL_RECORDS[@]}"; do
  IFS='|' read -r gpu filename run_name encoder head position <<<"${record}"
  config="${SCRIPT_DIR}/configs/sweep/${filename}"
  output_path="${SWEEP_ROOT}/runs/${run_name}"
  log_path="${SWEEP_ROOT}/logs/${run_name}.log"
  if [[ ! "${gpu}" =~ ^[0-7]$ ]] || [[ -n "${SEEN_GPU[${gpu}]:-}" ]]; then
    printf 'Invalid or duplicate canonical GPU assignment: %s\n' "${gpu}" >&2
    preflight_failed=1
  fi
  SEEN_GPU["${gpu}"]=1
  for seen_key in "${config}" "${run_name}" "${output_path}" "${log_path}"; do
    case "${seen_key}" in
      "${config}") seen_set="SEEN_CONFIG" ;;
      "${run_name}") seen_set="SEEN_RUN" ;;
      "${output_path}") seen_set="SEEN_OUTPUT" ;;
      *) seen_set="SEEN_LOG" ;;
    esac
    declare -n seen="${seen_set}"
    if [[ -n "${seen[${seen_key}]:-}" ]]; then
      printf 'Duplicate canonical sweep value: %s\n' "${seen_key}" >&2
      preflight_failed=1
    fi
    seen["${seen_key}"]=1
  done

  if ! conda run -n starvla python -c '
from pathlib import Path
import os
import stat
import sys
import yaml
from examples.LIBERO.edl_pred.config import load_config

def require_lexical_regular_file(value, description):
    path = Path(value).expanduser().absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"{description} has a symlink ancestor: {ancestor}")
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{description} does not exist: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{description} is not a regular file: {path}")
    return path

config_path = require_lexical_regular_file(sys.argv[1], "canonical sweep config")
default_path = require_lexical_regular_file(sys.argv[2], "default config")
raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected_raw_datasets = {
    "libero_spatial": "../../../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_spatial.hdf5",
    "libero_object": "../../../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_object.hdf5",
    "libero_goal": "../../../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_goal.hdf5",
    "libero_10": "../../../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_10.hdf5",
}
if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict) or raw["data"].get("datasets") != expected_raw_datasets:
    raise ValueError("raw YAML dataset paths do not match canonical relative paths")
for relative_path in expected_raw_datasets.values():
    require_lexical_regular_file(config_path.parent / relative_path, "canonical dataset")

config = load_config(config_path)
default = load_config(default_path)
expected = sys.argv[3:]
expected_name, expected_root, expected_encoder, expected_head, expected_position = expected[:5]
expected_suites = tuple(expected[5:9])
expected_datasets = tuple(require_lexical_regular_file(Path(item), "canonical dataset").resolve() for item in expected[9:13])
if config.run.name != expected_name or config.run.output_root != Path(expected_root).resolve():
    raise ValueError("config run mapping does not match canonical record")
if config.run.overwrite or config.run.seed != 7 or config.data.split_seed != 7:
    raise ValueError("config does not satisfy canonical overwrite/seed policy")
if (config.model.chunk_encoder, config.model.head, config.model.token_position_encoding) != (expected_encoder, expected_head, expected_position):
    raise ValueError("config model does not match canonical record")
if config.data.selected_suites != expected_suites or tuple(config.data.datasets) != expected_suites:
    raise ValueError("config suites do not match canonical record")
if tuple(config.data.datasets[name] for name in expected_suites) != expected_datasets:
    raise ValueError("config datasets do not match canonical record")
if config.training != default.training or config.edl != default.edl or config.loss != default.loss:
    raise ValueError("config shared defaults do not match default.yaml")
for field in ("token_embed_dim", "chunk_embed_dim", "attention_heads", "self_attention_layers", "encoder_dropout", "lstm_hidden_dim", "lstm_layers", "lstm_dropout"):
    if getattr(config.model, field) != getattr(default.model, field):
        raise ValueError(f"config model default differs for {field}")
' "${config}" "${DEFAULT_CONFIG}" "${run_name}" "${SWEEP_ROOT}/runs" "${encoder}" "${head}" "${position}" \
      "libero_spatial" "libero_object" "libero_goal" "libero_10" \
      "${SCRIPT_DIR}/../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_spatial.hdf5" \
      "${SCRIPT_DIR}/../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_object.hdf5" \
      "${SCRIPT_DIR}/../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_goal.hdf5" \
      "${SCRIPT_DIR}/../eval_files/datasets/qwen3fast_libero_all_edl_1e-2_steps_30000_pytorch_model_libero_10.hdf5"; then
    printf 'Canonical config/input preflight failed: %s\n' "${config}" >&2
    preflight_failed=1
  fi
  RECORDS+=("${gpu}|${config}|${run_name}|${output_path}|${log_path}")
done

for gpu in {0..7}; do
  if [[ -z "${SEEN_GPU[${gpu}]:-}" ]]; then
    printf 'Missing required canonical GPU assignment: %s\n' "${gpu}" >&2
    preflight_failed=1
  fi
done

for path in "${MANIFEST_PATH}" "${SWEEP_ROOT}/sweep_summary.json" "${SWEEP_ROOT}/sweep_summary.csv" "${SWEEP_ROOT}/sweep_comparison.png"; do
  assert_no_symlink_ancestors "${path}" || preflight_failed=1
  if [[ -e "${path}" || -L "${path}" ]]; then
    printf 'Refusing existing fixed sweep destination: %s\n' "${path}" >&2
    preflight_failed=1
  fi
done
for record in "${RECORDS[@]}"; do
  IFS='|' read -r gpu config run_name output_path log_path <<<"${record}"
  assert_no_symlink_ancestors "${output_path}" || preflight_failed=1
  assert_no_symlink_ancestors "${log_path}" || preflight_failed=1
  if [[ -e "${output_path}" || -L "${output_path}" || -e "${log_path}" || -L "${log_path}" ]]; then
    printf 'Refusing existing run or log destination: %s ; %s\n' "${output_path}" "${log_path}" >&2
    preflight_failed=1
  fi
done
if (( preflight_failed )); then
  printf 'Sweep preflight failed; no training processes were launched.\n' >&2
  exit 1
fi

if ! gpu_query="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"; then
  printf 'Unable to query GPU free memory with nvidia-smi.\n' >&2
  exit 1
fi
declare -A FREE_MIB=()
gpu_row_count=0
while IFS= read -r row || [[ -n "${row}" ]]; do
  if [[ ! "${row}" =~ ^[[:space:]]*([0-9]+)[[:space:]]*,[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
    printf 'malformed GPU query row: %s\n' "${row}" >&2
    exit 1
  fi
  gpu="${BASH_REMATCH[1]}"
  free_mib="${BASH_REMATCH[2]}"
  if [[ ! "${gpu}" =~ ^[0-7]$ ]]; then
    printf 'GPU query index is outside canonical range 0..7: %s\n' "${gpu}" >&2
    exit 1
  fi
  if [[ -n "${FREE_MIB[${gpu}]:-}" ]]; then
    printf 'duplicate GPU query index: %s\n' "${gpu}" >&2
    exit 1
  fi
  FREE_MIB["${gpu}"]="${free_mib}"
  ((gpu_row_count += 1))
done <<<"${gpu_query}"
if (( gpu_row_count != 8 )); then
  printf 'GPU query must contain exactly eight rows, found %s.\n' "${gpu_row_count}" >&2
  exit 1
fi
for gpu in {0..7}; do
  if [[ -z "${FREE_MIB[${gpu}]:-}" ]] || (( FREE_MIB[${gpu}] < 2048 )); then
    printf 'GPU %s is unavailable or has less than 2048 MiB free.\n' "${gpu}" >&2
    exit 1
  fi
done

mkdir "${SWEEP_ROOT}"
mkdir "${SWEEP_ROOT}/runs" "${SWEEP_ROOT}/logs"
assert_no_symlink_ancestors "${SWEEP_ROOT}/runs"
assert_no_symlink_ancestors "${SWEEP_ROOT}/logs"
set -C  # noclobber: reserve every log without replacing an existing file.
reservation_failed=0
for record in "${RECORDS[@]}"; do
  IFS='|' read -r gpu config run_name output_path log_path <<<"${record}"
  if ! : >"${log_path}"; then
    printf 'Unable to reserve log with no-clobber semantics: %s\n' "${log_path}" >&2
    reservation_failed=1
  fi
done
set +C
if (( reservation_failed )); then
  printf 'Log reservation failed; no training processes were launched.\n' >&2
  exit 1
fi

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
conda run -n starvla python -c '
import json
import os
from pathlib import Path
import sys
import tempfile

manifest_path = Path(sys.argv[1])
payload = {"sweep_id": sys.argv[2], "started_at": sys.argv[3], "runs": []}
for item in sys.argv[4:]:
    gpu, config_path, run_name, output_path, log_path = item.split("|", 4)
    payload["runs"].append({"gpu_index": int(gpu), "config_path": config_path, "run_name": run_name, "output_path": output_path, "log_path": log_path})
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=manifest_path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
try:
    os.link(temporary, manifest_path)
except FileExistsError as error:
    raise RuntimeError(f"manifest reservation collision: {manifest_path}") from error
finally:
    temporary.unlink(missing_ok=True)
' "${MANIFEST_PATH}" "${SWEEP_ID}" "${started_at}" "${RECORDS[@]}"

declare -A PIDS=()
declare -A GPUS=()
for record in "${RECORDS[@]}"; do
  IFS='|' read -r gpu config run_name output_path log_path <<<"${record}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n starvla python "${SCRIPT_DIR}/train.py" --config "${config}" >>"${log_path}" 2>&1 &
  PIDS["${run_name}"]=$!
  GPUS["${run_name}"]="${gpu}"
  printf 'Launched %s on GPU %s (pid %s).\n' "${run_name}" "${gpu}" "${PIDS[${run_name}]}"
done

failed=0
for record in "${RECORDS[@]}"; do
  IFS='|' read -r gpu config run_name output_path log_path <<<"${record}"
  pid="${PIDS[${run_name}]}"
  if wait "${pid}"; then
    status=0
  else
    status=$?
  fi
  printf 'Run %s on GPU %s exited with status %s.\n' "${run_name}" "${GPUS[${run_name}]}" "${status}"
  if (( status != 0 )); then
    failed=1
  fi
done
if (( failed )); then
  printf 'One or more sweep runs failed; partial artifacts were preserved.\n' >&2
  exit 1
fi
