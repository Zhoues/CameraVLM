#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHONPATH_ENTRY="$(cd "${SCRIPT_DIR}/../model/qwen-vl-finetune" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash start_activeqwen3_openai_batched.sh --checkpoint /path/to/checkpoint [options]

Required:
  --checkpoint PATH              Model checkpoint to load.

Options:
  --processor-path PATH          Optional processor/tokenizer path.
  --served-model-name NAME       OpenAI API model name. Default: activeqwen3vl
  --host HOST                    Host to bind. Default: 0.0.0.0
  --port PORT                    Port to bind. Default: 25556
  --conda-env NAME               Conda env to activate. Default: anno
  --conda-sh PATH                Explicit path to conda.sh
  --pythonpath PATH              Extra PYTHONPATH entry. Default: CameraVLM/model/qwen-vl-finetune
  --device-map VALUE             Transformers device_map. Default: auto
  --dtype VALUE                  One of: bfloat16, float16, float32. Default: bfloat16
  --attn-implementation VALUE    Attention implementation. Default: flash_attention_2
  --max-new-tokens-default N     Default max_new_tokens for API requests. Default: 2048
  --max-batch-size N             Maximum micro-batch size. Default: 4
  --batch-wait-ms N              Micro-batch collection window in ms. Default: 20
  --request-timeout-sec N        Per-request server wait timeout. Default: 600
  --max-queue-size N             Request queue size. 0 means unbounded. Default: 0
  --help                         Show this message.
EOF
}

resolve_path() {
  local target="$1"
  if [[ -z "${target}" ]]; then
    return 0
  fi
  if [[ "${target}" = /* ]]; then
    printf '%s\n' "${target}"
  else
    printf '%s\n' "$(cd "$(dirname "${target}")" && pwd)/$(basename "${target}")"
  fi
}

detect_conda_sh() {
  if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base
    conda_base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  return 1
}

activate_conda_env_if_needed() {
  if [[ -z "${CONDA_ENV}" ]]; then
    return 0
  fi
  if ! detect_conda_sh; then
    return 1
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    return 0
  fi
  set +u
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
}

CHECKPOINT="${CHECKPOINT:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-activeqwen3vl}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-25556}"
CONDA_ENV="${CONDA_ENV:-anno}"
CONDA_SH="${CONDA_SH:-}"
PYTHONPATH_ENTRY="${PYTHONPATH_ENTRY:-${DEFAULT_PYTHONPATH_ENTRY}}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-20}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-600}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --processor-path)
      PROCESSOR_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --conda-sh)
      CONDA_SH="$2"
      shift 2
      ;;
    --pythonpath)
      PYTHONPATH_ENTRY="$2"
      shift 2
      ;;
    --device-map)
      DEVICE_MAP="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --attn-implementation)
      ATTN_IMPLEMENTATION="$2"
      shift 2
      ;;
    --max-new-tokens-default)
      MAX_NEW_TOKENS_DEFAULT="$2"
      shift 2
      ;;
    --max-batch-size)
      MAX_BATCH_SIZE="$2"
      shift 2
      ;;
    --batch-wait-ms)
      BATCH_WAIT_MS="$2"
      shift 2
      ;;
    --request-timeout-sec)
      REQUEST_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --max-queue-size)
      MAX_QUEUE_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Checkpoint is required. Pass --checkpoint /path/to/checkpoint." >&2
  exit 1
fi

CHECKPOINT="$(resolve_path "${CHECKPOINT}")"
if [[ -n "${PROCESSOR_PATH}" ]]; then
  PROCESSOR_PATH="$(resolve_path "${PROCESSOR_PATH}")"
fi
if [[ -n "${PYTHONPATH_ENTRY}" ]]; then
  PYTHONPATH_ENTRY="$(resolve_path "${PYTHONPATH_ENTRY}")"
fi

if ! activate_conda_env_if_needed; then
  echo "Warning: could not find conda.sh, running without explicit conda activation." >&2
fi

if [[ -n "${PYTHONPATH_ENTRY}" ]]; then
  export PYTHONPATH="${PYTHONPATH_ENTRY}${PYTHONPATH:+:${PYTHONPATH}}"
fi

CMD=(
  python "${SCRIPT_DIR}/activeqwen3_openai_batched_server.py"
  --checkpoint "${CHECKPOINT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --device-map "${DEVICE_MAP}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --max-new-tokens-default "${MAX_NEW_TOKENS_DEFAULT}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --batch-wait-ms "${BATCH_WAIT_MS}"
  --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"
  --max-queue-size "${MAX_QUEUE_SIZE}"
)

if [[ -n "${PROCESSOR_PATH}" ]]; then
  CMD+=(--processor-path "${PROCESSOR_PATH}")
fi

echo "Launching activeqwen3vl batched OpenAI server:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
