#!/usr/bin/env bash
# train_smolvla.sh — Wrapper for SmolVLA fine-tuning via lerobot-train
#
# Usage:
#   bash scripts/train_smolvla.sh [--steps N] [--batch-size N] [--device cpu|cuda]
#
# Reads hyperparams from config/smolvla.yaml and secrets from .env.
# All paths resolved relative to this script's location (no hardcoded absolutes).
#
# Expected keys in config/smolvla.yaml:
#   policy_path  : lerobot/smolvla_base         # HF model hub path
#   dataset_repo_id: user/igus_rebel_pick_place  # the converted dataset
#   batch_size   : 32
#   steps        : 20000
#   device       : cpu                           # or cuda / auto
#   amp          : false                         # mixed precision (GPU only)
#   wandb_enable : false
#   output_dir   : outputs/train                 # relative to repo root
#   num_workers  : 4
#   save_freq    : 1000                          # checkpoint every N steps
#
# Expected keys in .env:
#   HF_TOKEN     — Hugging Face token (for private datasets)
#   WANDB_API_KEY — Weights & Biases key (only needed if wandb_enable=true)
#   DEVICE       — override device (optional; smolvla.yaml takes precedence)
#   WANDB        — override wandb.enable (optional)

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Resolve repo root (parent of parent of this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"        # src/igus_vla/
REPO_ROOT="$(cd "${PKG_ROOT}/../.." && pwd)"       # repo root

CONFIG_FILE="${PKG_ROOT}/config/smolvla.yaml"
DATASET_CONFIG_FILE="${PKG_ROOT}/config/dataset.yaml"
ENV_FILE="${REPO_ROOT}/.env"

# Prefer the uv venv (lerobot + torch). Used for both lerobot-train and YAML reads.
VENV_PY="${PKG_ROOT}/.venv/bin/python"
PYBIN="python3"
[[ -x "${VENV_PY}" ]] && PYBIN="${VENV_PY}"

echo "=========================================="
echo " SmolVLA training wrapper"
echo " pkg root  : ${PKG_ROOT}"
echo " repo root : ${REPO_ROOT}"
echo " config    : ${CONFIG_FILE}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 2. Load .env (optional)
# ---------------------------------------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
    echo "[INFO] Loading ${ENV_FILE}"
    # Export only lines that look like KEY=VALUE (skip comments and blanks)
    set -o allexport
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$' | grep '=')
    set +o allexport
else
    echo "[WARN] No .env found at ${REPO_ROOT}/.env — HF_TOKEN / WANDB_API_KEY not loaded"
fi

# ---------------------------------------------------------------------------
# 3. Parse config/smolvla.yaml with Python (no yq dependency)
# ---------------------------------------------------------------------------
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[WARN] ${CONFIG_FILE} not found — using built-in defaults"
    POLICY_PATH="lerobot/smolvla_base"
    POLICY_TYPE="smolvla"
    LOAD_VLM="true"
    DATASET_REPO_ID="${DATASET_REPO_ID:-}"
    DATASET_ROOT="${DATASET_ROOT:-}"
    BATCH_SIZE=1
    STEPS=5
    DEVICE="${DEVICE:-cpu}"
    AMP="false"
    WANDB_ENABLE="${WANDB:-false}"
    NUM_WORKERS=0
    SAVE_FREQ=5
    OUTPUT_DIR="${REPO_ROOT}/outputs/train"
else
    # Use Python to read YAML safely (from an arbitrary file)
    _read_yaml_file() {
        # args: <file> <key> <default>
        "${PYBIN}" -c "
import yaml
with open('${1}') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('${2}', '${3}'))
" 2>/dev/null || echo "${3}"
    }
    _read_yaml() { _read_yaml_file "${CONFIG_FILE}" "$1" "$2"; }

    POLICY_PATH=$(_read_yaml policy_path "lerobot/smolvla_base")
    POLICY_TYPE=$(_read_yaml policy_type "smolvla")
    LOAD_VLM=$(_read_yaml load_vlm_weights "true")
    # Dataset id/root live in dataset.yaml (LeRobot convention); smolvla.yaml may
    # still override via dataset_repo_id if present.
    DATASET_REPO_ID=$(_read_yaml dataset_repo_id "")
    if [[ -z "${DATASET_REPO_ID}" && -f "${DATASET_CONFIG_FILE}" ]]; then
        DATASET_REPO_ID=$(_read_yaml_file "${DATASET_CONFIG_FILE}" repo_id "")
    fi
    DATASET_ROOT=""
    if [[ -f "${DATASET_CONFIG_FILE}" ]]; then
        DATASET_ROOT=$(_read_yaml_file "${DATASET_CONFIG_FILE}" root "")
    fi
    BATCH_SIZE=$(_read_yaml batch_size "32")
    STEPS=$(_read_yaml steps "20000")
    DEVICE=$(_read_yaml device "cpu")
    AMP=$(_read_yaml amp "false")
    WANDB_ENABLE=$(_read_yaml wandb_enable "false")
    NUM_WORKERS=$(_read_yaml num_workers "4")
    SAVE_FREQ=$(_read_yaml save_freq "1000")
    _out_dir=$(_read_yaml output_dir "outputs/train")
    # output_dir may be relative → resolve against repo root
    if [[ "${_out_dir}" = /* ]]; then
        OUTPUT_DIR="${_out_dir}"
    else
        OUTPUT_DIR="${REPO_ROOT}/${_out_dir}"
    fi
fi

# ---------------------------------------------------------------------------
# 4. CLI overrides (optional positional-style flags)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)     STEPS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --repo-id)   DATASET_REPO_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

# ENV overrides (DEVICE / WANDB set in .env or shell)
DEVICE="${DEVICE:-cpu}"
WANDB_ENABLE="${WANDB:-${WANDB_ENABLE}}"

# ---------------------------------------------------------------------------
# 5. Validate
# ---------------------------------------------------------------------------
if [[ -z "${DATASET_REPO_ID}" ]]; then
    echo "[ERROR] dataset_repo_id is not set."
    echo "        Set it in config/smolvla.yaml or pass --repo-id USER/NAME"
    exit 1
fi

# Ensure the PARENT exists; lerobot-train (0.4.4) creates --output_dir itself and
# refuses a pre-existing one unless --resume. Do NOT mkdir OUTPUT_DIR here.
mkdir -p "$(dirname "${OUTPUT_DIR}")"

# ---------------------------------------------------------------------------
# 6. Detect lerobot entry point — build CMD_PREFIX array
# ---------------------------------------------------------------------------
VENV_TRAIN="${PKG_ROOT}/.venv/bin/lerobot-train"
if [[ -x "${VENV_TRAIN}" ]]; then
    CMD_PREFIX=("${VENV_TRAIN}")
elif command -v lerobot-train &>/dev/null; then
    CMD_PREFIX=("lerobot-train")
elif "${PYBIN}" -c "import lerobot.scripts.lerobot_train" 2>/dev/null; then
    CMD_PREFIX=("${PYBIN}" "-m" "lerobot.scripts.lerobot_train")
else
    echo "[ERROR] lerobot-train not found and lerobot.scripts.lerobot_train not importable."
    echo "        Install lerobot: uv pip install 'lerobot[smolvla]'"
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. Build command
# ---------------------------------------------------------------------------
# 0.4.4 CLI: dashed flags, --output_dir / --save_freq / --num_workers (NOT hydra/training.*)
# --policy.type=smolvla : features dérivées du dataset (1 caméra "front" + dim 7),
# backbone SmolVLM2 pré-entraîné chargé via --policy.load_vlm_weights.
# (Fine-tuning depuis lerobot/smolvla_base = incompatible embodiment, cf. smolvla.yaml.)
CMD=(
    "${CMD_PREFIX[@]}"
    "--policy.type=${POLICY_TYPE}"
    "--policy.load_vlm_weights=${LOAD_VLM}"
    "--policy.device=${DEVICE}"
    "--policy.push_to_hub=false"
    "--dataset.repo_id=${DATASET_REPO_ID}"
    "--batch_size=${BATCH_SIZE}"
    "--steps=${STEPS}"
    "--save_freq=${SAVE_FREQ}"
    "--num_workers=${NUM_WORKERS}"
    "--output_dir=${OUTPUT_DIR}"
)

# Local dataset root (if dataset.yaml provides one and it's not a pure Hub id)
if [[ -n "${DATASET_ROOT}" ]]; then
    # Resolve relative dataset root against repo root
    if [[ "${DATASET_ROOT}" != /* ]]; then
        DATASET_ROOT="${REPO_ROOT}/${DATASET_ROOT}"
    fi
    CMD+=("--dataset.root=${DATASET_ROOT}")
fi

# Wandb
if [[ "${WANDB_ENABLE}" == "true" || "${WANDB_ENABLE}" == "True" || "${WANDB_ENABLE}" == "1" ]]; then
    CMD+=("--wandb.enable=true")
else
    CMD+=("--wandb.enable=false")
fi

# AMP / mixed precision (GPU only) → --policy.use_amp
if [[ "${AMP}" == "true" || "${AMP}" == "True" ]]; then
    CMD+=("--policy.use_amp=true")
else
    CMD+=("--policy.use_amp=false")
fi

# HF_TOKEN for private datasets
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

# ---------------------------------------------------------------------------
# 8. Print resolved command, then run
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Resolved training command:"
echo "  ${CMD[*]}"
echo ""
echo "[INFO] Device       : ${DEVICE}"
echo "[INFO] Steps        : ${STEPS}"
echo "[INFO] Batch size   : ${BATCH_SIZE}"
echo "[INFO] Output dir   : ${OUTPUT_DIR}"
echo "[INFO] Dataset      : ${DATASET_REPO_ID}"
echo "[INFO] Policy       : ${POLICY_PATH}"
echo "[INFO] WandB        : ${WANDB_ENABLE}"
echo "[INFO] AMP          : ${AMP}"
echo ""
echo "--- Starting training ---"
exec "${CMD[@]}"
