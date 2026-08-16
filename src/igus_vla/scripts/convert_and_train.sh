#!/usr/bin/env bash
# convert_and_train.sh — Pipeline VRAIES DONNÉES : raw enregistré → LeRobotDataset → entraînement.
#
# Enchaîne, sur les épisodes RÉELLEMENT enregistrés par record_demos
# (datasets/raw/episode_*), les étapes 2 et 3 du flux VLA :
#   1. CONVERT : to_lerobot_dataset  (PNG+NPZ → parquet+MP4, LeRobotDataset v2.1)
#   2. TRAIN   : train_smolvla.sh    (lerobot-train, hyperparams de config/smolvla.yaml)
#
# ≠ smoke_test.sh : celui-ci NE génère PAS d'épisodes factices — il consomme tes
# vraies démos. Pour un VRAI entraînement, surcharge les pas/le device, ex. :
#   bash scripts/convert_and_train.sh --steps 20000 --device cuda --batch-size 32
#
# Usage :
#   bash scripts/convert_and_train.sh [--force] [--all]
#                                     [--steps N] [--batch-size N] [--device cpu|cuda]
#
#   --force : ré-convertir en EFFAÇANT un dataset de sortie existant (sinon refus).
#   --all   : convertir aussi les épisodes échoués (défaut : seulement les réussis).
#   Les autres flags (--steps/--batch-size/--device/--repo-id/--output-dir) sont
#   transmis tels quels à train_smolvla.sh.
#
# Tous les chemins/clés viennent de config/dataset.yaml + config/smolvla.yaml + .env
# (aucun chemin absolu en dur).

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Résolution des chemins (parent du parent de ce script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # src/igus_vla/
REPO_ROOT="$(cd "${PKG_ROOT}/../.." && pwd)"         # racine du dépôt

DATASET_CONFIG_FILE="${PKG_ROOT}/config/dataset.yaml"
ENV_FILE="${REPO_ROOT}/.env"

# Le venv uv porte lerobot + torch + le module igus_vla.to_lerobot_dataset.
VENV_PY="${PKG_ROOT}/.venv/bin/python"
PYBIN="python3"
[[ -x "${VENV_PY}" ]] && PYBIN="${VENV_PY}"

# ---------------------------------------------------------------------------
# 2. Séparer les flags de CE script de ceux transmis à train_smolvla.sh
# ---------------------------------------------------------------------------
FORCE=0
CONVERT_EXTRA=()      # ex. --all
TRAIN_ARGS=()         # transmis à train_smolvla.sh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --all)   CONVERT_EXTRA+=("--all"); shift ;;
        *)       TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 3. Lire raw_root / root depuis dataset.yaml (résolus contre la racine dépôt)
# ---------------------------------------------------------------------------
if [[ ! -f "${DATASET_CONFIG_FILE}" ]]; then
    echo "[ERROR] ${DATASET_CONFIG_FILE} introuvable." >&2
    exit 1
fi
_read_yaml() {
    "${PYBIN}" -c "
import yaml
with open('${DATASET_CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('${1}', '${2}'))
" 2>/dev/null || echo "${2}"
}
_resolve() { [[ "$1" = /* ]] && echo "$1" || echo "${REPO_ROOT}/$1"; }

RAW_ROOT="$(_resolve "$(_read_yaml raw_root datasets/raw)")"
OUTPUT_ROOT="$(_resolve "$(_read_yaml root datasets/lerobot)")"
REPO_ID="$(_read_yaml repo_id "")"

echo "=========================================="
echo " Pipeline VRAIES DONNÉES (convert → train)"
echo " raw_root    : ${RAW_ROOT}"
echo " dataset out : ${OUTPUT_ROOT}"
echo " repo_id     : ${REPO_ID}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 4. Vérifier qu'il y a des épisodes bruts à convertir
# ---------------------------------------------------------------------------
shopt -s nullglob
EPISODES=("${RAW_ROOT}"/episode_*)
shopt -u nullglob
if [[ ${#EPISODES[@]} -eq 0 ]]; then
    echo "[ERROR] Aucun épisode dans ${RAW_ROOT}." >&2
    echo "        Enregistre d'abord des démos :" >&2
    echo "        ros2 launch igus_vla record_demos.launch.py headless:=true num_episodes:=20" >&2
    exit 1
fi
echo "[INFO] ${#EPISODES[@]} épisode(s) brut(s) trouvé(s)."

# ---------------------------------------------------------------------------
# 5. Gérer un dataset de sortie pré-existant (create() refuse un dossier non vide)
# ---------------------------------------------------------------------------
if [[ -d "${OUTPUT_ROOT}" ]] && [[ -n "$(ls -A "${OUTPUT_ROOT}" 2>/dev/null)" ]]; then
    if [[ "${FORCE}" -eq 1 ]]; then
        echo "[INFO] --force : suppression du dataset existant ${OUTPUT_ROOT}"
        rm -rf "${OUTPUT_ROOT}"
    else
        echo "[ERROR] Un dataset existe déjà à ${OUTPUT_ROOT}." >&2
        echo "        Relance avec --force pour l'écraser (ré-conversion)." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 6. CONVERT (pur Python, sans ROS)
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/2] Conversion en LeRobotDataset ---"
export PYTHONPATH="${PKG_ROOT}:${PYTHONPATH:-}"
CONVERT_CMD=("${PYBIN}" "-m" "igus_vla.to_lerobot_dataset" "--config" "${DATASET_CONFIG_FILE}")
[[ -f "${ENV_FILE}" ]] && CONVERT_CMD+=("--env-file" "${ENV_FILE}")
[[ ${#CONVERT_EXTRA[@]} -gt 0 ]] && CONVERT_CMD+=("${CONVERT_EXTRA[@]}")
echo "  ${CONVERT_CMD[*]}"
"${CONVERT_CMD[@]}"

# ---------------------------------------------------------------------------
# 7. TRAIN (réutilise le wrapper validé ; hyperparams de smolvla.yaml + overrides)
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/2] Entraînement SmolVLA ---"
TRAIN_CMD=("bash" "${SCRIPT_DIR}/train_smolvla.sh")
[[ ${#TRAIN_ARGS[@]} -gt 0 ]] && TRAIN_CMD+=("${TRAIN_ARGS[@]}")
echo "  ${TRAIN_CMD[*]}"
exec "${TRAIN_CMD[@]}"
