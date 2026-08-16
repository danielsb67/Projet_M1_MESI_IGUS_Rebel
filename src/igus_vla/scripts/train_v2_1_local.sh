#!/usr/bin/env bash
# ============================================================================
# train_v2_1_local.sh — Entraînement SmolVLA v2.1 en LOCAL (RTX 5070 Ti)
# ----------------------------------------------------------------------------
# Dataset : datasets/lerobot_v2_1 (379 ép. anneau v3, filtre B — voir
#           TRAIN_LOCAL_V2_1.md et DIAG_REGRESSION_V2.md §4.1)
# Sortie  : outputs/train/v2_1_smolvla/ (convention IHM : détection auto)
# Reprise : RESUME=1 bash train_v2_1_local.sh   (après coupure)
# ============================================================================
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$WS/src/igus_vla/.venv/bin/python"
OUT="$WS/outputs/train/v2_1_smolvla"
DATASET="$WS/datasets/lerobot_v2_1"
STEPS=36000    # batch 16 ≈ 2,6 epochs sur 379 ép. (~224k frames)

[ -x "$VENV_PY" ] || { echo "✗ venv introuvable : $VENV_PY"; exit 1; }
[ -d "$DATASET" ] || { echo "✗ dataset introuvable : $DATASET"; exit 1; }
"$VENV_PY" -c "import torch; assert torch.cuda.is_available(), \
  'CUDA indisponible — voir TRAIN_LOCAL_V2_1.md §1 (driver 570 + torch cu128)'"

# ⚠ RÈGLE LR SCHEDULE : garder --steps IDENTIQUE en cas de reprise (le cosinus
# est dimensionné dessus). Ne jamais relancer avec un steps différent.
ARGS=(
  --policy.type=smolvla --policy.load_vlm_weights=true
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false
  --dataset.repo_id=dbal67/igus_rebel_pick_place_v2_1
  --dataset.root="$DATASET"
  --batch_size=16 --steps=$STEPS --save_freq=2000 --num_workers=4
  --output_dir="$OUT" --wandb.enable=false
)
if [ "${RESUME:-0}" = "1" ]; then
  ARGS=(
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
    --resume=true --steps=$STEPS --output_dir="$OUT" --wandb.enable=false
  )
  echo "▶ REPRISE depuis $OUT/checkpoints/last"
fi

# systemd-inhibit : empêche la veille pendant le run (leçon runs de nuit :
# la veille > durée du run corrompt l'entraînement).
TRAIN="$WS/src/igus_vla/.venv/bin/lerobot-train"
[ -x "$TRAIN" ] || { echo "✗ lerobot-train introuvable : $TRAIN"; exit 1; }
systemd-inhibit --what=sleep --why="train SmolVLA v2.1" \
  "$TRAIN" "${ARGS[@]}" 2>&1 | tee -a "$WS/outputs/train_v2_1.log"
