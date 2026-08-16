#!/usr/bin/env bash
# ============================================================================
# train_v2_1b_local.sh — Entraînement SmolVLA v2.1b en LOCAL (RTX 5070 Ti)
# ----------------------------------------------------------------------------
# MÊMES DONNÉES que v2.1 (datasets/lerobot_v2_1, 373 ép. / 212 837 frames,
# caméras front + wrist), RECETTE CORRIGÉE. Objectif : isoler la variable
# « recette » de la variable « données » avant d'investir 7 h de collecte.
#
# Sortie  : outputs/train/v2_1b_smolvla/ (convention IHM : détection auto)
# Reprise : RESUME=1 bash train_v2_1b_local.sh
#
# ============================================================================
# CE QUI CHANGE PAR RAPPORT À train_v2_1_local.sh, ET POURQUOI
# ============================================================================
#
# (1) --policy.path=lerobot/smolvla_base  AU LIEU DE  --policy.type=smolvla
#     ------------------------------------------------------------------
#     v2.1 utilisait `--policy.type=smolvla --policy.load_vlm_weights=true`.
#     `load_vlm_weights` ne charge QUE le VLM (SmolVLM2-500M)… qui est GELÉ
#     pendant l'entraînement (freeze_vision_encoder=True, train_expert_only=
#     True). Les 100 M de paramètres réellement entraînés — l'expert d'action —
#     partaient donc d'une initialisation ALÉATOIRE : tout le pré-entraînement
#     robotique de SmolVLA était jeté.
#
#     Mesuré (2026-08-13, comparaison tenseur par tenseur avec le safetensors
#     de lerobot/smolvla_base) :
#       • from_pretrained('lerobot/smolvla_base') : 500/500 tenseurs identiques
#       • construction from scratch               : 378/500 identiques,
#         122 tenseurs DIFFÉRENTS = 99,9 M params = EXACTEMENT l'ensemble
#         requires_grad=True (action_in_proj, action_out_proj, action_time_mlp,
#         state_proj, les 16 couches de l'expert).
#     Autrement dit : 100 % de ce qu'on entraînait démarrait de zéro.
#
#     L'objection inscrite dans le projet (config/smolvla.yaml:16-20,
#     smoke_test.sh:223-227 : « smolvla_base impose 3 caméras et une action de
#     dim 6 ») est FAUSSE, vérifiée par exécution :
#       • max_state_dim = max_action_dim = 32 des deux côtés ; l'état et
#         l'action sont zero-paddés à 32 (modeling_smolvla.py:472-480), donc
#         nos 7 DOF passent sans réglage ;
#       • make_policy() écrase output_features avec celles du DATASET
#         (factory.py:470) → action de dim 7 dans le checkpoint produit ;
#       • les caméras manquantes sont simplement ignorées quand
#         empty_cameras=0 (modeling_smolvla.py:436-437) : 2 caméras sur 3
#         attendues, aucune erreur, aucune image vide injectée ;
#       • factory.py:525 court-circuite la validation dès qu'un --rename_map
#         est fourni.
#
#     PREUVE EMPIRIQUE (smoke test 200 steps, réglages STRICTEMENT identiques,
#     batch 32, chunk 20, même seed, même LR) :
#         step   pré-entraîné   aléatoire (recette v2.1)
#           20        0.183          1.220
#           40        0.082          0.739
#          100        0.042          0.303
#          200        0.031          0.224      → 7,2× plus bas
#     (le run v2.1 réel démarrait à loss 1.209 au step 200, cohérent avec la
#      colonne « aléatoire »).
#
# (2) --rename_map  (NOUVEAU, indispensable avec --policy.path)
#     ------------------------------------------------------------------
#     smolvla_base déclare observation.images.camera1/2/3 ; notre dataset
#     fournit observation.images.front et .wrist (vérifié dans
#     datasets/lerobot_v2_1/meta/info.json). Sans rename, prepare_images()
#     lève « All image features are missing from the batch ».
#       front → camera1   (caméra fixe)
#       wrist → camera2   (caméra poignet)
#       camera3 : absente, ignorée (empty_cameras=0).
#
# (3) --policy.scheduler_decay_steps = --steps  (=27000)
#     ------------------------------------------------------------------
#     Le défaut est 30 000. Le run v2.1 faisait 36 000 steps : le cosinus
#     touchait son plancher (2,5e-06) au step 30 000 et y restait pour les
#     6 000 derniers steps, soit 16,7 % du budget passé à LR minimal
#     (vérifié dans outputs/train_v2_1.log : lr=2.5e-06 dès step 32K).
#     ATTENTION : LeRobot n'auto-scale le scheduler QUE si steps < decay_steps
#     (optim/schedulers.py:99). Dans le sens inverse — notre cas v2.1 — il ne
#     fait rien. D'où l'égalité imposée ici.
#
# (4) --policy.chunk_size=20 (au lieu de 50) + --policy.n_action_steps=10
#     ------------------------------------------------------------------
#     En horizon fuyant on ne réexécute jamais les pas lointains du chunk.
#     La perte est une moyenne UNIFORME sur le chunk (modeling_smolvla.py:389) :
#     avec chunk=50 et n_action_steps=10, 40 pas sur 50 — soit 80 % du budget
#     de perte — portent sur des actions jamais jouées. chunk=20 ramène cette
#     proportion à 50 % et concentre la capacité sur ce qui est exécuté.
#     PIÈGE : n_action_steps (50 dans la config de smolvla_base) doit être
#     <= chunk_size, sinon ValueError au démarrage. On le fixe à 10, valeur
#     retenue côté déploiement.
#
# (5) --batch_size=32 (au lieu de 16), --num_workers=8 (au lieu de 4)
#     ------------------------------------------------------------------
#     Le débit est invariant du batch entre 16 et 48 (~75 échantillons/s) :
#     doubler le batch divise par deux le nombre de steps à durée égale, avec
#     des gradients moins bruités. VRAM mesurée au smoke : 7,84 Go / 16,3 Go
#     (data_s = 0,012 s → le dataloader n'est pas le goulot).
#
# (6) STEPS = 27 000 (au lieu de 36 000)
#     ------------------------------------------------------------------
#     212 837 frames / batch 32 = 6 651 steps par époque.
#     4 époques = 26 605 steps → arrondi à 27 000 (4,06 époques).
#     v2.1 ne faisait que 2,7 époques (36 000 × 16 / 212 837).
#     Durée attendue : 27 000 / 2,36 step/s ≈ 3 h 10 (débit MESURÉ au smoke,
#     0,411 s/step de calcul + 0,012 s de data).
#     Espace disque : 14 checkpoints × 1,3 Go ≈ 18 Go.
# ============================================================================
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$WS/src/igus_vla/.venv/bin/python"
TRAIN="$WS/src/igus_vla/.venv/bin/lerobot-train"
OUT="$WS/outputs/train/v2_1b_smolvla"
DATASET="$WS/datasets/lerobot_v2_1"
LOG="$WS/outputs/train_v2_1b.log"

# ⚠ RÈGLE LR SCHEDULE : STEPS doit rester IDENTIQUE en cas de reprise (le
# cosinus est dimensionné dessus). Ne jamais relancer avec un STEPS différent.
STEPS=27000        # 4,06 époques à batch 32 sur 212 837 frames
BATCH=32
WORKERS=8
SAVE_FREQ=2000     # 13 checkpoints intermédiaires + le final, ~1,3 Go pièce

# --- garde-fous -------------------------------------------------------------
[ -x "$VENV_PY" ] || { echo "✗ venv introuvable : $VENV_PY"; exit 1; }
[ -x "$TRAIN" ]   || { echo "✗ lerobot-train introuvable : $TRAIN"; exit 1; }
[ -d "$DATASET" ] || { echo "✗ dataset introuvable : $DATASET"; exit 1; }

# nvidia_uvm : après une veille/réveil, le module peut être décroché et CUDA
# devient inutilisable (incident du run v2.1 au step 22 000).
if ! "$VENV_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "✗ CUDA indisponible."
  echo "  → Après un réveil du PC :  sudo modprobe nvidia_uvm"
  echo "  → Sinon voir TRAIN_LOCAL_V2_1.md §1 (driver ≥ 570 + torch cu128)"
  exit 1
fi

FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 25 ]; then
  echo "✗ Disque : ${FREE_GB} Go libres, il en faut ~20 pour les checkpoints."
  exit 1
fi

if [ "${RESUME:-0}" != "1" ] && [ -d "$OUT" ]; then
  echo "✗ $OUT existe déjà."
  echo "  → reprise :  RESUME=1 bash $(basename "$0")"
  echo "  → repartir de zéro : archiver/supprimer ce dossier d'abord."
  exit 1
fi

# --- ⚠ VEILLE DU PC ---------------------------------------------------------
# systemd-inhibit --what=sleep NE BLOQUE PAS la mise en veille *idle* de GNOME.
# C'est exactement ce qui a corrompu le contexte CUDA au step 22 000 du run
# v2.1. On ajoute :idle (respecté par logind) mais ça ne dispense PAS de
# désactiver l'extinction automatique côté GNOME.
cat <<'EOF'

┌──────────────────────────────────────────────────────────────────────────┐
│ ⚠  AVANT DE LAISSER TOURNER (~3 h 10)                                    │
│                                                                          │
│   La veille automatique de GNOME casse le contexte CUDA en plein run     │
│   (incident v2.1, step 22 000). systemd-inhibit ne suffit pas.           │
│                                                                          │
│   Paramètres > Alimentation > Extinction auto de l'écran : Jamais        │
│   Paramètres > Alimentation > Suspension automatique : Désactivé         │
│   (équivalent CLI :                                                      │
│     gsettings set org.gnome.settings-daemon.plugins.power \              │
│       sleep-inactive-ac-type 'nothing')                                  │
│                                                                          │
│   En cas de coupure : RESUME=1 bash train_v2_1b_local.sh                 │
│   (et si CUDA refuse au redémarrage : sudo modprobe nvidia_uvm)          │
└──────────────────────────────────────────────────────────────────────────┘

EOF
if command -v gsettings >/dev/null 2>&1; then
  SLEEP_AC=$(gsettings get org.gnome.settings-daemon.plugins.power \
             sleep-inactive-ac-type 2>/dev/null || echo "?")
  if [ "$SLEEP_AC" != "'nothing'" ] && [ "$SLEEP_AC" != "?" ]; then
    echo "⚠  GNOME sleep-inactive-ac-type = $SLEEP_AC  (attendu : 'nothing')"
    echo "   Le run RISQUE d'être interrompu par la veille. Ctrl-C pour corriger."
    echo ""
  fi
fi

# --- recette ----------------------------------------------------------------
# Le dataset expose observation.images.front / .wrist ; smolvla_base attend
# observation.images.camera1/2/3 (camera3 restera absente, c'est supporté).
RENAME='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

ARGS=(
  --policy.path=lerobot/smolvla_base
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false
  --policy.chunk_size=20 --policy.n_action_steps=10
  --policy.scheduler_decay_steps=$STEPS
  --dataset.repo_id=dbal67/igus_rebel_pick_place_v2_1
  --dataset.root="$DATASET"
  --rename_map="$RENAME"
  --batch_size=$BATCH --steps=$STEPS --save_freq=$SAVE_FREQ --num_workers=$WORKERS
  --output_dir="$OUT" --wandb.enable=false
)

if [ "${RESUME:-0}" = "1" ]; then
  LAST="$OUT/checkpoints/last"
  [ -e "$LAST" ] || { echo "✗ rien à reprendre : $LAST absent"; exit 1; }
  # `last` est un LIEN SYMBOLIQUE : on résout le nom réel (utile pour archiver
  # et pour savoir où l'on repart).
  REAL="$(readlink -f "$LAST")"
  echo "▶ REPRISE depuis $REAL  (step $(basename "$REAL"))"
  # --config_path relit toute la config (rename_map inclus) ; on ne repasse
  # QUE les champs qui doivent rester maîtrisés. STEPS reste identique.
  ARGS=(
    --config_path="$LAST/pretrained_model/train_config.json"
    --resume=true --steps=$STEPS --output_dir="$OUT" --wandb.enable=false
  )
fi

mkdir -p "$(dirname "$LOG")"
echo "▶ sortie      : $OUT"
echo "▶ journal     : $LOG"
echo "▶ steps       : $STEPS (batch $BATCH → 4,06 époques)"
echo "▶ durée est.  : ~3 h 10 à 2,36 step/s (débit mesuré)"
echo ""

# :idle en plus de :sleep (cf. bloc d'avertissement ci-dessus).
systemd-inhibit --what=sleep:idle --why="train SmolVLA v2.1b" \
  "$TRAIN" "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo ""
echo "✓ Terminé. Dernier checkpoint : $(readlink -f "$OUT/checkpoints/last" 2>/dev/null || echo '?')"
