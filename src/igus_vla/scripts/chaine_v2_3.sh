#!/usr/bin/env bash
# ============================================================================
# chaine_v2_3.sh — Chaîne v2.3 « re-saisie + secteurs faibles » (2026-08-15)
# ----------------------------------------------------------------------------
# Cible : pick&place > 90 %. Les 7 échecs de l'éval v2_2 (13/20) se décomposent
# en 5 near-miss (fermeture à 4,3-4,9 cm puis tâtonnement — la re-saisie n'a
# jamais été démontrée) et 2 ratés francs dans les secteurs sous-échantillonnés
# (−122/−128°, +56°, bord intérieur r=0,18).
#
# Chaîne : +400 ép. ciblés (45 % retry démontré, tirage biaisé 35 % secteurs
# faibles, perturbations 28 % sur le reste) → fusion HARDLINKS avec les 1000
# ép. v2_2 (même optique wrist 1,20, mélange légitime — précédent : raw_v2_1)
# → conversion lerobot_v2_3 → entraînement (recette v2.1b, ~4 époques calculées
# sur les frames réelles) → éval appariée 20 seeds vs eval_v2_2.
#
# Mêmes règles opérationnelles que nuit_v2_2.sh (stagnation, SIGINT, temps mur).
# Lancement : bash src/igus_vla/scripts/chaine_v2_3.sh   (détaché conseillé)
# ============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$WS/src/igus_vla/.venv/bin/python"
TRAIN_BIN="$WS/src/igus_vla/.venv/bin/lerobot-train"

TARGET_NEW="${TARGET_NEW:-400}"
RAW_BASE="$WS/datasets/raw_v2_2"          # les 1000 ép. de la nuit (intouchés)
RAW_NEW="$WS/datasets/raw_v2_3"           # la collecte ciblée du jour
RAW_FULL="$WS/datasets/raw_v2_3_full"     # fusion hardlinks (jetable/reconstructible)
DS="$WS/datasets/lerobot_v2_3"
OUT="$WS/outputs/train/v2_3_smolvla"
BATCH=32
POSITIONS="$WS/outputs/eval/positions_seed42.json"

# Collecte ciblée : profil rapide_08 + blend 5 cm (validés), retry démontré 45 %,
# perturbations 28 % (exclusives du retry, cf. pick_place_ia), secteurs faibles 35 %.
PROFIL_ARGS=(vel_scale_transit:=0.80 acc_scale_transit:=0.55
             vel_scale_fine:=0.35 acc_scale_fine:=0.25 gripper_wait:=0.6
             blend_radius:=0.05
             perturb_prob:=0.51 perturb_min_cm:=1.0 perturb_max_cm:=3.0
             perturb_max_par_episode:=2
             retry_demo_prob:=0.45
             boost_secteurs_prob:=0.35)

EVAL_ARGS=(headless:=true device:=cuda n_action_steps:=10
           episode_timeout_s:=45.0 result_margin_s:=45.0
           ensemble:=true ensemble_m:=0.1 ensemble_ramp:=10
           num_episodes:=20 "positions_file:=$POSITIONS")

RUN_DIR="$WS/outputs/chaine_v2_3/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
JOURNAL="$RUN_DIR/chaine.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$JOURNAL"; }

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID=10 ROS_LOCALHOST_ONLY=1
unset RMW_IMPLEMENTATION CYCLONEDDS_URI 2>/dev/null || true
cd "$WS"

kill_all() { bash "$WS/kill_all.sh" >>"$JOURNAL" 2>&1; sleep 2; }
compter() { find "$1" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l; }

# ── Préconditions ────────────────────────────────────────────────────────────
log "══════ CHAÎNE v2.3 — préconditions ══════"
if ! "$VENV_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "❌ CUDA indisponible. ABANDON."; exit 1
fi
[ "$(compter "$RAW_BASE")" -ge 1000 ] || { log "❌ base v2_2 incomplète. ABANDON."; exit 1; }
[ -f "$POSITIONS" ] || { log "❌ positions figées absentes. ABANDON."; exit 1; }
log "préconditions OK — base $(compter "$RAW_BASE") ép., cible +$TARGET_NEW"

# ── Étape 1 : collecte ciblée (boucle de relance + stagnation, cf. nuit) ─────
STAGNATION_S=900
tentative=0
log "══════ Étape 1/5 : collecte ciblée (+$TARGET_NEW ép.) ══════"
while :; do
  n=$(compter "$RAW_NEW"); reste=$((TARGET_NEW - n))
  [ "$reste" -le 0 ] && { log "✅ collecte : $n épisodes"; break; }
  tentative=$((tentative + 1))
  [ "$tentative" -gt 8 ] && { log "❌ 8 relances épuisées ($n ép.) — on continue avec ce qu'on a."; break; }
  log "collecte, tentative $tentative : $n présents, $reste restants"
  kill_all
  ros2 launch igus_vla record_demos.launch.py headless:=true \
      raw_root:=datasets/raw_v2_3 num_episodes:="$reste" fill_to_target:=true \
      randomize:=true shutdown_when_done:=true save_failures:=true \
      "${PROFIL_ARGS[@]}" >>"$RUN_DIR/collecte_$tentative.log" 2>&1 &
  LPID=$!
  n_prec=$n; t_prog=$(date +%s)
  while kill -0 "$LPID" 2>/dev/null; do
    sleep 60
    n_now=$(compter "$RAW_NEW")
    [ "$n_now" -gt "$n_prec" ] && { n_prec=$n_now; t_prog=$(date +%s); }
    if [ $(( $(date +%s) - t_prog )) -ge "$STAGNATION_S" ]; then
      log "⚠ stagnation $((STAGNATION_S / 60)) min → SIGINT et relance."
      kill -INT "$LPID" 2>/dev/null; break
    fi
  done
  for _ in $(seq 1 60); do kill -0 "$LPID" 2>/dev/null || break; sleep 2; done
  kill -0 "$LPID" 2>/dev/null && kill -TERM "$LPID" 2>/dev/null
  wait "$LPID" 2>/dev/null
done
kill_all

# ── Étape 2 : filtre qualité sur la collecte NEUVE seulement ─────────────────
log "══════ Étape 2/5 : filtre qualité (nouveaux épisodes) ══════"
if python3 "$WS/src/igus_vla/scripts/filtre_qualite.py" "$RAW_NEW" --apply \
     >>"$RUN_DIR/filtre.log" 2>&1; then
  log "✅ filtre : $(compter "$RAW_NEW") ép. gardés (détail : filtre.log)"
else
  log "❌ filtre en échec — fusion sur le RAW non filtré, à revoir."
fi

# ── Étape 3 : fusion hardlinks + conversion ──────────────────────────────────
log "══════ Étape 3/5 : fusion hardlinks + conversion ══════"
python3 - "$RAW_BASE" "$RAW_NEW" "$RAW_FULL" <<'EOF' >>"$JOURNAL" 2>&1
import os, sys
from pathlib import Path
base, new, full = map(Path, sys.argv[1:4])
if full.exists():
    import shutil; shutil.rmtree(full)      # reconstructible : jamais de données primaires
full.mkdir(parents=True)
idx = 0
for src_root in (base, new):
    for ep in sorted(src_root.glob("episode_*")):
        dst = full / f"episode_{idx:06d}"
        dst.mkdir()
        for item in ep.rglob("*"):
            rel = item.relative_to(ep)
            if item.is_dir():
                (dst / rel).mkdir(parents=True, exist_ok=True)
            else:
                os.link(item, dst / rel)     # hardlink : 0 octet copié
        idx += 1
print(f"fusion : {idx} épisodes dans {full}")
EOF
log "fusion faite : $(compter "$RAW_FULL") ép. dans $RAW_FULL"
if [ ! -f "$DS/meta/info.json" ]; then
  PYTHONPATH="$WS/src/igus_vla:${PYTHONPATH:-}" \
    "$VENV_PY" -m igus_vla.to_lerobot_dataset --raw-root "$RAW_FULL" --root "$DS" \
      --repo-id "dbal67/igus_rebel_pick_place_v2_3" --fps 15 \
      >>"$RUN_DIR/conversion.log" 2>&1 \
    && log "✅ dataset : $DS" || { log "❌ conversion en échec. ABANDON."; exit 1; }
else
  log "conversion déjà faite — sautée."
fi

# ── Étape 4 : entraînement (steps = ~4 époques sur les frames réelles) ───────
FRAMES=$(python3 -c "import json; print(json.load(open('$DS/meta/info.json'))['total_frames'])")
STEPS=$(python3 -c "print(max(27000, min(36000, round($FRAMES*4/$BATCH/1000)*1000)))")
log "══════ Étape 4/5 : entraînement ($FRAMES frames → $STEPS steps) ══════"
RENAME='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
train_args=(--policy.path=lerobot/smolvla_base
            --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false
            --policy.chunk_size=20 --policy.n_action_steps=10
            --policy.scheduler_decay_steps=$STEPS
            --dataset.repo_id="dbal67/igus_rebel_pick_place_v2_3" --dataset.root="$DS"
            --rename_map="$RENAME"
            --batch_size=$BATCH --steps=$STEPS --save_freq=3000
            --num_workers=8 --output_dir="$OUT" --wandb.enable=false)
if [ -e "$OUT/checkpoints/last" ]; then
  log "checkpoint existant → REPRISE (steps inchangés)."
  train_args=(--config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
              --resume=true --steps=$STEPS --output_dir="$OUT" --wandb.enable=false)
fi
if systemd-inhibit --what=sleep:idle --why="train SmolVLA v2_3" \
     "$TRAIN_BIN" "${train_args[@]}" >>"$RUN_DIR/train_v2_3.log" 2>&1; then
  log "✅ train terminé : $(readlink -f "$OUT/checkpoints/last")"
elif [ -e "$OUT/checkpoints/last" ]; then
  log "⚠ train tombé — UNE reprise."
  systemd-inhibit --what=sleep:idle --why="train v2_3 reprise" \
    "$TRAIN_BIN" --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --steps=$STEPS --output_dir="$OUT" --wandb.enable=false \
    >>"$RUN_DIR/train_v2_3.log" 2>&1 \
    && log "✅ train terminé (après reprise)" || { log "❌ train en échec définitif. ABANDON."; exit 1; }
else
  log "❌ train en échec sans checkpoint. ABANDON."; exit 1
fi

# ── Étape 5 : éval appariée ──────────────────────────────────────────────────
log "══════ Étape 5/5 : éval 20 seeds ══════"
kill_all
timeout --signal=INT --kill-after=60 5400 \
  ros2 launch igus_vla vla_eval.launch.py "${EVAL_ARGS[@]}" \
    model_path:="$OUT/checkpoints/last/pretrained_model" \
    run_id:=eval_v2_3 report_root:=outputs/eval/eval_v2_3 \
    >>"$RUN_DIR/eval_v2_3.log" 2>&1
kill_all
ANALYSE="$WS/src/igus_vla/scripts/analyse_telemetrie.py"
python3 "$ANALYSE" eval_v2_3 >"$RUN_DIR/rapport_eval_v2_3.txt" 2>&1
python3 "$ANALYSE" --compare eval_v2_2 eval_v2_3 >"$RUN_DIR/compare_v2_2_vs_v2_3.txt" 2>&1
log "══════ CHAÎNE v2.3 TERMINÉE ══════"
log "→ lire : $RUN_DIR/compare_v2_2_vs_v2_3.txt"
