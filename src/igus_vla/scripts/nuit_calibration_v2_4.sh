#!/usr/bin/env bash
# ============================================================================
# nuit_calibration_v2_4.sh — Nuit 15→16/08 : variance de seed + masse uniforme
# ----------------------------------------------------------------------------
# A. CALIBRATION : retrain v2_2 à l'IDENTIQUE (même data, même recette, autre
#    graine) → éval 20 seeds → la différence mesure le BRUIT d'entraînement,
#    qui borne l'interprétation de toutes les comparaisons passées et futures.
# B. MASSE : +1000 ép. UNIFORMES (le ciblage v2_3 a corrigé localement mais
#    dilué ailleurs — leçon apprise) → corpus 2400 (1000 v2_2 + 400 v2_3 +
#    1000 neufs) → train v2_4 → évals élargies à 50 positions, v2_4 ET v2_2
#    sur les MÊMES 50 (à n=20 la marge est de ±15 points ; pour viser 90 %
#    il faut un thermomètre qui sache le lire).
# Mêmes protections que nuit_v2_2.sh (stagnation, SIGINT, temps mur, reprises).
# ============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$WS/src/igus_vla/.venv/bin/python"
TRAIN_BIN="$WS/src/igus_vla/.venv/bin/lerobot-train"
BATCH=32
RENAME='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

DS_V22="$WS/datasets/lerobot_v2_2"
OUT_SEEDB="$WS/outputs/train/v2_2_seedB_smolvla"
RAW_NEW="$WS/datasets/raw_v2_4"          # +1000 uniformes
RAW_FULL="$WS/datasets/raw_v2_4_full"    # 1000 + 400 + 1000 (hardlinks)
DS_V24="$WS/datasets/lerobot_v2_4"
OUT_V24="$WS/outputs/train/v2_4_smolvla"
POS20="$WS/outputs/eval/positions_seed42.json"
POS50="$WS/outputs/eval/positions_seed42_50.json"   # généré au 1er usage (seed 42,
                                                    # les 20 premières = identiques)
TARGET_NEW="${TARGET_NEW:-1000}"

RUN_DIR="$WS/outputs/nuit_calib_v2_4/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
JOURNAL="$RUN_DIR/nuit.log"
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

entrainer() { # $1=dataset $2=repo_id $3=out $4=steps $5=nom $6+=extra args
  local ds="$1" repo="$2" out="$3" steps="$4" nom="$5"; shift 5
  local args=(--policy.path=lerobot/smolvla_base
              --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false
              --policy.chunk_size=20 --policy.n_action_steps=10
              --policy.scheduler_decay_steps="$steps"
              --dataset.repo_id="$repo" --dataset.root="$ds"
              --rename_map="$RENAME"
              --batch_size=$BATCH --steps="$steps" --save_freq=3000
              --num_workers=8 --output_dir="$out" --wandb.enable=false "$@")
  if [ -e "$out/checkpoints/last" ]; then
    log "train $nom : REPRISE (steps inchangés)."
    args=(--config_path="$out/checkpoints/last/pretrained_model/train_config.json"
          --resume=true --steps="$steps" --output_dir="$out" --wandb.enable=false)
  fi
  log "train $nom : $steps steps → $out"
  if systemd-inhibit --what=sleep:idle --why="train $nom" \
       "$TRAIN_BIN" "${args[@]}" >>"$RUN_DIR/train_$nom.log" 2>&1; then
    log "✅ train $nom terminé"; return 0
  fi
  if [ -e "$out/checkpoints/last" ]; then
    log "⚠ train $nom tombé — UNE reprise."
    systemd-inhibit --what=sleep:idle --why="train $nom reprise" \
      "$TRAIN_BIN" --config_path="$out/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true --steps="$steps" --output_dir="$out" --wandb.enable=false \
      >>"$RUN_DIR/train_$nom.log" 2>&1 && { log "✅ train $nom (reprise)"; return 0; }
  fi
  log "❌ train $nom en échec définitif"; return 1
}

evaluer() { # $1=ckpt $2=run_id $3=num_episodes $4=positions_file
  kill_all
  timeout --signal=INT --kill-after=60 9000 \
    ros2 launch igus_vla vla_eval.launch.py headless:=true device:=cuda \
      n_action_steps:=10 episode_timeout_s:=45.0 result_margin_s:=45.0 \
      ensemble:=true ensemble_m:=0.1 ensemble_ramp:=10 \
      num_episodes:="$3" positions_file:="$4" \
      model_path:="$1" run_id:="$2" report_root:="outputs/eval/$2" \
      >>"$RUN_DIR/eval_$2.log" 2>&1
  [ -f "$WS/outputs/telemetry/$2/episode.csv" ] \
    && log "✅ éval $2 : $(($(wc -l <"$WS/outputs/telemetry/$2/episode.csv") - 1)) ép." \
    || log "❌ éval $2 sans télémétrie"
}

# ── Préconditions ────────────────────────────────────────────────────────────
log "══════ NUIT calibration + v2.4 — préconditions ══════"
"$VENV_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || { log "❌ CUDA indisponible. ABANDON."; exit 1; }
SLEEP_AC=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || echo "?")
[ "$SLEEP_AC" = "'nothing'" ] || { log "❌ veille GNOME active. ABANDON."; exit 1; }
FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 100 ] || { log "❌ ${FREE_GB} Go libres < 100. ABANDON."; exit 1; }
log "préconditions OK (${FREE_GB} Go libres)"

# ══ A1. Train calibration (même data v2_2, même recette, GRAINE 2026) ════════
log "══════ Étape A1 : train calibration seedB (27000 steps, seed 2026) ══════"
entrainer "$DS_V22" "dbal67/igus_rebel_pick_place_v2_2" "$OUT_SEEDB" 27000 "seedB" \
  --seed=2026 || log "⚠ calibration en échec — la nuit continue (B ne dépend pas de A)."

# ══ A2. Éval calibration (20 seeds, appariée vs eval_v2_2) ═══════════════════
if [ -e "$OUT_SEEDB/checkpoints/last" ]; then
  log "══════ Étape A2 : éval calibration (20 positions) ══════"
  evaluer "$OUT_SEEDB/checkpoints/last/pretrained_model" "eval_v2_2_seedB" 20 "$POS20"
fi

# ══ B1. Collecte +1000 UNIFORMES (retry léger 15 %, perturbations 28 %) ══════
STAGNATION_S=900
tentative=0
log "══════ Étape B1 : collecte uniforme (+$TARGET_NEW ép.) ══════"
while :; do
  n=$(compter "$RAW_NEW"); reste=$((TARGET_NEW - n))
  [ "$reste" -le 0 ] && { log "✅ collecte : $n épisodes"; break; }
  tentative=$((tentative + 1))
  [ "$tentative" -gt 12 ] && { log "❌ 12 relances épuisées ($n ép.) — on continue avec ce qu'on a."; break; }
  log "collecte, tentative $tentative : $n présents, $reste restants"
  kill_all
  ros2 launch igus_vla record_demos.launch.py headless:=true \
      raw_root:=datasets/raw_v2_4 num_episodes:="$reste" fill_to_target:=true \
      randomize:=true shutdown_when_done:=true save_failures:=true \
      vel_scale_transit:=0.80 acc_scale_transit:=0.55 \
      vel_scale_fine:=0.35 acc_scale_fine:=0.25 gripper_wait:=0.6 \
      blend_radius:=0.05 \
      perturb_prob:=0.33 perturb_min_cm:=1.0 perturb_max_cm:=3.0 \
      perturb_max_par_episode:=2 \
      retry_demo_prob:=0.15 boost_secteurs_prob:=0.0 \
      >>"$RUN_DIR/collecte_$tentative.log" 2>&1 &
  LPID=$!
  n_prec=$n; t_prog=$(date +%s)
  while kill -0 "$LPID" 2>/dev/null; do
    sleep 60
    n_now=$(compter "$RAW_NEW")
    [ "$n_now" -gt "$n_prec" ] && { n_prec=$n_now; t_prog=$(date +%s); }
    if [ $(( $(date +%s) - t_prog )) -ge "$STAGNATION_S" ]; then
      log "⚠ stagnation → SIGINT et relance."; kill -INT "$LPID" 2>/dev/null; break
    fi
  done
  for _ in $(seq 1 60); do kill -0 "$LPID" 2>/dev/null || break; sleep 2; done
  kill -0 "$LPID" 2>/dev/null && kill -TERM "$LPID" 2>/dev/null
  wait "$LPID" 2>/dev/null
  log "tentative $tentative finie : $(compter "$RAW_NEW") ép."
done
kill_all

# ══ B2. Filtre + fusion (1000 v2_2 + 400 v2_3 + neufs) + conversion ══════════
log "══════ Étape B2 : filtre + fusion + conversion ══════"
python3 "$WS/src/igus_vla/scripts/filtre_qualite.py" "$RAW_NEW" --apply \
  >>"$RUN_DIR/filtre.log" 2>&1 && log "✅ filtre : $(compter "$RAW_NEW") gardés" \
  || log "⚠ filtre en échec — fusion sur RAW non filtré"
python3 - "$WS/datasets/raw_v2_2" "$WS/datasets/raw_v2_3" "$RAW_NEW" "$RAW_FULL" <<'EOF' >>"$JOURNAL" 2>&1
import os, sys, shutil
from pathlib import Path
*srcs, full = map(Path, sys.argv[1:5])
if full.exists(): shutil.rmtree(full)
full.mkdir(parents=True)
idx = 0
for root in srcs:
    for ep in sorted(root.glob("episode_*")):
        dst = full / f"episode_{idx:06d}"; dst.mkdir()
        for item in ep.rglob("*"):
            rel = item.relative_to(ep)
            (dst / rel).mkdir(parents=True, exist_ok=True) if item.is_dir() else os.link(item, dst / rel)
        idx += 1
print(f"fusion : {idx} épisodes")
EOF
log "fusion : $(compter "$RAW_FULL") ép. dans $RAW_FULL"
if [ ! -f "$DS_V24/meta/info.json" ]; then
  PYTHONPATH="$WS/src/igus_vla:${PYTHONPATH:-}" \
    "$VENV_PY" -m igus_vla.to_lerobot_dataset --raw-root "$RAW_FULL" --root "$DS_V24" \
      --repo-id "dbal67/igus_rebel_pick_place_v2_4" --fps 15 \
      >>"$RUN_DIR/conversion.log" 2>&1 \
    && log "✅ dataset : $DS_V24" || { log "❌ conversion en échec. ABANDON de B."; exit 1; }
fi

# ══ B3. Train v2_4 (≈4 époques, plafonné 36000 — durée maîtrisée) ════════════
FRAMES=$(python3 -c "import json; print(json.load(open('$DS_V24/meta/info.json'))['total_frames'])")
STEPS=$(python3 -c "print(max(27000, min(36000, round($FRAMES*4/$BATCH/1000)*1000)))")
log "══════ Étape B3 : train v2_4 ($FRAMES frames → $STEPS steps) ══════"
entrainer "$DS_V24" "dbal67/igus_rebel_pick_place_v2_4" "$OUT_V24" "$STEPS" "v2_4" \
  || { log "❌ train v2_4 KO — évals annulées"; exit 1; }

# ══ B4. Évals élargies : v2_4 ET v2_2 sur les MÊMES 50 positions ═════════════
log "══════ Étape B4 : évals 50 positions (v2_4 puis v2_2) ══════"
evaluer "$OUT_V24/checkpoints/last/pretrained_model" "eval_v2_4_50" 50 "$POS50"
evaluer "$WS/outputs/train/v2_2_smolvla/checkpoints/last/pretrained_model" "eval_v2_2_50" 50 "$POS50"

# ══ Analyses ═════════════════════════════════════════════════════════════════
log "══════ Analyses ══════"
AN="$WS/src/igus_vla/scripts/analyse_telemetrie.py"
python3 "$AN" --compare eval_v2_2 eval_v2_2_seedB >"$RUN_DIR/compare_variance_seed.txt" 2>&1 || true
python3 "$AN" --compare eval_v2_2_50 eval_v2_4_50 >"$RUN_DIR/compare_v2_2_vs_v2_4_50.txt" 2>&1 || true
python3 "$AN" eval_v2_4_50 >"$RUN_DIR/rapport_eval_v2_4_50.txt" 2>&1 || true
log "══════ NUIT TERMINÉE ══════"
log "→ variance de seed : $RUN_DIR/compare_variance_seed.txt"
log "→ v2_4 vs v2_2 (50 pos.) : $RUN_DIR/compare_v2_2_vs_v2_4_50.txt"
