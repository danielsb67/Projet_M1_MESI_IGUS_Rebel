#!/usr/bin/env bash
# ============================================================================
# nuit_v2_2.sh — Nuit autonome v2.2 : collecte → filtre → conversion ×2 →
#                entraînement ×2 → éval ×2 → analyses (2026-08-14, plan
#                « async-grove », GO donné par le Lot 4 du jour)
# ----------------------------------------------------------------------------
# UNE collecte de 1000 épisodes à TROIS flux caméra (front, wrist 1.20,
# wrist_zoom 0.80 — même pose) → DEUX datasets par conversion (le flux poignet
# choisi sort toujours sous la clé 'wrist') → DEUX entraînements séquentiels
# (recette v2.1b : smolvla_base + rename_map + decay=steps) → DEUX campagnes
# d'éval appariées sur les 20 seeds figés.
#
# Leçons du projet appliquées (toutes payées cher) :
#   • kill_all.sh + ROS_LOCALHOST_ONLY=1 avant CHAQUE lancement de sim ;
#   • JAMAIS de motif pgrep/pkill qui matche sa propre ligne de commande —
#     ici AUCUN pgrep : les attentes se font sur le PID du launch (timeout) ;
#   • SIGINT, jamais kill -9 sur un nœud qui streame (gz_ros2_control) :
#     `timeout --signal=INT --kill-after=60` sur chaque étape sim ;
#   • timeouts en temps MUR partout (l'horloge sim s'est déjà figée un matin :
#     l'orchestrateur détecte maintenant « sim figée » et sort — la boucle de
#     relance ci-dessous reprend la collecte là où elle en était) ;
#   • veille GNOME vérifiée AVANT de partir (incident CUDA step 22000) ;
#   • ne PAS sourcer activate.bash (force un RMW absent) ;
#   • un échec du train B n'efface jamais A : étapes indépendantes, le script
#     continue et le dit (pas de set -e global).
#
# Lancement :  bash src/igus_vla/scripts/nuit_v2_2.sh
# Reprise    : relancer tel quel — chaque étape saute ce qui est déjà fait
#              (collecte : compte les épisodes ; conversion : dossier présent ;
#               train : checkpoint `last` présent → resume).
# ============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$WS/src/igus_vla/.venv/bin/python"
TRAIN_BIN="$WS/src/igus_vla/.venv/bin/lerobot-train"

TARGET="${TARGET:-1000}"               # épisodes GARDÉS visés
RAW="$WS/datasets/raw_v2_2"
DS_A="$WS/datasets/lerobot_v2_2"       # front + wrist 1.20
DS_B="$WS/datasets/lerobot_v2_2z"      # front + wrist_zoom 0.80
OUT_A="$WS/outputs/train/v2_2_smolvla"
OUT_B="$WS/outputs/train/v2_2z_smolvla"
STEPS=27000                            # = recette v2.1b, IDENTIQUE pour A et B
BATCH=32
WORKERS=8
SAVE_FREQ=3000
POSITIONS="$WS/outputs/eval/positions_seed42.json"
MODEL_REF="$WS/outputs/train/v2_1b_smolvla/checkpoints/last/pretrained_model"

# Profil expert v3 « rapide_08 » (speed_sweep, 17,9 s/cycle validé) + blending
# Pilz 5 cm (reco validée sim) + perturbations de récupération (Lot 5, pilote
# du jour : succès >= 85 % exigé avant ce script).
PROFIL_ARGS=(vel_scale_transit:=0.80 acc_scale_transit:=0.55
             vel_scale_fine:=0.35 acc_scale_fine:=0.25 gripper_wait:=0.6
             blend_radius:=0.05
             perturb_prob:=0.28 perturb_min_cm:=1.0 perturb_max_cm:=3.0
             perturb_max_par_episode:=2)

# Config d'éval du jour (Lot 2 : ensembling + rampe ; gains Lot 3 déjà dans la
# copie INSTALLÉE de ros2_controllers.yaml — ce script n'y touche pas).
EVAL_ARGS=(headless:=true device:=cuda n_action_steps:=10
           episode_timeout_s:=45.0 result_margin_s:=45.0
           ensemble:=true ensemble_m:=0.1 ensemble_ramp:=10
           num_episodes:=20 "positions_file:=$POSITIONS")

NUIT_DIR="$WS/outputs/nuit_v2_2/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$NUIT_DIR"
JOURNAL="$NUIT_DIR/nuit.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$JOURNAL"; }
etape_ok()   { log "✅ $*"; }
etape_ko()   { log "❌ $*"; }

# ── Environnement ROS (PAS activate.bash : il force un RMW absent) ───────────
# Les setup.bash de ROS référencent des variables non liées → incompatibles
# avec `set -u`, qu'on suspend le temps du sourcing.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID=10 ROS_LOCALHOST_ONLY=1
unset RMW_IMPLEMENTATION CYCLONEDDS_URI 2>/dev/null || true
# Tous les chemins relatifs des launch (raw_root, report_root, telemetry) sont
# résolus depuis le cwd : on se place à la racine du dépôt, toujours.
cd "$WS"

kill_all() { bash "$WS/kill_all.sh" >>"$JOURNAL" 2>&1; sleep 2; }

compter_episodes() { find "$RAW" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l; }

# ── Préconditions (échouer MAINTENANT, pas à 3 h du matin) ───────────────────
log "══════ NUIT v2.2 — préconditions ══════"
SLEEP_AC=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || echo "?")
if [ "$SLEEP_AC" != "'nothing'" ]; then
  etape_ko "veille GNOME active ($SLEEP_AC ≠ 'nothing') — la nuit serait interrompue (incident CUDA v2.1). ABANDON."
  exit 1
fi
if ! "$VENV_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  etape_ko "CUDA indisponible (sudo modprobe nvidia_uvm ?). ABANDON."
  exit 1
fi
FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 80 ]; then
  etape_ko "disque : ${FREE_GB} Go libres < 80 Go requis (raw ~26 Go + 2 datasets + 2×18 Go de checkpoints). ABANDON."
  exit 1
fi
[ -f "$POSITIONS" ] || { etape_ko "positions figées absentes : $POSITIONS. ABANDON."; exit 1; }
[ -d "$MODEL_REF" ] || log "⚠ checkpoint de référence v2.1b absent ($MODEL_REF) — comparaisons du matin limitées."
log "préconditions OK — veille off, CUDA ok, ${FREE_GB} Go libres, cible $TARGET ép., dossier $NUIT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# Étape 1 — COLLECTE (boucle de relance : une sim figée coûte une relance,
#           jamais la nuit ; le recorder numérote à la suite des épisodes déjà
#           présents, la reprise est donc native)
# ══════════════════════════════════════════════════════════════════════════════
TENTATIVES_MAX=12
STAGNATION_S=900      # 15 min sans NOUVEL épisode (≈ 35× la durée d'un cycle) =
                      # sim figée/coincée → SIGINT et relance. C'est la parade à
                      # la panne du 2026-08-14 (horloge sim gelée) : sans elle,
                      # un gel en début de nuit brûlerait tout le budget mur.
tentative=0
log "══════ Étape 1/6 : collecte ($TARGET ép. gardés, 3 caméras, dédup, perturbations 28 %) ══════"
while :; do
  n=$(compter_episodes)
  reste=$((TARGET - n))
  if [ "$reste" -le 0 ]; then etape_ok "collecte : $n épisodes présents (cible $TARGET)"; break; fi
  tentative=$((tentative + 1))
  if [ "$tentative" -gt "$TENTATIVES_MAX" ]; then
    etape_ko "collecte : $TENTATIVES_MAX relances épuisées, $n/$TARGET épisodes — on continue la nuit avec ce qu'on a."
    break
  fi
  log "collecte, tentative $tentative/$TENTATIVES_MAX : $n présents, $reste restants"
  kill_all
  ros2 launch igus_vla record_demos.launch.py headless:=true \
      raw_root:=datasets/raw_v2_2 num_episodes:="$reste" fill_to_target:=true \
      randomize:=true shutdown_when_done:=true save_failures:=true \
      "${PROFIL_ARGS[@]}" \
      >>"$NUIT_DIR/collecte_$tentative.log" 2>&1 &
  LPID=$!
  # Surveillance de progression sur NOTRE PID enfant (jamais de pgrep : un motif
  # qui matche sa propre ligne de commande a déjà coûté une nuit au projet).
  n_prec=$n
  t_progres=$(date +%s)
  while kill -0 "$LPID" 2>/dev/null; do
    sleep 60
    n_now=$(compter_episodes)
    if [ "$n_now" -gt "$n_prec" ]; then n_prec=$n_now; t_progres=$(date +%s); fi
    if [ $(( $(date +%s) - t_progres )) -ge "$STAGNATION_S" ]; then
      log "⚠ stagnation : aucun épisode depuis $((STAGNATION_S / 60)) min ($n_now ép.) → SIGINT propre puis relance."
      kill -INT "$LPID" 2>/dev/null
      break
    fi
  done
  # Attendre la sortie PROPRE du launch (SIGINT laisse gz_ros2_control sain) ;
  # escalade TERM seulement après 2 min de patience.
  for _ in $(seq 1 60); do kill -0 "$LPID" 2>/dev/null || break; sleep 2; done
  kill -0 "$LPID" 2>/dev/null && { log "⚠ launch encore vivant après 2 min → SIGTERM."; kill -TERM "$LPID" 2>/dev/null; }
  wait "$LPID" 2>/dev/null
  log "collecte tentative $tentative : launch sorti, $(compter_episodes) épisodes au total"
done
kill_all

# ══════════════════════════════════════════════════════════════════════════════
# Étape 2 — FILTRE QUALITÉ (rejet = déplacement vers raw_echecs/, jamais rm)
# ══════════════════════════════════════════════════════════════════════════════
log "══════ Étape 2/6 : filtre qualité ══════"
if python3 "$WS/src/igus_vla/scripts/filtre_qualite.py" "$RAW" --apply \
     >>"$NUIT_DIR/filtre.log" 2>&1; then
  n_final=$(compter_episodes)
  etape_ok "filtre qualité passé — $n_final épisodes gardés (détail : $NUIT_DIR/filtre.log)"
  if [ "$n_final" -lt $((TARGET * 80 / 100)) ]; then
    log "⚠ moins de 80 % de la cible après filtre ($n_final/$TARGET) — vérifier filtre.log au matin."
  fi
else
  etape_ko "filtre qualité en échec (voir filtre.log) — conversion sur le RAW NON filtré, à revoir au matin."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Étape 3 — CONVERSION ×2 (même RAW, flux poignet différent ; la clé de sortie
#           reste 'wrist' dans les deux datasets — uniformité des checkpoints)
# ══════════════════════════════════════════════════════════════════════════════
convertir() { # $1=dossier sortie  $2=repo_id  $3+=options
  local out="$1" repo="$2"; shift 2
  if [ -f "$out/meta/info.json" ]; then log "conversion $repo : déjà faite ($out) — sautée."; return 0; fi
  # PYTHONPATH scopé à CET appel : le venv ne connaît pas le paquet igus_vla
  # (même mécanique que convert_and_train.sh) ; on ne l'exporte pas globalement
  # pour que les nœuds ROS continuent de tourner sur la copie INSTALLÉE.
  PYTHONPATH="$WS/src/igus_vla:${PYTHONPATH:-}" \
    "$VENV_PY" -m igus_vla.to_lerobot_dataset --raw-root "$RAW" --root "$out" \
      --repo-id "$repo" --fps 15 "$@" >>"$NUIT_DIR/conversion.log" 2>&1
}
log "══════ Étape 3/6 : conversion ×2 ══════"
if convertir "$DS_A" "dbal67/igus_rebel_pick_place_v2_2"; then
  etape_ok "dataset A (front + wrist 1.20) : $DS_A"
else
  etape_ko "conversion A en échec (conversion.log) — l'entraînement A sera sauté."
fi
if convertir "$DS_B" "dbal67/igus_rebel_pick_place_v2_2z" --wrist-stream obs_wrist_zoom; then
  etape_ok "dataset B (front + wrist_zoom 0.80) : $DS_B"
else
  etape_ko "conversion B en échec (conversion.log) — l'entraînement B sera sauté."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Étape 4 — ENTRAÎNEMENTS séquentiels (recette v2.1b, STEPS identiques pour A/B
#           → seules les IMAGES poignet diffèrent : c'est la comparaison voulue)
# ══════════════════════════════════════════════════════════════════════════════
RENAME='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

entrainer() { # $1=dataset_root  $2=repo_id  $3=out_dir  $4=étiquette
  local ds="$1" repo="$2" out="$3" nom="$4"
  [ -f "$ds/meta/info.json" ] || { etape_ko "train $nom : dataset absent ($ds) — sauté."; return 1; }
  local args=(--policy.path=lerobot/smolvla_base
              --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false
              --policy.chunk_size=20 --policy.n_action_steps=10
              --policy.scheduler_decay_steps=$STEPS
              --dataset.repo_id="$repo" --dataset.root="$ds"
              --rename_map="$RENAME"
              --batch_size=$BATCH --steps=$STEPS --save_freq=$SAVE_FREQ
              --num_workers=$WORKERS --output_dir="$out" --wandb.enable=false)
  if [ -e "$out/checkpoints/last" ]; then
    log "train $nom : checkpoint existant → REPRISE (steps inchangés, règle LR schedule)."
    args=(--config_path="$out/checkpoints/last/pretrained_model/train_config.json"
          --resume=true --steps=$STEPS --output_dir="$out" --wandb.enable=false)
  fi
  log "train $nom : $STEPS steps, batch $BATCH → $out (~3 h 15 mesuré sur v2.1b)"
  if systemd-inhibit --what=sleep:idle --why="train SmolVLA $nom" \
       "$TRAIN_BIN" "${args[@]}" >>"$NUIT_DIR/train_$nom.log" 2>&1; then
    etape_ok "train $nom terminé : $(readlink -f "$out/checkpoints/last" 2>/dev/null)"
    return 0
  fi
  # Une reprise unique : un crash transitoire (CUDA, OOM ponctuel) ne doit pas
  # coûter la nuit ; un crash reproductible s'arrête là (pas de boucle infinie).
  if [ -e "$out/checkpoints/last" ]; then
    log "⚠ train $nom tombé — UNE reprise depuis $(readlink -f "$out/checkpoints/last")"
    if systemd-inhibit --what=sleep:idle --why="train SmolVLA $nom (reprise)" \
         "$TRAIN_BIN" --config_path="$out/checkpoints/last/pretrained_model/train_config.json" \
         --resume=true --steps=$STEPS --output_dir="$out" --wandb.enable=false \
         >>"$NUIT_DIR/train_$nom.log" 2>&1; then
      etape_ok "train $nom terminé (après reprise)"
      return 0
    fi
  fi
  etape_ko "train $nom en échec définitif (train_$nom.log) — son éval sera sautée."
  return 1
}

log "══════ Étape 4/6 : entraînements séquentiels ══════"
TRAIN_A_OK=0; TRAIN_B_OK=0
entrainer "$DS_A" "dbal67/igus_rebel_pick_place_v2_2"  "$OUT_A" "v2_2"  && TRAIN_A_OK=1
entrainer "$DS_B" "dbal67/igus_rebel_pick_place_v2_2z" "$OUT_B" "v2_2z" && TRAIN_B_OK=1

# ══════════════════════════════════════════════════════════════════════════════
# Étape 5 — ÉVALS appariées (mêmes 20 seeds ; B regarde /wrist_zoom_camera/image
#           sous la clé 'wrist' — c'est ce qu'il a vu à l'entraînement)
# ══════════════════════════════════════════════════════════════════════════════
evaluer() { # $1=checkpoint  $2=run_id  $3+=options supplémentaires
  local ckpt="$1" rid="$2"; shift 2
  kill_all
  timeout --signal=INT --kill-after=60 5400 \
    ros2 launch igus_vla vla_eval.launch.py "${EVAL_ARGS[@]}" \
      model_path:="$ckpt" run_id:="$rid" report_root:="outputs/eval/$rid" "$@" \
      >>"$NUIT_DIR/eval_$rid.log" 2>&1
  local rc=$?
  if [ -f "$WS/outputs/telemetry/$rid/episode.csv" ]; then
    etape_ok "éval $rid terminée (rc=$rc) — $(($(wc -l <"$WS/outputs/telemetry/$rid/episode.csv") - 1)) lignes d'épisode"
  else
    etape_ko "éval $rid : aucune télémétrie produite (rc=$rc, voir eval_$rid.log)"
  fi
}

log "══════ Étape 5/6 : évals appariées (20 seeds) ══════"
[ "$TRAIN_A_OK" = 1 ] && evaluer "$OUT_A/checkpoints/last/pretrained_model" "eval_v2_2"
[ "$TRAIN_B_OK" = 1 ] && evaluer "$OUT_B/checkpoints/last/pretrained_model" "eval_v2_2z" \
                                 wrist_image_topic:=/wrist_zoom_camera/image
kill_all

# ══════════════════════════════════════════════════════════════════════════════
# Étape 6 — ANALYSES (fichiers prêts à lire au matin ; l'appariement par seed et
#           la DMD interdisent d'annoncer un progrès sous le bruit)
# ══════════════════════════════════════════════════════════════════════════════
log "══════ Étape 6/6 : analyses ══════"
ANALYSE="$WS/src/igus_vla/scripts/analyse_telemetrie.py"
for rid in eval_v2_2 eval_v2_2z; do
  [ -d "$WS/outputs/telemetry/$rid" ] && \
    python3 "$ANALYSE" "$rid" >"$NUIT_DIR/rapport_$rid.txt" 2>&1
done
REF_JOUR="${REF_JOUR:-lot4_ref}"       # campagne de référence du jour (Lot 4)
if [ -d "$WS/outputs/telemetry/$REF_JOUR" ] && [ -d "$WS/outputs/telemetry/eval_v2_2" ]; then
  python3 "$ANALYSE" --compare "$REF_JOUR" eval_v2_2 >"$NUIT_DIR/compare_ref_vs_v2_2.txt" 2>&1
fi
if [ -d "$WS/outputs/telemetry/eval_v2_2" ] && [ -d "$WS/outputs/telemetry/eval_v2_2z" ]; then
  python3 "$ANALYSE" --compare eval_v2_2 eval_v2_2z >"$NUIT_DIR/compare_zoom.txt" 2>&1
fi

log "══════ NUIT TERMINÉE ══════"
log "raw          : $RAW ($(compter_episodes) ép.)"
log "datasets     : $DS_A · $DS_B"
log "checkpoints  : $OUT_A (ok=$TRAIN_A_OK) · $OUT_B (ok=$TRAIN_B_OK)"
log "analyses     : $NUIT_DIR/rapport_*.txt · compare_*.txt"
log "→ au matin : lire compare_ref_vs_v2_2.txt (recette+données) et compare_zoom.txt (zoom vs sans-zoom)."
