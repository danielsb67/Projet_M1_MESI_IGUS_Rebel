#!/usr/bin/env bash
# ============================================================================
# train_rl_pilote.sh — Pilote RL 2 h : sim Gazebo + learner SAC + actor RLPD
#                      (plan : src/igus_vla/RL_PLAN.md, config : config/rl_sac.json)
# ----------------------------------------------------------------------------
# Topologie (RL_PLAN.md §1) : le LEARNER (lerobot.rl.learner, hors ROS) démarre
# EN PREMIER et ouvre le gRPC :50051 ; l'ACTOR (igus_vla.rl.actor_igus, sous ROS)
# s'y connecte et pousse ses transitions par épisode. Si la sim gèle, seul
# l'actor meurt : on relance sim + actor, le learner garde son replay buffer.
#
# Leçons du projet appliquées (mêmes garde-fous que nuit_v2_2.sh) :
#   • REFUS de démarrer si un train/collecte tourne déjà : scan de /proc (jamais
#     de pgrep dont le motif matcherait sa propre ligne de commande) + refus si
#     nvidia-smi voit un processus de calcul (le GPU est partagé, incident CUDA) ;
#   • kill_all.sh + ROS_LOCALHOST_ONLY=1 avant CHAQUE lancement de sim — et
#     jamais pendant que le learner tourne ? si : ses motifs (vérifiés) ne
#     touchent ni lerobot.rl.learner ni notre actor ;
#   • SIGINT d'abord, escalade TERM après patience — jamais kill -9 sur un nœud
#     qui streame (wedge gz_ros2_control) ;
#   • budget et timeouts en temps MUR partout (l'horloge sim s'est déjà figée) ;
#   • veille GNOME vérifiée AVANT de partir ;
#   • ne PAS sourcer activate.bash (force un RMW absent) ;
#   • pas de set -e global : chaque échec est journalisé et traité.
#
# Lancement :  bash src/igus_vla/scripts/train_rl_pilote.sh
# Variables :  DUREE_S=7200 (budget mur) · RELANCES_MAX=6 (gels sim tolérés)
# NE PAS lancer pendant un train/une collecte : le script refuse, c'est voulu.
# ============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="$WS/src/igus_vla/.venv/bin/python"
CONFIG="$WS/src/igus_vla/config/rl_sac.json"
DS_RL="$WS/datasets/lerobot_v2_2_rl_demos"
DUREE_S="${DUREE_S:-7200}"
RELANCES_MAX="${RELANCES_MAX:-6}"
PORT_LEARNER=50051

RUN_DIR="$WS/outputs/rl/pilote_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
JOURNAL="$RUN_DIR/pilote.log"

log()      { echo "[$(date '+%F %T')] $*" | tee -a "$JOURNAL"; }
etape_ok() { log "✅ $*"; }
etape_ko() { log "❌ $*"; }
abandon()  { etape_ko "$* — ABANDON."; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# MODE DE RÉCOMPENSE (IGUS_RL_REWARD_MODE, défaut sparse = run n°1 à l'identique).
# L'appariement online↔offline exigé par gazebo_env est CÂBLÉ ici : en dense,
# le dataset démos devient la version REJOUÉE EN DENSE et les overrides draccus
# assortis sont passés au learner ET à l'actor (rl_sac.json reste sparse — le
# run sparse demeure reproductible sans toucher à la config). SEUIL_SUCCES :
# un épisode est un succès ssi « Episode reward » ≥ seuil — en dense le shaping
# seul (≤ k_progres·0,8 + r_saisie − r_temps·n ≈ 3,6 max) n'atteint jamais 5,
# r_depose=10 si ; en sparse la récompense vaut exactement 0.0 ou 1.0.
# ─────────────────────────────────────────────────────────────────────────────
REWARD_MODE="${IGUS_RL_REWARD_MODE:-sparse}"
case "$REWARD_MODE" in
  sparse)
    OVERRIDES_DATASET=()
    SEUIL_SUCCES="0.5" ;;
  dense)
    DS_RL="${DS_RL}_dense"
    OVERRIDES_DATASET=(
      "--dataset.root=$DS_RL"
      "--dataset.repo_id=dbal67/igus_rebel_pick_place_v2_2_rl_demos_dense" )
    SEUIL_SUCCES="5" ;;
  *) abandon "IGUS_RL_REWARD_MODE='$REWARD_MODE' invalide (attendu sparse ou dense)" ;;
esac
export IGUS_RL_REWARD_MODE="$REWARD_MODE"   # lu par gazebo_env dans l'actor

# ─────────────────────────────────────────────────────────────────────────────
# GARDE-FOU N°1 — refuser si un train / une collecte / un autre run RL tourne.
# Méthode : lecture DIRECTE de /proc/<pid>/cmdline (argv réel des AUTRES
# processus). Aucun pgrep : un motif qui matche sa propre ligne de commande a
# déjà coûté une nuit au projet ; ici $$ et $PPID sont exclus explicitement et
# notre propre argv ("bash .../train_rl_pilote.sh") ne contient aucun motif.
# ─────────────────────────────────────────────────────────────────────────────
verifier_machine_libre() {
  local occupants="" f pid cmd
  for f in /proc/[0-9]*/cmdline; do
    pid="${f#/proc/}"; pid="${pid%/cmdline}"
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    cmd="$(tr '\0' ' ' < "$f" 2>/dev/null)" || continue
    case "$cmd" in
      *lerobot-train*|*lerobot.rl.learner*|*lerobot.rl.actor*|*igus_vla.rl.actor_igus*|\
      *record_demos.launch*|*record_orchestrator*|*eval_orchestrator*|\
      *vla_policy_node*|*nuit_v2_2.sh*|*nuit_calibration*)
        occupants+="  pid $pid : $cmd"$'\n' ;;
    esac
  done
  if [ -n "$occupants" ]; then
    log "processus concurrents détectés :"
    printf '%s' "$occupants" | tee -a "$JOURNAL"
    abandon "un entraînement/une collecte tourne déjà (le GPU et la sim sont partagés)"
  fi
  # Second verrou, indépendant : tout processus de CALCUL vu par le pilote GPU.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local napps
    napps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
    [ "${napps:-0}" -gt 0 ] && abandon "nvidia-smi voit $napps processus de calcul GPU actifs"
  fi
  # Port learner déjà pris = un run RL précédent traîne encore.
  if ss -ltn 2>/dev/null | grep -q ":$PORT_LEARNER "; then
    abandon "le port $PORT_LEARNER est déjà occupé (learner résiduel ?)"
  fi
  etape_ok "machine libre (aucun train/collecte, GPU sans processus de calcul, port $PORT_LEARNER libre)"
}

log "══════ PILOTE RL — garde-fous et préconditions ══════"
log "mode récompense : $REWARD_MODE (démos : $DS_RL, seuil succès : ≥ $SEUIL_SUCCES)"
verifier_machine_libre

# ── Préconditions (échouer MAINTENANT, pas au milieu du pilote) ──────────────
SLEEP_AC=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || echo "?")
[ "$SLEEP_AC" = "'nothing'" ] || abandon "veille GNOME active ($SLEEP_AC ≠ 'nothing') — incident CUDA v2.1"
[ -f "$CONFIG" ] || abandon "config absente : $CONFIG"
python3 -c "import json; json.load(open('$CONFIG'))" 2>>"$JOURNAL" || abandon "config JSON invalide : $CONFIG"
"$VENV_PY" -c "import torch; assert torch.cuda.is_available()" 2>>"$JOURNAL" || \
  abandon "CUDA indisponible (sudo modprobe nvidia_uvm ?)"

# lerobot.rl importable ? learner.py:55 fait `import grpc` : tant que grpcio n'est
# pas installé dans le venv (`.venv/bin/pip install grpcio==1.73.1`, de jour, réseau —
# RL_PLAN.md §9), TOUTE la topologie actor/learner est infonctionnelle. Diagnostic
# dédié : sans ce test, l'échec ressortirait plus loin comme « modules RL manquants »,
# un message trompeur (la cause serait la dépendance, pas notre code).
"$VENV_PY" -c "import lerobot.rl.learner, lerobot.rl.actor" 2>>"$JOURNAL" || \
  abandon "dépendance grpcio manquante : lerobot.rl inimportable — .venv/bin/pip install grpcio==1.73.1 (RL_PLAN.md §9)"

# Dataset de démos RL : DOIT exister ET porter next.reward — sans lui,
# ReplayBuffer._lerobotdataset_to_transitions lève KeyError (buffer.py:675).
# Produit par igus_vla/rl/demos_to_rl_dataset.py (RL_PLAN.md §4).
[ -f "$DS_RL/meta/info.json" ] || \
  abandon "dataset démos RL absent : $DS_RL — lancer d'abord demos_to_rl_dataset.py --reward-mode $REWARD_MODE (RL_PLAN.md §4)"
python3 - "$DS_RL/meta/info.json" <<'EOF' 2>>"$JOURNAL" || abandon "dataset démos RL sans 'next.reward' (conversion incomplète)"
import json, sys
feats = json.load(open(sys.argv[1]))["features"]
assert "next.reward" in feats, "next.reward manquant"
EOF

# Encodeur visuel pré-téléchargé ? (pas de dépendance réseau en cours de run)
ENCODEUR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['policy'].get('vision_encoder_name') or '')")
if [ -n "$ENCODEUR" ]; then
  CACHE_DIR="$HOME/.cache/huggingface/hub/models--${ENCODEUR//\//--}"
  [ -d "$CACHE_DIR" ] || abandon "encodeur '$ENCODEUR' absent du cache HF — pré-télécharger de jour (RL_PLAN.md §5.1)"
fi

FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 30 ] || abandon "disque : ${FREE_GB} Go libres < 30 Go (checkpoints + buffer sérialisé)"
RAM_DISPO_MB=$(free -m | awk '/^Mem/{print $7}')
[ "${RAM_DISPO_MB:-0}" -ge 14000 ] || \
  abandon "RAM disponible ${RAM_DISPO_MB} Mo < 14 Go (buffers replay ≈ 8 Go + marges, RL_PLAN.md §6)"

# ── Environnement ROS (PAS activate.bash : il force un RMW absent) ───────────
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID=10 ROS_LOCALHOST_ONLY=1
unset RMW_IMPLEMENTATION CYCLONEDDS_URI 2>/dev/null || true
cd "$WS"

# Modules RL importables (env + actor fork) — vérifié APRÈS sourçage ROS car
# l'env importe rclpy. L'import charge lerobot (~15 s), c'est le prix du filet.
PYTHONPATH="$WS/src/igus_vla:${PYTHONPATH:-}" "$VENV_PY" - <<'EOF' 2>>"$JOURNAL" || \
  abandon "modules RL manquants ou cassés : écrire igus_vla/rl/{gazebo_env,actor_igus}.py (RL_PLAN.md §9)"
import igus_vla.rl.gazebo_env    # noqa: F401
import igus_vla.rl.actor_igus    # noqa: F401
EOF
etape_ok "préconditions OK — veille off, CUDA ok, ${FREE_GB} Go disque, ${RAM_DISPO_MB} Mo RAM, démos RL présentes"

kill_all() { bash "$WS/kill_all.sh" >>"$JOURNAL" 2>&1; sleep 2; }

SIM_PID=""; SHIM_PID=""; LEARNER_PID=""; ACTOR_PID=""

# Arrêt ordonné : actor (SIGINT, il ferme proprement env + gRPC) → sim/shim
# (SIGINT, gz_ros2_control sain) → learner en DERNIER (il draine ses queues).
# ⚠ Sur SIGINT le learner NE fait PAS de checkpoint final (learner.py : la
# boucle casse avant le save) — le dernier checkpoint est le dernier multiple
# de save_freq ; assumé pour un pilote (RL_PLAN.md §8.5).
arret_doux() { # $1=pid $2=nom $3=patience_s
  local pid="$1" nom="$2" patience="$3" i
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
  log "arrêt $nom (SIGINT, patience ${patience}s)"
  kill -INT "$pid" 2>/dev/null
  for i in $(seq 1 "$patience"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    log "⚠ $nom encore vivant après ${patience}s → SIGTERM"
    kill -TERM "$pid" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  wait "$pid" 2>/dev/null
}

arret_general() {
  log "── arrêt général ──"
  arret_doux "$ACTOR_PID"   "actor"   45
  arret_doux "$SIM_PID"     "sim"     60
  arret_doux "$SHIM_PID"    "gripper_shim" 15
  arret_doux "$LEARNER_PID" "learner" 120
  kill_all
}
trap 'log "signal reçu"; arret_general; exit 130' INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# Sim + shim, avec attentes BORNÉES EN TEMPS MUR.
# ─────────────────────────────────────────────────────────────────────────────
lire_clock_sec() { timeout 8 ros2 topic echo --once /clock 2>/dev/null | awk '/^ *sec:/{print $2; exit}'; }

demarrer_sim() {
  kill_all
  log "lancement sim headless (check_robot_in_world.launch.py, déjà installée — zéro colcon build)"
  ros2 launch igus_vla check_robot_in_world.launch.py headless:=true \
      >>"$RUN_DIR/sim.log" 2>&1 &
  SIM_PID=$!
  # pose_pub_hz : 5 Hz (défaut shim) en sparse — inchangé run n°1 ; 15 Hz en
  # dense pour que la distance pince↔objet du shaping online soit fraîche à la
  # cadence de contrôle, comme dans le rejeu offline des démos (qui n'a aucun
  # retard de pose — cf. en-tête « RÉCOMPENSE » de demos_to_rl_dataset.py).
  local shim_pose_hz=5.0
  [ "$REWARD_MODE" = "dense" ] && shim_pose_hz=15.0
  ros2 run igus_vla gripper_shim --ros-args \
      -p use_sim_time:=true -p world_name:=default -p object_model:=roulette \
      -p pose_pub_hz:="$shim_pose_hz" \
      >>"$RUN_DIR/shim.log" 2>&1 &
  SHIM_PID=$!

  # Attente stack prête : topics essentiels visibles, borne mur 180 s.
  local t0 fin topics
  t0=$(date +%s); fin=$((t0 + 180))
  while [ "$(date +%s)" -lt "$fin" ]; do
    kill -0 "$SIM_PID" 2>/dev/null || { etape_ko "la sim est morte au démarrage (sim.log)"; return 1; }
    topics="$(timeout 10 ros2 topic list 2>/dev/null)" || topics=""
    if echo "$topics" | grep -q "^/joint_states$" && \
       echo "$topics" | grep -q "^/clock$" && \
       echo "$topics" | grep -q "^/front_camera/image$"; then
      break
    fi
    sleep 5
  done
  [ "$(date +%s)" -lt "$fin" ] || { etape_ko "stack sim incomplète après 180 s (sim.log)"; return 1; }

  # L'horloge sim doit AVANCER (leçon 2026-08-14 : gz figé = topics présents
  # mais /clock immobile). Deux lectures à 4 s d'écart mur.
  local c1 c2
  c1="$(lire_clock_sec)"; sleep 4; c2="$(lire_clock_sec)"
  if [ -z "$c1" ] || [ -z "$c2" ] || [ "$c2" -le "$c1" ] 2>/dev/null; then
    etape_ko "horloge sim immobile (c1='$c1' c2='$c2') — sim considérée figée"
    return 1
  fi
  etape_ok "sim prête (t_sim ${c1}s → ${c2}s), gripper_shim lancé"
  return 0
}

demarrer_actor() {
  log "lancement actor (igus_vla.rl.actor_igus, venv + PYTHONPATH src)"
  PYTHONPATH="$WS/src/igus_vla:${PYTHONPATH:-}" \
    "$VENV_PY" -m igus_vla.rl.actor_igus \
      --config_path="$CONFIG" \
      "${OVERRIDES_DATASET[@]}" \
      --output_dir="$RUN_DIR/actor_$(date +%H%M%S)" \
      >>"$RUN_DIR/actor.log" 2>&1 &
  ACTOR_PID=$!
}

# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — LEARNER (hors ROS, démarre EN PREMIER : il ouvre le gRPC ; l'actor
#           réessaie 30×2 s mais autant partir dans le bon ordre).
# output_dir DISTINCT de celui de l'actor : validate() refuse un dossier déjà
# créé par l'autre processus, et seul le learner écrit des checkpoints.
# ─────────────────────────────────────────────────────────────────────────────
log "══════ Étape 1/4 : learner ══════"
"$VENV_PY" -m lerobot.rl.learner \
    --config_path="$CONFIG" \
    "${OVERRIDES_DATASET[@]}" \
    --output_dir="$RUN_DIR/learner" \
    >>"$RUN_DIR/learner.log" 2>&1 &
LEARNER_PID=$!

T0=$(date +%s); FIN_ATTENTE=$((T0 + 300))   # chargement du buffer offline compris
while [ "$(date +%s)" -lt "$FIN_ATTENTE" ]; do
  kill -0 "$LEARNER_PID" 2>/dev/null || abandon "learner mort au démarrage (learner.log)"
  ss -ltn 2>/dev/null | grep -q ":$PORT_LEARNER " && break
  sleep 3
done
ss -ltn 2>/dev/null | grep -q ":$PORT_LEARNER " || \
  { arret_general; abandon "gRPC :$PORT_LEARNER jamais ouvert en 300 s (learner.log)"; }
etape_ok "learner prêt, gRPC :$PORT_LEARNER ouvert"

# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — SIM + ACTOR
# ─────────────────────────────────────────────────────────────────────────────
log "══════ Étape 2/4 : sim + actor ══════"
demarrer_sim || { arret_general; abandon "sim jamais prête"; }
demarrer_actor

# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — SURVEILLANCE (budget MUR ; gel sim = relance sim+actor, learner
#           conservé — c'est LA propriété de l'architecture, RL_PLAN.md §1.1)
# ─────────────────────────────────────────────────────────────────────────────
log "══════ Étape 3/4 : run pilote (budget ${DUREE_S}s mur, ${RELANCES_MAX} relances sim max) ══════"
T_FIN=$(( $(date +%s) + DUREE_S ))
relances=0
while [ "$(date +%s)" -lt "$T_FIN" ]; do
  sleep 30
  if ! kill -0 "$LEARNER_PID" 2>/dev/null; then
    etape_ko "learner mort en cours de run (learner.log) — rien à entraîner sans lui"
    break
  fi
  if ! kill -0 "$ACTOR_PID" 2>/dev/null || ! kill -0 "$SIM_PID" 2>/dev/null; then
    relances=$((relances + 1))
    log "⚠ actor ou sim tombé (actor vivant=$(kill -0 "$ACTOR_PID" 2>/dev/null && echo oui || echo non), " \
        "sim vivante=$(kill -0 "$SIM_PID" 2>/dev/null && echo oui || echo non)) — relance $relances/$RELANCES_MAX"
    if [ "$relances" -gt "$RELANCES_MAX" ]; then
      etape_ko "trop de relances ($RELANCES_MAX) — NO-GO robustesse sim (RL_PLAN.md §7)"
      break
    fi
    arret_doux "$ACTOR_PID" "actor" 30
    arret_doux "$SIM_PID"   "sim"   60
    arret_doux "$SHIM_PID"  "gripper_shim" 15
    demarrer_sim || { etape_ko "relance sim impossible"; break; }
    demarrer_actor
  fi
done
log "budget mur écoulé (ou sortie anticipée) — relances sim : $relances"

# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — ARRÊT ORDONNÉ + BILAN
# ─────────────────────────────────────────────────────────────────────────────
log "══════ Étape 4/4 : arrêt et bilan ══════"
trap - INT TERM
arret_general

# UNE seule source par processus : la redirection console ($RUN_DIR/actor.log,
# $RUN_DIR/learner.log — cumulatives à travers les relances). lerobot journalise
# CHAQUE ligne DEUX fois (init_logging, utils.py:169-183 : handler console + handler
# fichier logs/*.log) — additionner les deux jeux de fichiers doublait épisodes,
# succès et NaN, et le verdict GO/NO-GO se rendait sur des chiffres faux.
NB_EPISODES=$(grep -c "Episode reward" "$RUN_DIR/actor.log" 2>/dev/null)
# Succès = seuil NUMÉRIQUE sur la valeur (dernier champ de la ligne actor.py:352
# « … Episode reward: <valeur> ») — un simple « ≠ 0.0 » compterait ~100 % de
# succès en dense (quasi tout épisode a une récompense non nulle).
NB_SUCCES=$(grep "Episode reward" "$RUN_DIR/actor.log" 2>/dev/null | \
            awk -v seuil="$SEUIL_SUCCES" '$NF + 0 >= seuil' | wc -l)
DERNIER_OPT=$(grep "Number of optimization step" "$RUN_DIR/learner.log" 2>/dev/null | \
              tail -1)
NB_NAN=$(grep -c "NaN detected" "$RUN_DIR/learner.log" 2>/dev/null)
DERNIER_CKPT=$(readlink -f "$RUN_DIR/learner/checkpoints/last" 2>/dev/null || echo "aucun")

log "── BILAN PILOTE ──"
log "épisodes actor        : ${NB_EPISODES:-0} (dont succès reward ≥ $SEUIL_SUCCES : ${NB_SUCCES:-0})"
log "learner               : ${DERNIER_OPT:-aucune optimisation journalisée}"
log "NaN filtrés           : ${NB_NAN:-0}"
log "relances sim          : $relances"
log "dernier checkpoint    : $DERNIER_CKPT"
# loss_critic / température : le learner ne les écrit QUE via wandb (learner.py:564-565,
# gardé par \`if wandb_logger:\`) — d'où wandb.mode=offline dans la config ; le run est
# local, à tracer avec \`wandb sync\` (critère GO n°3, RL_PLAN.md §7).
log "métriques SAC (loss_critic, température) : run wandb OFFLINE → $RUN_DIR/learner/wandb"
log "→ verdict GO/NO-GO : appliquer les 4 familles de critères de RL_PLAN.md §7"
log "   (plomberie ≥8k transitions/≥5k pas · ≤3 relances · loss stable + ≥1 succès · 15 Hz tenu)"
log "journaux : $RUN_DIR/{pilote,sim,shim,learner,actor}.log"
