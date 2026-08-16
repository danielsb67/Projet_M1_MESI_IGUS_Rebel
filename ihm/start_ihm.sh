#!/usr/bin/env bash
# Lanceur de l'IHM tactile du robot Igus Rebel.
# Source l'environnement ROS2 du workspace puis démarre l'application Tkinter.
# Détection dynamique des chemins — aucun chemin codé en dur.
#
# NB : pas de « set -u » — les scripts setup.bash de ROS2/colcon référencent
# des variables non définies et le feraient échouer au sourcing.

set -e

# Dossier de ce script (ihm/) et racine du workspace (projet_igus/)
IHM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "$IHM_DIR")"

# Environnement ROS2 (CycloneDDS, ROS_DOMAIN_ID, workspace compilé)
# shellcheck disable=SC1091
source "$WS_DIR/activate.bash"

# Lancement de l'IHM (remplace le shell courant)
exec python3 "$IHM_DIR/ihm_robot.py"
