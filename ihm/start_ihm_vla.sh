#!/usr/bin/env bash
# Lanceur de l'IHM VLA (pipeline SmolVLA, IGUS Rebel).
#
# IMPORTANT : on NE source PAS ROS ici. L'IHM est une simple appli Tkinter qui
# n'utilise pas rclpy ; elle lance chaque action dans un sous-processus `bash -lc`
# qui source lui-même l'environnement adapté (Simu : ROS+install / Réel :
# activate.bash). Sourcer ROS dans CE process pollue LD_LIBRARY_PATH/PYTHONPATH
# et peut faire SEGFAULTER Tk → on garde donc un environnement PROPRE.
set -e

IHM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$IHM_DIR/ihm_vla.py"
