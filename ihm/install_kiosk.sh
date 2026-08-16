#!/usr/bin/env bash
# Installeur kiosque PORTABLE de l'IHM Igus Rebel.
# À exécuter sur la machine cible (PC de test ou Raspberry Pi 4) :
#   ./install_kiosk.sh
# Génère et active un service systemd UTILISATEUR qui démarre l'IHM au boot.

set -euo pipefail

# --- Détection des chemins (aucun chemin codé en dur) -----------------------
IHM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$IHM_DIR/ihm-robot.service"
START_SCRIPT="$IHM_DIR/start_ihm.sh"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/ihm-robot.service"

echo "=== Installation kiosque IHM Igus Rebel ==="
echo "[*] Dossier IHM     : $IHM_DIR"
echo "[*] Service cible   : $SERVICE_FILE"

# --- Vérifications ----------------------------------------------------------
if [ ! -f "$TEMPLATE" ]; then
    echo "[ERREUR] Modèle introuvable : $TEMPLATE" >&2
    exit 1
fi
if [ ! -f "$START_SCRIPT" ]; then
    echo "[ERREUR] Lanceur introuvable : $START_SCRIPT" >&2
    exit 1
fi

# --- Rendre le lanceur exécutable ------------------------------------------
chmod +x "$START_SCRIPT"
echo "[*] start_ihm.sh rendu exécutable"

# --- Génération du fichier service final ----------------------------------
mkdir -p "$SERVICE_DIR"
sed "s|__IHM_DIR__|$IHM_DIR|g" "$TEMPLATE" > "$SERVICE_FILE"
echo "[*] Service généré  : $SERVICE_FILE"

# --- Activation du service systemd utilisateur -----------------------------
systemctl --user daemon-reload
systemctl --user enable ihm-robot.service
echo "[*] Service activé (systemctl --user enable)"

# --- Linger : permet le démarrage du service avant toute connexion ---------
if sudo loginctl enable-linger "$USER"; then
    echo "[*] Linger activé pour l'utilisateur '$USER'"
else
    echo "[ATTENTION] Impossible d'activer le linger (sudo refusé/indisponible)."
    echo "            Le service ne démarrera qu'après une ouverture de session."
    echo "            Exécutez manuellement : sudo loginctl enable-linger $USER"
fi

# --- Étapes suivantes -------------------------------------------------------
echo
echo "=== Installation terminée ==="
echo "Étapes suivantes :"
echo "  1. Redémarrer la machine pour tester le démarrage automatique :"
echo "       sudo reboot"
echo "  2. Vérifier l'état du service :"
echo "       systemctl --user status ihm-robot.service"
echo "  3. Suivre les logs en direct :"
echo "       journalctl --user -u ihm-robot -f"
echo
echo "Pour désinstaller : ./uninstall_kiosk.sh"
