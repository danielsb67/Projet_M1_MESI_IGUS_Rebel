#!/usr/bin/env bash
# Désinstalleur kiosque de l'IHM Igus Rebel.
# Désactive, arrête et supprime le service systemd utilisateur.

set -uo pipefail

SERVICE_FILE="$HOME/.config/systemd/user/ihm-robot.service"

echo "=== Désinstallation kiosque IHM Igus Rebel ==="

# Désactive et arrête le service (ne bloque pas si déjà absent)
systemctl --user disable --now ihm-robot.service 2>/dev/null \
    && echo "[*] Service désactivé et arrêté" \
    || echo "[*] Service déjà inactif ou inexistant"

# Suppression du fichier service
if [ -f "$SERVICE_FILE" ]; then
    rm -f "$SERVICE_FILE"
    echo "[*] Fichier supprimé : $SERVICE_FILE"
else
    echo "[*] Aucun fichier service à supprimer"
fi

# Rechargement de systemd
systemctl --user daemon-reload
echo "[*] systemd rechargé"

echo
echo "=== Désinstallation terminée ==="
echo "Le linger utilisateur n'a pas été désactivé."
echo "Pour le retirer : sudo loginctl disable-linger $USER"
