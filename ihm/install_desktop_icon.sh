#!/usr/bin/env bash
# Installe l'IHM VLA (pipeline SmolVLA, IGUS Rebel) comme une vraie application :
#   • une icône cliquable sur le Bureau (double-clic, aucun terminal) ;
#   • une entrée dans le menu des applications (recherche « IHM VLA »).
# À exécuter sur la machine cible :  bash ihm/install_desktop_icon.sh
#
# NB : le lanceur ne compile rien. L'IHM est une appli Tkinter pure (ihm_vla.py)
# qui lance chaque action ROS dans un sous-processus avec le bon environnement.

set -e

# --- Détection des chemins (aucun chemin codé en dur) -----------------------
IHM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$IHM_DIR/IHM-VLA.desktop"
LAUNCHER="$IHM_DIR/start_ihm_vla.sh"
APP_BASENAME="IHM-VLA.desktop"

for f in "$TEMPLATE" "$LAUNCHER"; do
    if [ ! -f "$f" ]; then
        echo "[ERREUR] Fichier introuvable : $f" >&2
        exit 1
    fi
done

# --- Le lanceur doit être exécutable ---------------------------------------
chmod +x "$LAUNCHER"

# --- Dossier Bureau (gère les locales : Bureau, Desktop, ...) ---------------
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -n "$DESKTOP_DIR" ] && [ -d "$DESKTOP_DIR" ] || DESKTOP_DIR="$HOME/Desktop"
[ -d "$DESKTOP_DIR" ] || DESKTOP_DIR="$HOME/Bureau"
[ -d "$DESKTOP_DIR" ] || DESKTOP_DIR="$HOME"

# Génère un fichier .desktop avec le bon chemin absolu.
# Permissions 744 (rwxr--r--) : exécutable par le propriétaire, NON modifiable
# par le groupe/les autres — exigé par XFCE (xfdesktop), sinon le lanceur est
# refusé (« Permissions du fichier XDG Desktop incorrectes »).
generer() {
    sed "s|__IHM_DIR__|$IHM_DIR|g" "$TEMPLATE" > "$1"
    chmod 744 "$1"
}

# --- 1. Icône sur le Bureau -------------------------------------------------
DESK_FILE="$DESKTOP_DIR/$APP_BASENAME"
generer "$DESK_FILE"

# Marquer le lanceur comme « de confiance » pour un lancement d'un double-clic
# sans avertissement :
#   - GNOME / Nautilus : metadata::trusted
#   - XFCE / xfdesktop : metadata::xfce-exe-checksum = SHA-256 du fichier
# On privilégie /usr/bin/gio (le gio d'un snap n'a pas de backend metadata).
GIO_BIN="$(command -v /usr/bin/gio || command -v gio || true)"
if [ -n "$GIO_BIN" ]; then
    SUM="$(sha256sum "$DESK_FILE" | awk '{print $1}')"
    "$GIO_BIN" set "$DESK_FILE" metadata::trusted true 2>/dev/null || true
    "$GIO_BIN" set "$DESK_FILE" metadata::xfce-exe-checksum "$SUM" 2>/dev/null || true
fi
echo "[*] Icône créée sur le Bureau : $DESK_FILE"

# --- 2. Entrée dans le menu des applications -------------------------------
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_DIR"
generer "$APP_DIR/$APP_BASENAME"
update-desktop-database "$APP_DIR" 2>/dev/null || true
echo "[*] Entrée ajoutée au menu des applications (cherche « IHM VLA »)"

echo
echo "=== Terminé ==="
echo "Rechargez le Bureau (clic droit > Recharger), puis double-cliquez sur"
echo "l'icône « IHM VLA — SmolVLA »."
echo "Si un avertissement apparaît encore au 1er lancement :"
echo "  - GNOME : clic droit sur l'icône > « Autoriser le lancement » ;"
echo "  - XFCE  : choisir « Exécuter » / « Lancer »."
