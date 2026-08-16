#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IHM VLA v2 — Console de pilotage SmolVLA (IGUS ReBeL), inspirée d'iRC V14
=========================================================================

Refonte complète (2026-07-02) calquée sur l'ergonomie d'igus Robot Control :

  +----------------------------------------------------------------------+
  | Fichier  Édition  Affichage  Outils  Aide                    (menus) |
  +----------------------------------------------------------------------+
  | [SIMULATION|RÉEL] | 🧪 Test  🎥 Enregistrer …      ⏹ ARRÊT  ⛶ ⊟ ✕ |
  +-------------+---------------------------------+----------------------+
  | Cartes      |  Visualisation caméras          |  Paramètres          |
  | d'état      |  (onglets Front / Wrist,        |  (panneau défilable) |
  | Système     |   PNG streamés par              |  Modèle VLA          |
  | Robot       |   ihm/camera_stream.py)         |  Épisodes / Pick /   |
  | Dataset     |                                 |  Place               |
  | Modèle      |                                 |                      |
  +-------------+---------------------------------+----------------------+
  | Moniteur de run : état · progression · ETA                           |
  | Journal filtrable colorisé (erreurs / succès / [IHM])                |
  +----------------------------------------------------------------------+
  | ● Prêt · message · dataset · modèle · mode · horloge   (barre d'état)|
  +----------------------------------------------------------------------+

Contraintes conservées de la v1 :
  - l'IHM tourne SANS environnement ROS (leçon segfault Tk+ROS, voir
    VLA_PROGRESS.md) ; la visualisation caméras passe par le sous-processus
    ihm/camera_stream.py (env ROS sourcé) qui écrit des PNG ATOMIQUES dans
    .ihm_cache/ ; l'IHM les lit par simple polling mtime. (L'intégration de
    la fenêtre Gazebo est impossible sous Wayland → flux caméras à la place.)
  - construction PHASÉE avec update_idletasks() entre étapes (anti-segfault
    Tk) ; plein écran DIFFÉRÉ ; IHM_VLA_WINDOWED=1 → fenêtré ; F11 bascule,
    Échap quitte ;
  - TOUTE la logique métier v1 reprise à l'identique : commandes ros2 launch,
    préfixe kill_all.sh (hygiène DDS), moniteur CSV rapport_run_courant.csv,
    couplage curseurs X/Y (anneau 0.15–0.54 m + exclusion bac), mode nuit,
    sélecteur de modèles v1_smolvla / v2_smolvla, ProcessRunner.

L'ancienne interface est conservée dans ihm/ihm_vla_legacy.py (rollback).
"""
from __future__ import annotations

import faulthandler
import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

faulthandler.enable()

# ============================================================================
#  Chemins (portables — dérivés de l'emplacement du script)
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE  = SCRIPT_DIR.parent
PKG_DIR    = WORKSPACE / "src" / "igus_vla"
VENV_PY    = PKG_DIR / ".venv" / "bin" / "python"
ACTIVATE   = WORKSPACE / "activate.bash"
RAW_ROOT   = WORKSPACE / "datasets" / "raw"
# Cache caméras en RAM (/dev/shm) : zéro usure SSD + lecture instantanée.
# PPM plutôt que PNG : décodage Tk quasi gratuit → flux fluide à 15 Hz.
CAM_CACHE  = (Path("/dev/shm/ihm_cam_cache") if Path("/dev/shm").is_dir()
              else WORKSPACE / ".ihm_cache")
CAM_EXT    = "ppm"

# Pose HOME du pipeline (= HOME_JOINTS de vla_policy_node / pick_place_ia).
# Sert de pose par défaut à la vue Robot quand ROS ne tourne pas, et c'est
# aussi la pose de SPAWN sim (igus_rebel.control.xacro, initial_position).
HOME_JOINTS = (0.0, -0.3491, 1.9199, 0.0, 1.5359, 0.0)

ROS_SETUP     = "source /opt/ros/humble/setup.bash"
INSTALL_SETUP = f"source {shlex.quote(str(WORKSPACE / 'install' / 'setup.bash'))}"
# ROS_LOCALHOST_ONLY=1 : toute la sim tourne sur CETTE machine → on garde le trafic
# ROS/DDS sur l'interface loopback (lo, toujours UP). Sinon FastDDS (RMW défaut) fait
# sa découverte multicast sur le Wi-Fi/ethernet ; couper le Wi-Fi en plein run casse
# la liaison DDS (expert↔move_group) → tous les cycles échouent. Vu le 2026-06-26.
LOCALHOST_ONLY = "export ROS_LOCALHOST_ONLY=1"
SIM_ENV       = f"{LOCALHOST_ONLY} && {ROS_SETUP} && {INSTALL_SETUP}"
REAL_ENV      = (f"source {shlex.quote(str(ACTIVATE))}" if ACTIVATE.exists() else SIM_ENV)

# Topics image publiés par la sim (voir camera_stream.py — mêmes défauts).
CAM_TOPICS = {"front": "/front_camera/image", "wrist": "/wrist_camera/image"}


def _vla_python(module_args: str) -> str:
    py = str(VENV_PY) if VENV_PY.exists() else "python3"
    return (f'cd {shlex.quote(str(WORKSPACE))} && '
            f'PYTHONPATH={shlex.quote(str(PKG_DIR))}:$PYTHONPATH '
            f'{shlex.quote(py)} -m {module_args}')


# ============================================================================
#  Modèles VLA disponibles (checkpoints locaux)
# ============================================================================
# Convention : un modèle = outputs/train/<nom>/checkpoints/last/pretrained_model
# (noms clairs : v1_smolvla = 1 caméra front ; v2_smolvla = 2 caméras front+wrist).
MODELS_DIR = WORKSPACE / "outputs" / "train"


def _detect_models() -> dict:
    """Modèles déployables détectés : {nom: chemin relatif du pretrained_model}."""
    models = {}
    if MODELS_DIR.exists():
        for cfg in sorted(MODELS_DIR.glob("*/checkpoints/last/pretrained_model/config.json")):
            pm = cfg.parent
            name = pm.relative_to(MODELS_DIR).parts[0]
            models[name] = str(pm.relative_to(WORKSPACE))
    return models


DEFAULT_CKPT = _detect_models().get(
    "v2_smolvla", "outputs/train/v2_smolvla/checkpoints/last/pretrained_model")


# ============================================================================
#  Palette — sombre industrielle, accent orange igus
# ============================================================================
BG        = "#14181d"    # fond général
BG_PANEL  = "#1b222a"    # barres (outils, moniteur, état)
BG_CARD   = "#212a34"    # cartes d'état + panneau paramètres
BG_PARAM  = BG_CARD      # alias : les helpers v1 (sections/labels) l'utilisent
FG        = "#e8edf2"
FG_DIM    = "#8b98a5"
ACCENT    = "#ff9e1b"    # orange igus — actions principales / titres
ACCENT2   = "#34c07c"    # vert — tests / validation
ACCENT_BLUE = "#2f81f7"  # bleu — données / modèle (convertir, entraîner, déployer)
ACCENT_REAL = "#e0562a"  # orange brûlé — mode robot réel
OK_GREEN  = "#34c07c"
STOP_RED  = "#e05252"
WARN_YEL  = "#d4a017"
LOG_BG    = "#0c1015"
ENTRY_BG  = "#0d1117"
ENTRY_BG_BAD = "#3a1518"   # rouge sombre : champ hors portée
TXT_ON_ACCENT = "#15181c"  # texte sombre sur fonds orange/jaune (contraste)

# ── Limites de portée du bras Igus Rebel (m) ────────────────────────────────
# Contrainte empirique v3 (validée sur 450 épisodes) : √(x²+y²) ∈ [MIN, MAX].
# Zone accessible = anneau 0.15–0.54 m sans singularités/hors-portée.
# L'expert refuse aussi √(x²+y²) > 0.62 m ; on est bien en dessous.
# Voir record_orchestrator.py (_sample_object_xy v3) et documentation .md.
# Curseurs X/Y couplés pour respecter les limites en temps réel (IHM).
MAX_REACH_XY = 0.54   # rayon max empirique (tous les succès observés ≤ 0.549 m)
MIN_REACH_XY = 0.15   # rayon min empirique (éloigne des singularités d'origine)

# ZONE D'EXCLUSION BAC : le bac (0.184×0.184 m) est centré sur (place_x, place_y).
# On interdit tout pick dans ce rectangle + marge pince, sinon la roulette se pose
# contre/dans une paroi → grasp impossible. Voir record_orchestrator._sample_object_xy.
BIN_CENTER_X = 0.0    # = place_x (centre du bac, world_with_camera.sdf <model bac>)
BIN_CENTER_Y = 0.25   # = place_y
BIN_KEEPOUT  = 0.222  # demi-côté exclusion = bin_half (0.092) + marge pince (0.13)


# ============================================================================
#  ProcessRunner — sous-processus streamé (identique v1)
# ============================================================================
class ProcessRunner:
    def __init__(self, out_queue: "queue.Queue[str]") -> None:
        self._q = out_queue
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, command: str, label: str) -> bool:
        if self.running:
            self._q.put("[IHM] Un processus tourne déjà — arrêtez-le d'abord.\n")
            return False
        self._q.put(f"\n{'─' * 60}\n[IHM] ▶ {label}\n[IHM] $ {command}\n{'─' * 60}\n")
        try:
            self._proc = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
                cwd=str(WORKSPACE),
            )
        except Exception as exc:  # noqa: BLE001
            self._q.put(f"[IHM] ✗ Échec du lancement : {exc}\n")
            self._proc = None
            return False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return True

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._q.put(line)
        code = self._proc.wait()
        self._q.put(f"[IHM] ⏹ Terminé (code {code}).\n")

    def stop(self) -> None:
        if not self.running or self._proc is None:
            return
        self._q.put("[IHM] ⏹ Arrêt demandé…\n")
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGINT)
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ============================================================================
#  CameraFeed — sous-processus camera_stream.py (ROS → PNG dans .ihm_cache/)
# ============================================================================
class CameraFeed:
    """Gère le streamer caméras : un SEUL sous-processus ROS, silencieux,
    indépendant du ProcessRunner (peut tourner PENDANT un run). L'IHM lit
    les PNG par polling mtime (écriture atomique côté streamer)."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        if self.running:
            return True
        CAM_CACHE.mkdir(parents=True, exist_ok=True)
        stream = SCRIPT_DIR / "camera_stream.py"
        topics = " ".join(f"--topic {n}={t}" for n, t in CAM_TOPICS.items())
        cmd = (f"{SIM_ENV} && exec python3 {shlex.quote(str(stream))} "
               f"--out {shlex.quote(str(CAM_CACHE))} {topics} "
               f"--hz 15 --ext {CAM_EXT}")
        try:
            self._proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, cwd=str(WORKSPACE))
        except Exception:  # noqa: BLE001
            self._proc = None
            return False
        return True

    def stop(self) -> None:
        if not self.running or self._proc is None:
            self._proc = None
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGINT)
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._proc = None


# ============================================================================
#  Utilitaires
# ============================================================================
def _dataset_stats(raw_root: Path) -> dict:
    """Lit le dossier dataset/raw et renvoie {total, success, fail, last_pick}."""
    stats = {"total": 0, "success": 0, "fail": 0, "last_pick": None}
    if not raw_root.exists():
        return stats
    for ep_dir in sorted(raw_root.glob("episode_*")):
        meta_file = ep_dir / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:  # noqa: BLE001
            continue
        stats["total"] += 1
        if meta.get("success"):
            stats["success"] += 1
        else:
            stats["fail"] += 1
        # Dernière position pick connue
        if "pick_x" in meta and "pick_y" in meta:
            px, py = meta["pick_x"], meta["pick_y"]
            if not (isinstance(px, float) and px != px):  # exclude NaN
                stats["last_pick"] = (px, py)
    return stats


def _list_datasets() -> list[str]:
    """Liste les dossiers dataset sous datasets/ (raw*, datasets contenant episode_*).
    Pour le sélecteur de dataset de l'IHM."""
    base = WORKSPACE / "datasets"
    found: list[str] = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d.name.startswith("raw") or any(d.glob("episode_*"))):
                found.append(f"datasets/{d.name}")
    # Garantit la présence des dossiers usuels même vides
    for default in ("datasets/raw_v2", "datasets/raw_echecs"):
        if default not in found:
            found.append(default)
    return found


def _lighten(hex_color: str, f: float) -> str:
    """Éclaircit une couleur hex de la fraction f (0..1) vers le blanc.
    Sert aux effets de survol des boutons et aux fins séparateurs."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * f)
    g = int(g + (255 - g) * f)
    b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def _entry(parent, var: tk.StringVar, w: int = 7) -> tk.Entry:
    e = tk.Entry(parent, textvariable=var, width=w,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG,
                 font=("DejaVu Sans Mono", 11), relief="flat",
                 highlightbackground=FG_DIM, highlightthickness=1)
    return e


def _label(parent, text: str, dim: bool = False, bold: bool = False) -> tk.Label:
    font = ("DejaVu Sans", 11, "bold") if bold else ("DejaVu Sans", 11)
    return tk.Label(parent, text=text, bg=BG_PARAM,
                    fg=FG_DIM if dim else FG, font=font)


# ============================================================================
#  ScrollFrame — cadre à défilement vertical (panneau Paramètres)
# ============================================================================
class ScrollFrame(tk.Frame):
    """Canvas + frame interne + molette (active au survol seulement)."""

    def __init__(self, parent: tk.Widget, bg: str) -> None:
        super().__init__(parent, bg=bg)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfigure(self._win, width=e.width))
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._canvas.bind("<Enter>", lambda _e: self._bind_wheel())
        self._canvas.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _bind_wheel(self) -> None:
        self._canvas.bind_all("<Button-4>",
                              lambda _e: self._canvas.yview_scroll(-2, "units"))
        self._canvas.bind_all("<Button-5>",
                              lambda _e: self._canvas.yview_scroll(2, "units"))
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))

    def _unbind_wheel(self) -> None:
        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self._canvas.unbind_all(seq)


# ============================================================================
#  RButton — bouton à coins arrondis (Tk n'en propose pas nativement)
# ============================================================================
class RButton(tk.Canvas):
    """Canvas dessinant un rectangle arrondi (polygone lissé) + texte centré.
    Survol éclairci, curseur main — même usage qu'un tk.Button."""

    def __init__(self, parent: tk.Widget, text: str, command,
                 bg: str, fg: str = "white",
                 font=("DejaVu Sans", 10, "bold"),
                 padx: int = 14, pady: int = 7, radius: int = 11) -> None:
        # Mesure du texte via un Label sonde (winfo_req*) : tkfont.Font.metrics()
        # SEGFAULTE dans le contexte de l'app sur cette machine (fragilité Tk déjà
        # vue → construction phasée) ; le chemin widget, lui, est éprouvé partout.
        probe = tk.Label(parent, text=text, font=font)
        w = probe.winfo_reqwidth() + 2 * padx
        h = probe.winfo_reqheight() + 2 * pady
        probe.destroy()
        try:
            pbg = parent.cget("bg")
        except tk.TclError:
            pbg = BG
        super().__init__(parent, width=w, height=h, bg=pbg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._base, self._hover = bg, _lighten(bg, 0.15)
        r = min(radius, h // 2)
        pts = [w - 1, 1 + r, w - 1, h - 1 - r, w - 1 - r, h - 1, 1 + r, h - 1,
               1, h - 1 - r, 1, 1 + r, 1 + r, 1, w - 1 - r, 1]
        self._shape = self.create_polygon(pts, smooth=True, fill=bg, outline="")
        self._txt = self.create_text(w // 2, h // 2, text=text, fill=fg, font=font)
        self.bind("<Enter>",
                  lambda _e: self.itemconfigure(self._shape, fill=self._hover),
                  add="+")
        self.bind("<Leave>",
                  lambda _e: self.itemconfigure(self._shape, fill=self._base),
                  add="+")
        if command is not None:
            self.bind("<Button-1>", lambda _e: command())

    def set_text(self, text: str) -> None:
        """Change le libellé (ex. ▶ ↔ ⏹) sans redimensionner le bouton."""
        self.itemconfigure(self._txt, text=text)


# ============================================================================
#  Application principale
# ============================================================================
class IhmVla(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IHM VLA — IGUS ReBeL · SmolVLA (console de pilotage)")
        self.configure(bg=BG)
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self._apply_ttk_theme()
        self._fullscreen = False
        self._fs_var = tk.BooleanVar(value=False)   # suit l'état plein écran (menu)
        self.bind("<Escape>", lambda _e: self._quit())
        self.bind("<F11>", lambda _e: self._toggle_fullscreen())
        self.bind("<Control-l>", lambda _e: self._clear_log())
        self.protocol("WM_DELETE_WINDOW", self._quit)

        # ── Variables de paramètres ──────────────────────────────
        self._mode        = tk.StringVar(value="simu")
        self._ckpt        = tk.StringVar(value=DEFAULT_CKPT)
        self._raw_root    = tk.StringVar(value="datasets/raw_v2")
        self._q: "queue.Queue[str]" = queue.Queue()
        self._runner = ProcessRunner(self._q)

        # --- Paramètres de collecte ---
        self._n_episodes  = tk.StringVar(value="50")
        self._randomize   = tk.BooleanVar(value=True)
        # Rayon de blending Pilz en CM (0 = arrêts classiques à chaque étape).
        # Converti en mètres au lancement (blend_radius:=cm/100) ; l'expert
        # borne lui-même à offset−1 cm (9 cm max utile). Prépa collecte v3.
        self._blend_cm    = tk.StringVar(value="0")
        # Mode nuit : Gazebo headless + IHM auto-masquée pendant le run (libère le CPU
        # pour la sim/caméra → moins de risque de caméra figée sur les longs runs).
        self._night_mode  = tk.BooleanVar(value=False)
        self._night_active = False   # vrai pendant un run lancé en mode nuit

        # Paramètres avancés masqués par défaut (chemin checkpoint, coords bac) :
        # visibles via Affichage → « Paramètres avancés ». Journal compact par
        # défaut (préfixes ROS réécrits) ; brut via « Journal verbeux ».
        self._advanced    = tk.BooleanVar(value=False)
        self._log_verbose = tk.BooleanVar(value=False)

        # Vue Robot (onglet central) : pose HOME par défaut, live si joints.json
        self._robot_joints: list[float] = list(HOME_JOINTS)
        self._robot_live = False
        self._joints_mtime = 0.0

        # ── État UI ──────────────────────────────────────────────
        # Journal : buffer complet + filtre d'affichage (tout/erreurs/succès)
        self._log_buffer: list[str] = []
        self._log_filter = tk.StringVar(value="tout")
        # Moniteur de run live : cible N capturée au lancement + dossier suivi
        self._run_target = 0
        self._run_active_raw = ""
        # Progression GÉNÉRIQUE de tâche (conversion, entraînement, …) :
        # chrono depuis le début + ETA + barre, par parsing du stdout.
        self._task_kind = ""          # "record" | "convert" | "train" | "other"
        self._task_label = ""
        self._task_start = 0.0        # horodatage de départ (chrono)
        self._task_elapsed = 0.0      # gelé à la fin du process
        self._task_k = 0              # étape courante (parsée)
        self._task_total = 0          # total (parsé)
        self._task_pct = 0.0          # pourcentage (parsé, repli)

        # Flux caméras (visualisation centrale)
        self._cam_feed = CameraFeed()
        self._cam_lbls: dict[str, tk.Label] = {}
        self._cam_info: dict[str, tk.Label] = {}
        self._cam_imgs: dict[str, tk.PhotoImage] = {}
        self._cam_mtime: dict[str, float] = {}

        # Coordonnées pick par défaut (zone accessible empirique : anneau 0.15–0.54 m)
        self._pick_x      = tk.StringVar(value="0.35")
        self._pick_y      = tk.StringVar(value="0.15")
        self._pick_z      = tk.StringVar(value="0.018")
        # Garde anti-réentrance : les curseurs X/Y se mettent à jour mutuellement
        # (couplage √(x²+y²) ≤ cap) ; ce drapeau évite les boucles de traces.
        self._coupling    = False
        # Validation live de la portée + couplage X/Y : à chaque frappe, l'axe
        # édité (maître) borne l'autre, puis on recolore les champs.
        self._pick_x.trace_add("write", lambda *_: self._on_pick_xy_edited("x"))
        self._pick_y.trace_add("write", lambda *_: self._on_pick_xy_edited("y"))

        # Coordonnées place (bac fixe)
        self._place_x     = tk.StringVar(value="0.00")
        self._place_y     = tk.StringVar(value="0.25")
        self._place_z     = tk.StringVar(value="0.01")

        # Cartes d'état + barre d'état suivent dataset/modèle en live.
        self._ckpt.trace_add("write", lambda *_: self._on_ckpt_changed())
        self._raw_root.trace_add("write", lambda *_: self._on_raw_root_changed())

        # NB : le rayon de saisie (grasp_radius) est FIGÉ dans le backend
        # (gripper_shim.py) → plus exposé dans l'IHM (évite qu'un objet
        # « se colle » à la pince de trop loin = mauvaise donnée VLA).

        # ── Construction PHASÉE de l'UI (anti-segfault Tk) ───────
        self._build_menubar();      self.update_idletasks()
        self._build_toolbar();      self.update_idletasks()
        self._build_statusbar();    self.update_idletasks()
        self._build_main_layout();  self.update_idletasks()
        self._render_mode();        self.update_idletasks()
        self._on_ckpt_changed()
        self._on_raw_root_changed()

        self.after(100, self._drain_log)
        self.after(2000, self._refresh_stats)        # stats dataset
        self.after(1500, self._refresh_run_monitor)  # moniteur de run live
        self.after(400, self._poll_cameras)          # flux caméras (PNG)
        self.after(1000, self._tick_clock)           # horloge barre d'état

        if not os.environ.get("IHM_VLA_WINDOWED"):
            self.after(200, self._enter_fullscreen)

    # ------------------------------------------------------------------
    #  Thème ttk sombre
    # ------------------------------------------------------------------
    def _apply_ttk_theme(self) -> None:
        """Habille les widgets ttk (Progressbar, Combobox, Scrollbar, Notebook)
        aux couleurs de la palette. Sans ça ils gardent un look clair par défaut
        qui jure avec le reste de l'IHM sombre."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")   # thème le plus malléable pour la coloration
        except tk.TclError:
            pass

        # Barre de progression (moniteur de run)
        style.configure("Horizontal.TProgressbar",
                        troughcolor=ENTRY_BG, bordercolor=BG_PANEL,
                        background=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT)

        # Combobox — champ fermé
        style.configure("TCombobox",
                        fieldbackground=ENTRY_BG, background=BG_PANEL,
                        foreground=FG, arrowcolor=FG, bordercolor=FG_DIM,
                        lightcolor=BG_PANEL, darkcolor=BG_PANEL,
                        selectbackground=ENTRY_BG, selectforeground=FG)
        style.map("TCombobox",
                  fieldbackground=[("readonly", ENTRY_BG)],
                  foreground=[("readonly", FG)],
                  arrowcolor=[("active", ACCENT)])
        # Liste déroulante du Combobox : Listbox Tk classique (pas ttk),
        # stylée via la base d'options puisqu'elle est créée à l'ouverture.
        self.option_add("*TCombobox*Listbox.background", ENTRY_BG)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", TXT_ON_ACCENT)

        # Barres de défilement
        style.configure("Vertical.TScrollbar",
                        troughcolor=BG, bordercolor=BG, arrowcolor=FG_DIM,
                        background=BG_PANEL, lightcolor=BG_PANEL, darkcolor=BG_PANEL)
        style.map("Vertical.TScrollbar",
                  background=[("active", _lighten(BG_PANEL, 0.25))])

        # Onglets caméras (zone Visualisation)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_DIM,
                        padding=(16, 7), font=("DejaVu Sans", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", ACCENT)])

    def _add_hover(self, btn: tk.Button, base: str, hover: str) -> None:
        """Éclaircit le fond du bouton au survol (restauré au départ du curseur).
        `add="+"` pour ne pas écraser un éventuel bind <Enter>/<Leave> de tooltip."""
        btn.bind("<Enter>", lambda _e: btn.configure(bg=hover), add="+")
        btn.bind("<Leave>", lambda _e: btn.configure(bg=base), add="+")

    # ------------------------------------------------------------------
    #  Barre de menus (style iRC : Fichier / Édition / Affichage / Outils / Aide)
    # ------------------------------------------------------------------
    def _menu(self, parent: tk.Menu) -> tk.Menu:
        return tk.Menu(parent, tearoff=0, bg=BG_PANEL, fg=FG,
                       activebackground=ACCENT, activeforeground=TXT_ON_ACCENT,
                       bd=0, font=("DejaVu Sans", 10))

    def _build_menubar(self) -> None:
        bar = tk.Menu(self, bg=BG_PANEL, fg=FG,
                      activebackground=ACCENT, activeforeground=TXT_ON_ACCENT,
                      bd=0, font=("DejaVu Sans", 10))

        m_file = self._menu(bar)
        m_file.add_command(label="📁 Choisir un dataset…", command=self._browse_dataset)
        m_file.add_command(label="🗂 Ouvrir le dossier datasets/",
                           command=lambda: self._open_folder(WORKSPACE / "datasets"))
        m_file.add_command(label="🗂 Ouvrir le dataset courant",
                           command=lambda: self._open_folder(
                               WORKSPACE / self._raw_root.get()))
        m_file.add_command(label="🗂 Ouvrir raw_echecs", command=self._open_raw_echecs)
        m_file.add_separator()
        m_file.add_command(label="⊟ Réduire la fenêtre", command=self.iconify)
        m_file.add_command(label="✕ Quitter            (Échap)", command=self._quit)
        bar.add_cascade(label="Fichier", menu=m_file)

        m_edit = self._menu(bar)
        m_edit.add_command(label="📋 Copier le journal", command=self._copy_log)
        m_edit.add_command(label="🧹 Effacer le journal   (Ctrl+L)",
                           command=self._clear_log)
        bar.add_cascade(label="Édition", menu=m_edit)

        m_view = self._menu(bar)
        m_view.add_checkbutton(label="⛶ Plein écran   (F11)",
                               variable=self._fs_var,
                               command=lambda: self._set_fullscreen(self._fs_var.get()))
        m_view.add_separator()
        m_view.add_checkbutton(label="⚙ Paramètres avancés (checkpoint, bac…)",
                               variable=self._advanced,
                               command=self._render_mode)
        m_view.add_checkbutton(label="Journal verbeux (sortie ROS brute)",
                               variable=self._log_verbose)
        m_view.add_separator()
        for val, lbl in [("tout", "Journal : tout"),
                         ("erreurs", "Journal : ⚠ erreurs"),
                         ("succes", "Journal : ✓ succès")]:
            m_view.add_radiobutton(label=lbl, variable=self._log_filter, value=val,
                                   command=self._apply_log_filter)
        bar.add_cascade(label="Affichage", menu=m_view)

        m_tools = self._menu(bar)
        m_tools.add_command(
            label="🧹 Purger DDS / Gazebo (kill_all.sh)",
            command=lambda: self._run_cmd(
                f"bash {shlex.quote(str(WORKSPACE / 'kill_all.sh'))}",
                "Purge DDS/Gazebo"))
        m_tools.add_command(
            label="🧪 Smoke test (5 steps CPU)",
            command=lambda: self._run_cmd(
                f"cd {shlex.quote(str(WORKSPACE))} && "
                f"bash src/igus_vla/scripts/smoke_test.sh",
                "Smoke test"))
        m_tools.add_separator()
        m_tools.add_command(label="🎲 Test randomisation (aperçu)",
                            command=self._run_preview_randomization)
        m_tools.add_command(label="🔍 Balayage visibilité (auto)",
                            command=self._run_visibility_sweep)
        m_tools.add_separator()
        m_tools.add_command(label="📊 Stats dataset", command=self._show_stats_popup)
        bar.add_cascade(label="Outils", menu=m_tools)

        m_help = self._menu(bar)
        m_help.add_command(label="📖 Ouvrir MANUEL.md",
                           command=lambda: self._open_folder(WORKSPACE / "MANUEL.md"))
        m_help.add_command(label="⌨ Raccourcis clavier", command=self._show_shortcuts)
        m_help.add_separator()
        m_help.add_command(label="ℹ À propos", command=self._show_about)
        bar.add_cascade(label="Aide", menu=m_help)

        self.configure(menu=bar)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "À propos",
            "IHM VLA v2.0 — Console de pilotage SmolVLA\n"
            "Robot : IGUS ReBeL 6 DOF (sim Gazebo / réel CRI)\n\n"
            "Interface inspirée d'igus Robot Control (iRC V14).\n"
            "Ancienne interface : ihm/ihm_vla_legacy.py\n"
            "Flux caméras : ihm/camera_stream.py (ROS → PNG)\n\n"
            "Projet : ~/projet_igus — voir MANUEL.md / HANDOFF.md")

    def _show_shortcuts(self) -> None:
        messagebox.showinfo(
            "Raccourcis clavier",
            "F11\tBasculer plein écran\n"
            "Échap\tQuitter l'IHM\n"
            "Ctrl+L\tEffacer le journal\n"
            "Molette\tDéfiler le panneau Paramètres")

    # ------------------------------------------------------------------
    #  Barre d'outils : mode (segmenté) + actions contextuelles + ARRÊT
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = tk.Frame(self, bg=BG_PANEL)
        tb.pack(fill="x", side="top")
        tk.Frame(self, bg=_lighten(BG_PANEL, 0.10), height=1).pack(fill="x")

        # -- Sélecteur de mode (segmenté, à gauche) --
        modebox = tk.Frame(tb, bg=BG_PANEL)
        modebox.pack(side="left", padx=(10, 6), pady=6)
        for m, lbl, col in (("simu", "🌍 SIMULATION", ACCENT),
                            ("real", "🔧 RÉEL", ACCENT_REAL)):
            tk.Radiobutton(
                modebox, text=lbl, variable=self._mode, value=m,
                command=self._render_mode, indicatoron=False,
                bg=BG_CARD, fg=FG, selectcolor=col,
                activebackground=_lighten(BG_CARD, 0.10), activeforeground=FG,
                font=("DejaVu Sans", 10, "bold"), bd=0, padx=12, pady=6,
            ).pack(side="left", padx=1)
        tk.Frame(tb, bg=_lighten(BG_PANEL, 0.15), width=1).pack(
            side="left", fill="y", pady=8, padx=4)

        # -- Actions contextuelles (reconstruites à chaque changement de mode) --
        self._toolbar_actions = tk.Frame(tb, bg=BG_PANEL)
        self._toolbar_actions.pack(side="left", fill="x", expand=True)

        # -- Contrôles fenêtre (style barre de titre) + ARRÊT (à droite, fixes) --
        self._winbtn(tb, "✕", self._quit, danger=True).pack(
            side="right", fill="y", padx=(0, 4), pady=2)
        self._winbtn(tb, "❐", self._toggle_fullscreen).pack(
            side="right", fill="y", pady=2)
        self._winbtn(tb, "–", self.iconify).pack(side="right", fill="y", pady=2)
        stop = RButton(tb, "⏹  ARRÊT", self._stop, bg=STOP_RED,
                       font=("DejaVu Sans", 11, "bold"), padx=20, pady=8)
        stop.pack(side="right", padx=10, pady=5)
        self._tooltip(stop, "Arrête le processus en cours\n(SIGINT → SIGTERM → SIGKILL).")

    def _winbtn(self, parent: tk.Widget, glyph: str, cmd,
                danger: bool = False) -> tk.Button:
        """Bouton de contrôle fenêtre discret : – ❐ ✕ (✕ rougit au survol)."""
        hbg = STOP_RED if danger else BG_CARD
        hfg = "white" if danger else FG
        b = tk.Button(parent, text=glyph, command=cmd, bg=BG_PANEL, fg=FG_DIM,
                      bd=0, relief="flat", width=4, font=("DejaVu Sans", 11),
                      activebackground=hbg, activeforeground=hfg)
        b.bind("<Enter>", lambda _e: b.configure(bg=hbg, fg=hfg), add="+")
        b.bind("<Leave>", lambda _e: b.configure(bg=BG_PANEL, fg=FG_DIM), add="+")
        return b

    def _tbtn(self, text: str, cmd, color: str | None = None,
              tip: str = "") -> RButton:
        """Bouton d'action arrondi. Texte sombre sur fonds clairs (orange/jaune)."""
        c = color or ACCENT
        fg = TXT_ON_ACCENT if c in (ACCENT, WARN_YEL) else "white"
        b = RButton(self._toolbar_actions, text, cmd, bg=c, fg=fg,
                    font=("DejaVu Sans", 10, "bold"), padx=14, pady=8)
        b.pack(side="left", padx=3, pady=6)
        if tip:
            self._tooltip(b, tip)
        return b

    def _build_simu_actions(self) -> None:
        self._tbtn("🧪 Test 1 cycle", self._run_one_test, ACCENT2,
                   tip="Lance 1 pick&place pour valider le grasp — sans enregistrer.\n"
                       "Utilise les coordonnées pick/place du panneau Paramètres.")
        self._tbtn("🎥 Enregistrer N démos", self._run_record, ACCENT,
                   tip="Lance Gazebo + expert + recorder jusqu'à N épisodes GARDÉS.\n"
                       "Les ratés sont rejoués automatiquement (plafond anti-boucle) →\n"
                       "tu obtiens bien N démos valides. Coords/randomisation : Paramètres.")
        self._tbtn("🔄 Convertir", self._run_convert, ACCENT_BLUE,
                   tip="Convertit le dataset CHOISI (📁) en LeRobotDataset (parquet + mp4),\n"
                       "TOUTES les caméras (front + wrist). Sortie : datasets/lerobot_<nom>.")
        self._tbtn("🎓 Entraîner", lambda: self._run_cmd(
                       f"cd {shlex.quote(str(WORKSPACE))} && "
                       f"bash src/igus_vla/scripts/train_smolvla.sh",
                       "Entraîner SmolVLA", kind="train"), ACCENT_BLUE,
                   tip="Lance train_smolvla.sh (lit config/smolvla.yaml).\n"
                       "GPU recommandé (~4 h sur A100 pour 20 000 steps).")
        self._tbtn("🤖 Déployer (sim)", self._run_deploy_sim, ACCENT_BLUE,
                   tip="vla_deploy.launch.py (backend gazebo) avec le modèle choisi\n"
                       "dans le panneau Paramètres (menu « Modèle VLA »).")

    def _build_real_actions(self) -> None:
        self._tbtn("🔌 Connecter (CRI)", lambda: self._run_cmd(
                       f"{REAL_ENV} && ros2 launch igus_rebel_moveit_config demo.launch.py "
                       "hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none "
                       "camera:=none load_base:=false",
                       "Bring-up CRI"), ACCENT_REAL,
                   tip="MoveIt + ros2_control sur le robot réel.")
        self._tbtn("🦾 Pick&place expert", lambda: self._run_cmd(
                       f"{REAL_ENV} && ros2 run mon_controleur pick_place_ia --ros-args "
                       f"-p mode:=static "
                       f"-p pick_x:={self._pick_x.get()} -p pick_y:={self._pick_y.get()} "
                       f"-p pick_z:={self._pick_z.get()} "
                       f"-p place_x:={self._place_x.get()} -p place_y:={self._place_y.get()} "
                       f"-p place_z:={self._place_z.get()}",
                       "Expert réel"), ACCENT_REAL,
                   tip="Exécute un cycle pick&place sur le vrai robot avec les "
                       "coordonnées saisies.")
        self._tbtn("🤖 Déployer (réel) ⚠ stub", self._run_deploy_real, WARN_YEL,
                   tip="CRIBackend est un STUB non fonctionnel. Confirmation demandée.")

    # ------------------------------------------------------------------
    #  Barre d'état (bas de fenêtre)
    # ------------------------------------------------------------------
    def _build_statusbar(self) -> None:
        tk.Frame(self, bg=_lighten(BG_PANEL, 0.10), height=1).pack(
            fill="x", side="bottom")
        sb = tk.Frame(self, bg=BG_PANEL)
        sb.pack(fill="x", side="bottom")

        def sep() -> None:
            tk.Frame(sb, bg=_lighten(BG_PANEL, 0.15), width=1).pack(
                side="left", fill="y", pady=4, padx=8)

        self._sb_dot = tk.Label(sb, text="●", bg=BG_PANEL, fg=FG_DIM,
                                font=("DejaVu Sans", 12))
        self._sb_dot.pack(side="left", padx=(10, 2), pady=3)
        self._sb_state = tk.Label(sb, text="Prêt", bg=BG_PANEL, fg=FG,
                                  font=("DejaVu Sans", 10, "bold"))
        self._sb_state.pack(side="left")
        sep()
        self._status = tk.Label(sb, text="Prêt.", bg=BG_PANEL, fg=FG_DIM,
                                font=("DejaVu Sans", 10), anchor="w")
        self._status.pack(side="left", fill="x", expand=True)

        self._sb_clock = tk.Label(sb, text="", bg=BG_PANEL, fg=FG_DIM,
                                  font=("DejaVu Sans Mono", 10))
        self._sb_clock.pack(side="right", padx=(8, 10))
        tk.Frame(sb, bg=_lighten(BG_PANEL, 0.15), width=1).pack(
            side="right", fill="y", pady=4, padx=8)
        self._sb_mode = tk.Label(sb, text="SIMULATION", bg=BG_PANEL, fg=ACCENT,
                                 font=("DejaVu Sans", 10, "bold"))
        self._sb_mode.pack(side="right")
        tk.Frame(sb, bg=_lighten(BG_PANEL, 0.15), width=1).pack(
            side="right", fill="y", pady=4, padx=8)
        self._sb_model = tk.Label(sb, text="🧠 —", bg=BG_PANEL, fg=FG_DIM,
                                  font=("DejaVu Sans", 10))
        self._sb_model.pack(side="right")
        tk.Frame(sb, bg=_lighten(BG_PANEL, 0.15), width=1).pack(
            side="right", fill="y", pady=4, padx=8)
        self._sb_ds = tk.Label(sb, text="📁 —", bg=BG_PANEL, fg=FG_DIM,
                               font=("DejaVu Sans", 10))
        self._sb_ds.pack(side="right")

    def _tick_clock(self) -> None:
        try:
            self._sb_clock.configure(text=datetime.now().strftime("%H:%M:%S"))
        except tk.TclError:
            return
        self.after(1000, self._tick_clock)

    # ------------------------------------------------------------------
    #  Layout principal : 3 colonnes redimensionnables + zone journal
    # ------------------------------------------------------------------
    def _build_main_layout(self) -> None:
        vp = tk.PanedWindow(self, orient="vertical", bg=BG, sashwidth=5,
                            sashrelief="flat", bd=0, opaqueresize=True)
        vp.pack(fill="both", expand=True)

        hp = tk.PanedWindow(vp, orient="horizontal", bg=BG, sashwidth=5,
                            sashrelief="flat", bd=0, opaqueresize=True)

        # -- Colonne gauche : cartes d'état --
        sidebar = tk.Frame(hp, bg=BG)
        self._build_sidebar(sidebar)
        hp.add(sidebar, width=270, minsize=220, stretch="never")

        # -- Colonne centrale : visualisation caméras --
        center = tk.Frame(hp, bg=BG)
        self._build_center(center)
        hp.add(center, minsize=360, stretch="always")

        # -- Colonne droite : paramètres (défilables) --
        params_col = tk.Frame(hp, bg=BG_CARD)
        head = tk.Frame(params_col, bg=BG_CARD)
        head.pack(fill="x")
        tk.Label(head, text="Paramètres", bg=BG_CARD, fg=FG,
                 font=("DejaVu Sans", 12, "bold")).pack(
            side="left", padx=10, pady=(8, 4))
        tk.Frame(params_col, bg=_lighten(BG_CARD, 0.10), height=1).pack(fill="x")
        self._params = ScrollFrame(params_col, bg=BG_CARD)
        self._params.pack(fill="both", expand=True)
        # `self._left` = frame interne du panneau : les builders v1 (sections,
        # curseurs, sélecteur modèle) s'y accrochent sans modification.
        self._left = self._params.inner
        hp.add(params_col, width=390, minsize=330, stretch="never")

        vp.add(hp, minsize=320, stretch="always")

        # -- Zone basse : moniteur de run + journal --
        bottom = tk.Frame(vp, bg=BG)
        self._build_run_monitor(bottom)
        self._build_log(bottom)
        vp.add(bottom, height=280, minsize=170, stretch="never")

    # ── Cartes d'état (colonne gauche) ────────────────────────────────
    def _card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=BG_CARD, highlightthickness=0)
        outer.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(outer, text=title, bg=BG_CARD, fg=ACCENT,
                 font=("DejaVu Sans", 10, "bold")).pack(
            anchor="w", padx=10, pady=(8, 2))
        body = tk.Frame(outer, bg=BG_CARD)
        body.pack(fill="x", padx=10, pady=(0, 10))
        return body

    def _build_sidebar(self, parent: tk.Frame) -> None:
        # Carte Système : chemins + prérequis détectés au lancement.
        sysb = self._card(parent, "SYSTÈME")
        tk.Label(sysb,
                 text=(f"Workspace : ~/{WORKSPACE.name}\n"
                       f"venv VLA : {'✓' if VENV_PY.exists() else '✗ absent'}\n"
                       f"install/ : "
                       f"{'✓' if (WORKSPACE / 'install').exists() else '✗ à builder'}"),
                 bg=BG_CARD, fg=FG, font=("DejaVu Sans Mono", 9),
                 justify="left").pack(anchor="w")

        robot = self._card(parent, "ROBOT")
        self._card_robot_lbl = tk.Label(robot, text="—", bg=BG_CARD, fg=FG,
                                        font=("DejaVu Sans Mono", 9),
                                        justify="left")
        self._card_robot_lbl.pack(anchor="w")

        ds = self._card(parent, "DATASET COURANT")
        self._card_ds_lbl = tk.Label(ds, text="—", bg=BG_CARD, fg=FG,
                                     font=("DejaVu Sans Mono", 9),
                                     justify="left", wraplength=230)
        self._card_ds_lbl.pack(anchor="w")
        self._stats_lbl = tk.Label(ds, text="Chargement…", bg=BG_CARD, fg=FG_DIM,
                                   font=("DejaVu Sans Mono", 9), justify="left")
        self._stats_lbl.pack(anchor="w", pady=(4, 0))

        mdl = self._card(parent, "MODÈLE VLA")
        self._card_model_lbl = tk.Label(mdl, text="—", bg=BG_CARD, fg=FG,
                                        font=("DejaVu Sans Mono", 9),
                                        justify="left", wraplength=230)
        self._card_model_lbl.pack(anchor="w")
        tk.Label(mdl, text=f"{len(_detect_models())} modèle(s) détecté(s)",
                 bg=BG_CARD, fg=FG_DIM, font=("DejaVu Sans", 9)).pack(
            anchor="w", pady=(4, 0))

    def _model_name(self) -> str:
        m = re.search(r"outputs/train/([^/]+)/", self._ckpt.get())
        return m.group(1) if m else (Path(self._ckpt.get()).name or "—")

    def _on_ckpt_changed(self) -> None:
        name = self._model_name()
        lbl = getattr(self, "_card_model_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lbl.configure(text=f"{name}\n{self._ckpt.get()}")
        sbm = getattr(self, "_sb_model", None)
        if sbm is not None and sbm.winfo_exists():
            sbm.configure(text=f"🧠 {name}")

    def _on_raw_root_changed(self) -> None:
        raw = self._raw_root.get()
        lbl = getattr(self, "_card_ds_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lbl.configure(text=raw)
        sbd = getattr(self, "_sb_ds", None)
        if sbd is not None and sbd.winfo_exists():
            sbd.configure(text=f"📁 {Path(raw).name}")

    # ── Visualisation (colonne centrale) : Caméras | Robot ────────────
    def _build_center(self, parent: tk.Frame) -> None:
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x", padx=4, pady=(6, 2))
        tk.Label(head, text="Visualisation", bg=BG, fg=FG,
                 font=("DejaVu Sans", 12, "bold")).pack(side="left", padx=6)

        self._nb = ttk.Notebook(parent)
        self._nb.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # --- Onglet Caméras : front + wrist CÔTE À CÔTE, ▶/⏹ dans l'onglet ---
        cams = tk.Frame(self._nb, bg=LOG_BG)
        bar = tk.Frame(cams, bg=LOG_BG)
        bar.pack(fill="x", padx=6, pady=(6, 0))
        self._cam_btn = RButton(bar, "▶  Démarrer les flux", self._toggle_cam_feed,
                                bg=BG_CARD, fg=FG,
                                font=("DejaVu Sans", 9, "bold"),
                                padx=12, pady=6, radius=9)
        self._cam_btn.pack(side="right")
        self._tooltip(self._cam_btn,
                      "Lance ihm/camera_stream.py (nœud ROS → images en RAM, 15 Hz).\n"
                      "Nécessite la sim (ou le robot) DÉMARRÉE pour avoir des images.\n"
                      "Peut rester actif pendant un run — process indépendant.")
        body = tk.Frame(cams, bg=LOG_BG)
        body.pack(fill="both", expand=True)
        body.grid_rowconfigure(1, weight=1)
        for i, name in enumerate(CAM_TOPICS):
            body.grid_columnconfigure(i, weight=1, uniform="cam")
            tk.Label(body, text=name.upper(), bg=LOG_BG, fg=FG_DIM,
                     font=("DejaVu Sans", 10, "bold")).grid(
                row=0, column=i, pady=(4, 0))
            lbl = tk.Label(body, bg=LOG_BG, fg=FG_DIM, justify="center",
                           font=("DejaVu Sans", 10),
                           text="Flux arrêté — ▶ ci-dessus")
            lbl.grid(row=1, column=i, sticky="nsew", padx=4, pady=2)
            info = tk.Label(body, bg=LOG_BG, fg=FG_DIM,
                            font=("DejaVu Sans Mono", 8), anchor="w")
            info.grid(row=2, column=i, sticky="ew", padx=8, pady=(0, 4))
            self._cam_lbls[name] = lbl
            self._cam_info[name] = info
        self._nb.add(cams, text="  📷 Caméras  ")

        # --- Onglet Robot : bras TOUJOURS visible (schéma cinématique),
        #     pose HOME hors ligne, animé via /joint_states quand ROS tourne.
        #     (Embed de la fenêtre Gazebo impossible sous Wayland → vue dédiée.)
        rob = tk.Frame(self._nb, bg=LOG_BG)
        self._robot_canvas = tk.Canvas(rob, bg=LOG_BG, highlightthickness=0)
        self._robot_canvas.pack(fill="both", expand=True)
        self._robot_canvas.bind("<Configure>", lambda _e: self._draw_robot())
        self._nb.add(rob, text="  🦾 Robot  ")

    def _toggle_cam_feed(self) -> None:
        if self._cam_feed.running:
            self._cam_feed.stop()
            self._cam_btn.set_text("▶  Démarrer les flux")
            self._append_log("[IHM] Flux caméras arrêté.\n")
            for name, lbl in self._cam_lbls.items():
                lbl.configure(image="", text="Flux arrêté — ▶ ci-dessus")
                self._cam_imgs.pop(name, None)
                self._cam_mtime.pop(name, None)
                self._cam_info[name].configure(text="", fg=FG_DIM)
        else:
            if self._cam_feed.start():
                self._cam_btn.set_text("⏹  Arrêter les flux")
                self._append_log("[IHM] ▶ Flux caméras démarré (RAM, 15 Hz).\n")
                for lbl in self._cam_lbls.values():
                    lbl.configure(text="En attente d'images…")
            else:
                self._append_log("[IHM] ✗ Impossible de lancer camera_stream.py.\n")

    def _poll_cameras(self) -> None:
        """Polling des fichiers écrits par camera_stream.py (atomiques) :
        images PPM (décodage Tk natif, fluide) + joints.json (vue Robot)."""
        for name in CAM_TOPICS:
            img_f = CAM_CACHE / f"{name}.{CAM_EXT}"
            try:
                mt = img_f.stat().st_mtime
            except OSError:
                mt = 0.0
            lbl = self._cam_lbls.get(name)
            info = self._cam_info.get(name)
            if lbl is None or info is None or not lbl.winfo_exists():
                continue
            if mt and mt != self._cam_mtime.get(name):
                self._cam_mtime[name] = mt
                try:
                    img = tk.PhotoImage(file=str(img_f))
                    # Deux vues côte à côte : sous-échantillonne si trop large.
                    lw = lbl.winfo_width()
                    if lw > 40 and img.width() > lw:
                        img = img.subsample(-(-img.width() // lw))
                    self._cam_imgs[name] = img   # référence gardée (anti-GC)
                    lbl.configure(image=img, text="")
                except tk.TclError:
                    pass   # fichier illisible → on retentera au tick suivant
            if self._cam_feed.running and mt:
                age = time.time() - mt
                if age > 4.0:
                    info.configure(text=f"⚠ figé {age:.0f}s", fg=WARN_YEL)
                else:
                    info.configure(text=f"● {datetime.fromtimestamp(mt):%H:%M:%S}",
                                   fg=OK_GREEN)
            elif self._cam_feed.running:
                info.configure(text="… en attente", fg=FG_DIM)

        # Vue Robot : joints.json → redessine si la pose a changé.
        jf = CAM_CACHE / "joints.json"
        try:
            jmt = jf.stat().st_mtime
        except OSError:
            jmt = 0.0
        if jmt and jmt != self._joints_mtime:
            self._joints_mtime = jmt
            try:
                d = json.loads(jf.read_text())
                self._robot_joints = [
                    float(d.get(f"joint{i}", self._robot_joints[i - 1]))
                    for i in range(1, 7)]
                self._robot_live = True
                self._draw_robot()
            except (ValueError, OSError):
                pass
        elif self._robot_live and (not jmt or time.time() - jmt > 5.0):
            self._robot_live = False   # flux coupé → on garde la dernière pose
            self._draw_robot()
        self.after(100, self._poll_cameras)

    def _draw_robot(self) -> None:
        """Schéma cinématique 2D du ReBeL (vue de profil, longueurs indicatives) :
        toujours visible, pose HOME hors ligne, live via /joint_states.
        j2/j3/j5 = plan vertical du bras ; j1/j4/j6 affichés en degrés."""
        c = getattr(self, "_robot_canvas", None)
        if c is None or not c.winfo_exists():
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 80 or h < 80:
            return
        c.delete("all")
        j = self._robot_joints
        gy = h - 46                                   # ligne de sol
        c.create_line(0, gy, w, gy, fill=_lighten(LOG_BG, 0.18), width=2)
        scale = min(w * 0.42, h * 0.62) / 0.87        # bras ≈ 0.87 m déployé
        x, y, a = w * 0.5, gy - 8.0, 0.0              # base ; 0 rad = vertical
        c.create_rectangle(x - 30, gy - 8, x + 30, gy, fill=BG_CARD, outline="")
        # (longueur m, articulation appliquée avant le segment)
        segs = [(0.135, 0.0), (0.30, j[1]), (0.32, j[2]), (0.10, j[4]), (0.09, 0.0)]
        pts = [(x, y)]
        for length, dj in segs:
            a += dj
            x += length * scale * math.sin(a)
            y -= length * scale * math.cos(a)
            pts.append((x, y))
        widths = (16, 13, 11, 8, 6)
        colors = (ACCENT, ACCENT, _lighten(ACCENT, 0.12),
                  _lighten(ACCENT, 0.25), FG_DIM)
        for i in range(len(pts) - 1):
            c.create_line(*pts[i], *pts[i + 1], width=widths[i],
                          fill=colors[i], capstyle="round")
        for px, py in pts[1:-1]:
            c.create_oval(px - 5, py - 5, px + 5, py + 5,
                          fill=BG_PANEL, outline=FG_DIM)
        gx, gyy = pts[-1]
        c.create_oval(gx - 4, gyy - 4, gx + 4, gyy + 4, fill=FG, outline="")
        c.create_text(14, 12, anchor="nw",
                      text=("● LIVE — /joint_states" if self._robot_live
                            else "Pose HOME (hors ligne)"),
                      fill=(OK_GREEN if self._robot_live else FG_DIM),
                      font=("DejaVu Sans", 10, "bold"))
        degs = "   ".join(f"j{i + 1} {math.degrees(v):+.0f}°"
                          for i, v in enumerate(j))
        c.create_text(14, h - 26, anchor="nw", text=degs,
                      fill=FG_DIM, font=("DejaVu Sans Mono", 9))

    # ── Moniteur de run + journal (zone basse) ────────────────────────
    def _build_run_monitor(self, parent: tk.Frame) -> None:
        """Bandeau de suivi live : état coloré + progression gardés/N + taux + causes."""
        mon = tk.Frame(parent, bg=BG_PANEL)
        mon.pack(fill="x", padx=4, pady=(4, 4))
        top = tk.Frame(mon, bg=BG_PANEL)
        top.pack(fill="x", padx=8, pady=(6, 2))
        self._state_dot = tk.Label(top, text="●", bg=BG_PANEL, fg=FG_DIM,
                                   font=("DejaVu Sans", 14))
        self._state_dot.pack(side="left")
        self._state_lbl = tk.Label(top, text="Prêt", bg=BG_PANEL, fg=FG,
                                   font=("DejaVu Sans", 11, "bold"))
        self._state_lbl.pack(side="left", padx=(4, 12))
        self._mon_eta = tk.Label(top, text="", bg=BG_PANEL, fg=FG_DIM,
                                 font=("DejaVu Sans", 10))
        self._mon_eta.pack(side="right")
        self._progress = ttk.Progressbar(mon, mode="determinate", maximum=100, value=0)
        self._progress.pack(fill="x", padx=8, pady=2)
        self._mon_detail = tk.Label(mon, text="Aucun run en cours.", bg=BG_PANEL,
                                    fg=FG_DIM, font=("DejaVu Sans Mono", 9),
                                    justify="left", anchor="w")
        self._mon_detail.pack(fill="x", padx=8, pady=(0, 6))

    def _build_log(self, parent: tk.Frame) -> None:
        lf = tk.Frame(parent, bg=BG)
        lf.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        head = tk.Frame(lf, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="Journal", bg=BG, fg=FG,
                 font=("DejaVu Sans", 11, "bold")).pack(side="left")
        for val, lbl in [("tout", "Tout"), ("erreurs", "⚠ Erreurs"),
                         ("succes", "✓ Succès")]:
            tk.Radiobutton(head, text=lbl, variable=self._log_filter, value=val,
                           command=self._apply_log_filter, bg=BG, fg=FG_DIM,
                           selectcolor=BG_PANEL, activebackground=BG,
                           font=("DejaVu Sans", 9)).pack(side="left", padx=2)
        for txt, cmd, tip in (("📋", self._copy_log, "Copier le journal"),
                              ("🧹", self._clear_log, "Effacer le journal (Ctrl+L)")):
            b = tk.Button(head, text=txt, command=cmd, bg=BG, fg=FG_DIM, bd=0,
                          font=("DejaVu Sans", 11), padx=6,
                          activebackground=BG_PANEL)
            b.pack(side="right", padx=2)
            self._tooltip(b, tip)
        tf = tk.Frame(lf, bg=BG)
        tf.pack(fill="both", expand=True)
        self._log = tk.Text(tf, bg=LOG_BG, fg="#cdd9e5", insertbackground=FG,
                            font=("DejaVu Sans Mono", 9), wrap="word")
        sb = ttk.Scrollbar(tf, command=self._log.yview)
        sb.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True, side="left")
        self._log.configure(yscrollcommand=sb.set, state="disabled")
        # Colorisation : erreurs en rouge, succès en vert, messages [IHM] en orange.
        self._log.tag_configure("err", foreground="#ff8a8a")
        self._log.tag_configure("ok", foreground="#7ee2a8")
        self._log.tag_configure("ihm", foreground=ACCENT)

    # ------------------------------------------------------------------
    #  Panneau Paramètres — contenu selon le mode
    # ------------------------------------------------------------------
    def _render_mode(self) -> None:
        for w in self._left.winfo_children():
            w.destroy()
        for w in self._toolbar_actions.winfo_children():
            w.destroy()
        mode = self._mode.get()
        if mode == "simu":
            self._build_simu_params()
            self._build_simu_actions()
            self._sb_mode.configure(text="SIMULATION", fg=ACCENT)
            self._card_robot_lbl.configure(
                text="IGUS ReBeL 6 DOF\nBackend : Gazebo Fortress\n"
                     "DDS : localhost only")
        else:
            self._build_real_params()
            self._build_real_actions()
            self._sb_mode.configure(text="RÉEL (CRI)", fg=ACCENT_REAL)
            self._card_robot_lbl.configure(
                text="IGUS ReBeL 6 DOF\nCRI : 192.168.3.11:3920\n"
                     "via Raspberry Pi\n⚠ CRIBackend = stub")

    def _section(self, title: str) -> tk.Frame:
        """Crée un sous-panneau titré dans le panneau Paramètres."""
        tk.Label(self._left, text=title, bg=BG_PARAM, fg=ACCENT,
                 font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(10, 2))
        tk.Frame(self._left, bg=_lighten(BG_PARAM, 0.12), height=1).pack(
            fill="x", padx=2, pady=(0, 4))
        f = tk.Frame(self._left, bg=BG_PARAM)
        f.pack(fill="x")
        return f

    def _row(self, parent: tk.Frame, label: str, var: tk.StringVar, w: int = 7,
             tooltip: str = "") -> tk.Entry:
        r = tk.Frame(parent, bg=BG_PARAM)
        r.pack(fill="x", pady=2)
        _label(r, label).pack(side="left", padx=(4, 6))
        e = _entry(r, var, w)
        e.pack(side="left")
        if tooltip:
            self._tooltip(e, tooltip)
        return e

    # ── Simulation ──────────────────────────────────────────────────
    def _build_simu_params(self) -> None:
        # --- Modèle VLA (en HAUT du panneau : visible sans scroller) ---
        self._build_ckpt_selector()

        # --- N épisodes ---
        f_ep = self._section("Épisodes")
        r = tk.Frame(f_ep, bg=BG_PARAM)
        r.pack(fill="x", pady=2)
        _label(r, "N cycles :").pack(side="left", padx=(4, 6))
        sp = tk.Spinbox(r, textvariable=self._n_episodes, from_=1, to=500, width=6,
                        bg=ENTRY_BG, fg=FG, insertbackground=FG,
                        buttonbackground=BG_PANEL, font=("DejaVu Sans Mono", 11))
        sp.pack(side="left")
        _label(r, "(1–500)", dim=True).pack(side="left", padx=6)

        r2 = tk.Frame(f_ep, bg=BG_PARAM)
        r2.pack(fill="x", pady=2)
        tk.Checkbutton(r2, text="Randomiser les positions pick",
                       variable=self._randomize,
                       bg=BG_PARAM, fg=FG, selectcolor=BG_PANEL,
                       activebackground=BG_PARAM, font=("DejaVu Sans", 11),
                       command=self._on_randomize_toggle).pack(side="left", padx=4)

        r2b = tk.Frame(f_ep, bg=BG_PARAM)
        r2b.pack(fill="x", pady=2)
        cb_night = tk.Checkbutton(r2b, text="🌙 Mode nuit (headless + IHM masquée)",
                                  variable=self._night_mode,
                                  bg=BG_PARAM, fg=FG, selectcolor=BG_PANEL,
                                  activebackground=BG_PARAM, font=("DejaVu Sans", 11))
        cb_night.pack(side="left", padx=4)
        self._tooltip(cb_night,
                      "Pour les longs runs sans surveillance :\n"
                      "• Gazebo en headless (pas de fenêtre 3D)\n"
                      "• l'IHM se minimise dès le lancement (CPU libéré)\n"
                      "→ moins de risque de caméra figée. L'IHM réapparaît à la fin.\n"
                      "N'affecte PAS le « Test 1 cycle » (toujours en GUI).")

        # --- Blending Pilz (lissage trajectoire, prépa v3) ---
        r2c = tk.Frame(f_ep, bg=BG_PARAM)
        r2c.pack(fill="x", pady=2)
        _label(r2c, "Blend (cm) :").pack(side="left", padx=(4, 6))
        sp_bl = tk.Spinbox(r2c, textvariable=self._blend_cm, from_=0, to=9,
                           width=4, bg=ENTRY_BG, fg=FG, insertbackground=FG,
                           buttonbackground=BG_PANEL,
                           font=("DejaVu Sans Mono", 11))
        sp_bl.pack(side="left")
        _label(r2c, "0 = arrêts classiques", dim=True).pack(side="left", padx=6)
        self._tooltip(sp_bl,
                      "Rayon de lissage des coins de trajectoire (Pilz).\n"
                      "0 cm : le bras s'arrête à chaque étape (comportement v1/v2).\n"
                      "3-6 cm : coins arrondis SANS arrêt aux approches et au retour\n"
                      "→ cycle plus rapide et démos plus fluides (collecte v3).\n"
                      "Segment d'approche : 7 cm ; au-delà de 6 cm de rayon, la\n"
                      "hauteur d'approche/levée s'adapte automatiquement (rayon + 1 cm).\n"
                      "Si l'arc dépasse une limite de vitesse articulaire, le rayon\n"
                      "est réduit de moitié automatiquement (voir journal « ↘ »).\n"
                      "Le stop de vérification de saisie après la levée est conservé.")

        # --- Dossier dataset (sélecteur déroulant + éditable) ---
        r3 = tk.Frame(f_ep, bg=BG_PARAM)
        r3.pack(fill="x", pady=2)
        _label(r3, "Dataset :").pack(side="left", padx=(4, 6))
        cb_ds = ttk.Combobox(r3, textvariable=self._raw_root, width=18,
                             values=_list_datasets(), font=("DejaVu Sans Mono", 10))
        cb_ds.pack(side="left")
        # Rafraîchit la liste à l'ouverture du menu + recalcule les stats au choix
        cb_ds.bind("<Button-1>",
                   lambda _e: cb_ds.configure(values=_list_datasets()))
        cb_ds.bind("<<ComboboxSelected>>", lambda _e: self._refresh_stats_now())
        tk.Button(r3, text="📁", command=self._browse_dataset,
                  bg=BG_PANEL, fg=FG, bd=0, font=("DejaVu Sans", 11),
                  padx=6).pack(side="left", padx=(4, 0))

        # --- Coordonnées pick ---
        f_pick = self._section("Coordonnées pick (m)")
        # Curseurs couplés X/Y : bouger un axe borne l'autre via √(x²+y²) ≤ cap.
        self._build_xy_sliders(f_pick)
        self._row(f_pick, "Z pick :", self._pick_z,
                  tooltip="Hauteur de saisie = mi-hauteur de l'objet "
                          "(18 mm pour la roulette)")

        # Label d'état de portée : vert si OK, rouge si hors d'atteinte du bras.
        self._reach_lbl = tk.Label(f_pick, text="", bg=BG_PARAM, fg=FG_DIM,
                                   font=("DejaVu Sans", 10, "bold"))
        self._reach_lbl.pack(anchor="w", padx=4, pady=(4, 0))

        _label(f_pick, "Workspace (aléatoire) — anneau v3 :", dim=True).pack(
            anchor="w", padx=4, pady=(4, 0))
        _label(f_pick, "  0.15 ≤ √(x²+y²) ≤ 0.54 m", dim=True).pack(anchor="w", padx=4)
        _label(f_pick, "  y ≥ −0.52 m (pas derrière caméra)", dim=True).pack(
            anchor="w", padx=4)

        self._validate_reach()   # couleur initiale

        # --- Coordonnées place : bac FIXE → backend, masqué sauf mode avancé ---
        if self._advanced.get():
            f_pl = self._section("Coordonnées place / bac (m)")
            self._row(f_pl, "X bac :", self._place_x,
                      tooltip="X du bac de dépôt (fixe)")
            self._row(f_pl, "Y bac :", self._place_y,
                      tooltip="Y du bac de dépôt (fixe)")
            self._row(f_pl, "Z bac :", self._place_z,
                      tooltip="Hauteur de lâcher (m)")

    # ── Sélecteur de modèle VLA ───────────────────────────────────────
    def _build_ckpt_selector(self) -> None:
        """Menu déroulant (compact) des modèles détectés dans outputs/train/
        + champ chemin (avancé). Choisir un modèle remplit ``self._ckpt`` ;
        un chemin custom tapé dans le champ reste prioritaire (le menu ne
        l'écrase que si on re-sélectionne une entrée).
        """
        tk.Label(self._left, text="Modèle VLA", bg=BG_PARAM, fg=ACCENT,
                 font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(10, 2))
        tk.Frame(self._left, bg=_lighten(BG_PARAM, 0.12), height=1).pack(
            fill="x", padx=2, pady=(0, 4))
        hints = {"v1_smolvla": "1 caméra (front)",
                 "v2_smolvla": "2 caméras (front+wrist)"}
        by_label = {(f"{n} — {hints[n]}" if n in hints else n): p
                    for n, p in _detect_models().items()}
        labels = list(by_label) or ["(aucun checkpoint dans outputs/train/)"]
        # Présélection : l'entrée dont le chemin == checkpoint courant.
        current = next((l for l, p in by_label.items()
                        if p == self._ckpt.get()), labels[0])
        var = tk.StringVar(value=current)
        combo = ttk.Combobox(self._left, textvariable=var, values=labels,
                             state="readonly", font=("DejaVu Sans Mono", 10))
        combo.pack(fill="x", padx=4, pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda _e: self._ckpt.set(by_label.get(var.get(),
                                                          self._ckpt.get())))
        # Chemin brut du checkpoint : backend — visible seulement en mode avancé
        # (Affichage → « Paramètres avancés »).
        if self._advanced.get():
            tk.Label(self._left, text="Chemin checkpoint (avancé)", bg=BG_PARAM,
                     fg=FG_DIM, font=("DejaVu Sans", 9)).pack(anchor="w", padx=4,
                                                              pady=(4, 0))
            tk.Entry(self._left, textvariable=self._ckpt, width=36,
                     bg=ENTRY_BG, fg=FG, insertbackground=FG,
                     font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=4, pady=2)

    def _on_randomize_toggle(self) -> None:
        state = "normal" if not self._randomize.get() else "readonly"
        # quand randomize=True, les champs pick x/y sont indicatifs seulement
        # (la commande ignorera pick_x/y dans ce cas)

    # ── Curseurs couplés X/Y (contrainte √(x²+y²) ≤ MAX_REACH_XY) ─────
    def _build_xy_sliders(self, parent: tk.Frame) -> None:
        """Deux curseurs X et Y avec valeur affichée + éditable.

        Contrainte de portée : √(x²+y²) ≤ MAX_REACH_XY (= rayon de travail).
        Couplage bidirectionnel : l'axe que l'on bouge est « maître » ; l'autre
        est borné à son max (y_max = √(R²−x²), x_max = √(R²−y²)) et la plage de
        son curseur est rétrécie en conséquence. Ex. X = 0.60 → Y max = 0.
        Les champs Entry restent éditables au clavier (mêmes StringVars).
        """
        R = MAX_REACH_XY

        # --- Axe X (avant du robot : 0 → R) ---
        rx = tk.Frame(parent, bg=BG_PARAM)
        rx.pack(fill="x", pady=(4, 0))
        _label(rx, "X pick :").pack(side="left", padx=(4, 6))
        self._pick_x_entry = _entry(rx, self._pick_x, w=7)
        self._pick_x_entry.pack(side="left")
        _label(rx, "m", dim=True).pack(side="left", padx=(4, 0))
        self._tooltip(self._pick_x_entry,
                      "X de la zone de saisie (m) — utilisé si randomise=OFF.\n"
                      "Bornée par √(x²+y²) ≤ {:.2f} m.".format(R))
        self._scale_x = tk.Scale(
            parent, from_=0.0, to=R, resolution=0.005, orient="horizontal",
            showvalue=False, command=lambda v: self._on_scale("x", v),
            bg=BG_PARAM, fg=FG, troughcolor=ENTRY_BG, highlightthickness=0,
            activebackground=ACCENT, sliderrelief="flat", length=300)
        self._scale_x.pack(fill="x", padx=4)
        self._x_max_lbl = _label(parent, "", dim=True)
        self._x_max_lbl.pack(anchor="w", padx=4)

        # --- Axe Y (latéral : −R → R) ---
        ry = tk.Frame(parent, bg=BG_PARAM)
        ry.pack(fill="x", pady=(8, 0))
        _label(ry, "Y pick :").pack(side="left", padx=(4, 6))
        self._pick_y_entry = _entry(ry, self._pick_y, w=7)
        self._pick_y_entry.pack(side="left")
        _label(ry, "m", dim=True).pack(side="left", padx=(4, 0))
        self._tooltip(self._pick_y_entry,
                      "Y de la zone de saisie (m) — utilisé si randomise=OFF.\n"
                      "Bornée par √(x²+y²) ≤ {:.2f} m.".format(R))
        self._scale_y = tk.Scale(
            parent, from_=-R, to=R, resolution=0.005, orient="horizontal",
            showvalue=False, command=lambda v: self._on_scale("y", v),
            bg=BG_PARAM, fg=FG, troughcolor=ENTRY_BG, highlightthickness=0,
            activebackground=ACCENT, sliderrelief="flat", length=300)
        self._scale_y.pack(fill="x", padx=4)
        self._y_max_lbl = _label(parent, "", dim=True)
        self._y_max_lbl.pack(anchor="w", padx=4)

        # Position initiale des curseurs + affichage des max.
        self._recouple(driver="x")

    def _on_scale(self, axis: str, value: str) -> None:
        """Callback de glissement d'un curseur : écrit la valeur dans le champ
        puis recouple l'autre axe et revalide la portée."""
        if self._coupling:
            return
        var = self._pick_x if axis == "x" else self._pick_y
        self._coupling = True
        try:
            var.set(f"{float(value):.3f}")
        finally:
            self._coupling = False
        self._recouple(driver=axis)
        self._validate_reach()

    def _on_pick_xy_edited(self, axis: str) -> None:
        """Trace des StringVars X/Y (frappe clavier ou écriture programmée).
        `axis` = axe édité = maître du couplage."""
        if self._coupling:
            return
        self._recouple(driver=axis)
        self._validate_reach()

    def _recouple(self, driver: str) -> None:
        """Applique √(x²+y²) ≤ MAX_REACH_XY en gardant `driver` comme maître.

        L'axe maître est seulement borné à sa plage absolue ; l'axe esclave est
        ramené à son max courant. Met à jour les deux curseurs (valeur + plage)
        et les labels « X max / Y max ». Sans effet hors mode simu (pas de
        curseurs) ou pendant une saisie non numérique."""
        sx = getattr(self, "_scale_x", None)
        sy = getattr(self, "_scale_y", None)
        if sx is None or sy is None or not sx.winfo_exists() or not sy.winfo_exists():
            return
        try:
            x = float(self._pick_x.get())
            y = float(self._pick_y.get())
        except (ValueError, tk.TclError):
            return  # saisie en cours → _validate_reach affichera l'erreur
        R = MAX_REACH_XY
        if driver == "x":
            x = min(max(x, 0.0), R)
            y_lim = (max(0.0, R * R - x * x)) ** 0.5
            y = min(max(y, -y_lim), y_lim)
        else:
            y = min(max(y, -R), R)
            x_lim = (max(0.0, R * R - y * y)) ** 0.5
            x = min(max(x, 0.0), x_lim)
        x_max = (max(0.0, R * R - y * y)) ** 0.5   # plage de X sachant Y
        y_max = (max(0.0, R * R - x * x)) ** 0.5   # plage de Y sachant X

        self._coupling = True
        try:
            self._pick_x.set(f"{x:.3f}")
            self._pick_y.set(f"{y:.3f}")
            self._scale_x.config(to=max(x_max, 1e-3))
            self._scale_y.config(from_=-max(y_max, 1e-3), to=max(y_max, 1e-3))
            self._scale_x.set(x)
            self._scale_y.set(y)
        finally:
            self._coupling = False

        self._x_max_lbl.config(text=f"   X max = {x_max:.3f} m   (sachant Y = {y:.3f})")
        self._y_max_lbl.config(text=f"   Y max = ±{y_max:.3f} m   (sachant X = {x:.3f})")

    # ── Validation de la portée (pick_x, pick_y) ─────────────────────
    def _validate_reach(self) -> bool:
        """Vérifie que (pick_x, pick_y) est atteignable par le bras.
        Colore les champs X/Y en rouge si hors portée + met à jour le label
        d'état. Renvoie True si la position est valide."""
        try:
            x = float(self._pick_x.get())
            y = float(self._pick_y.get())
        except (ValueError, AttributeError, tk.TclError):
            self._set_reach_state("⛔ coordonnées non numériques", ok=False)
            return False
        dist = (x * x + y * y) ** 0.5
        if dist > MAX_REACH_XY:
            self._set_reach_state(
                f"⛔ {dist:.3f} m > {MAX_REACH_XY:.2f} m — HORS PORTÉE", ok=False)
            return False
        if dist < MIN_REACH_XY:
            self._set_reach_state(
                f"⛔ {dist:.3f} m < {MIN_REACH_XY:.2f} m — trop près de la base",
                ok=False)
            return False
        if (abs(x - BIN_CENTER_X) < BIN_KEEPOUT
                and abs(y - BIN_CENTER_Y) < BIN_KEEPOUT):
            self._set_reach_state(
                "⛔ dans la zone du bac — déplace en y négatif", ok=False)
            return False
        self._set_reach_state(
            f"✓ portée {dist:.3f} m  (max {MAX_REACH_XY:.2f} m)", ok=True)
        return True

    def _set_reach_state(self, msg: str, ok: bool) -> None:
        """Applique la couleur (vert/rouge) aux champs X/Y et au label d'état."""
        color   = OK_GREEN if ok else STOP_RED
        bg      = ENTRY_BG  if ok else ENTRY_BG_BAD
        for e in (getattr(self, "_pick_x_entry", None),
                  getattr(self, "_pick_y_entry", None)):
            if e is not None and e.winfo_exists():
                e.configure(bg=bg, highlightbackground=color, highlightcolor=color)
        lbl = getattr(self, "_reach_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lbl.configure(text=msg, fg=color)

    def _build_real_params(self) -> None:
        tk.Label(self._left, text="Robot réel (CRI)", bg=BG_PARAM, fg=ACCENT_REAL,
                 font=("DejaVu Sans", 12, "bold")).pack(anchor="w", pady=(4, 6))
        tk.Label(self._left,
                 text="IP : 192.168.3.11\nPort : 3920\nRéseau : via Raspberry Pi",
                 bg=BG_PARAM, fg=FG, font=("DejaVu Sans Mono", 11),
                 justify="left").pack(anchor="w", padx=4, pady=4)
        tk.Label(self._left,
                 text="⚠ CRIBackend = STUB\n(non implémenté)",
                 bg=BG_PARAM, fg=WARN_YEL, font=("DejaVu Sans", 11, "bold")).pack(
            anchor="w", padx=4, pady=8)
        tk.Label(self._left, text="Coordonnées expert (réel)", bg=BG_PARAM, fg=ACCENT,
                 font=("DejaVu Sans", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self._row(self._left, "X pick :", self._pick_x)
        self._row(self._left, "Y pick :", self._pick_y)
        if self._advanced.get():   # bac fixe → mode avancé uniquement
            self._row(self._left, "X bac :", self._place_x)
            self._row(self._left, "Y bac :", self._place_y)
        self._build_ckpt_selector()

    # ------------------------------------------------------------------
    #  Construction des commandes ROS2
    # ------------------------------------------------------------------
    def _with_killall(self, launch_cmd: str) -> str:
        """Préfixe une commande de lancement sim par ./kill_all.sh (purge DDS/Gazebo).
        Évite le piège n°1 : segments FastDDS /dev/shm résiduels d'un run précédent →
        gz_ros2_control ne reçoit pas /robot_description → robot non spawné → run vide."""
        ka = shlex.quote(str(WORKSPACE / "kill_all.sh"))
        return f"echo '[IHM] Nettoyage DDS/Gazebo…' && bash {ka} && sleep 2 && {launch_cmd}"

    def _blend_m(self) -> str:
        """Rayon de blending du champ IHM (cm) → mètres pour le launch.
        Saisie invalide = 0 (arrêts classiques, comportement historique)."""
        try:
            cm = max(0.0, min(9.0, float(self._blend_cm.get().replace(",", "."))))
        except (ValueError, tk.TclError):
            cm = 0.0
        return f"{cm / 100.0:.3f}"

    def _record_cmd(self, n_episodes: int = -1, randomize: bool | None = None,
                    test_only: bool = False) -> str:
        n   = n_episodes if n_episodes > 0 else int(self._n_episodes.get())
        rnd = randomize if randomize is not None else self._randomize.get()
        raw = self._raw_root.get()
        px  = self._pick_x.get()
        py  = self._pick_y.get()
        pz  = self._pick_z.get()
        plx = self._place_x.get()
        ply = self._place_y.get()
        plz = self._place_z.get()

        # grasp_radius : figé dans le backend (gripper_shim.py), plus passé ici.
        args = (f"num_episodes:={n} "
                f"raw_root:={shlex.quote(raw)} "
                f"randomize:={'true' if rnd else 'false'} "
                f"pick_x:={px} pick_y:={py} pick_z:={pz} "
                f"place_x:={plx} place_y:={ply} place_z:={plz} "
                f"blend_radius:={self._blend_m()} "
                f"sim_no_calib:=true allow_zero_state:=true "
                f"headless:={'true' if self._night_mode.get() else 'false'}")
        launch = f"{SIM_ENV} && ros2 launch igus_vla record_demos.launch.py {args}"
        return self._with_killall(launch)

    def _run_record(self) -> None:
        try:
            n = int(self._n_episodes.get())
            if n < 1 or n > 500:
                raise ValueError
        except ValueError:
            messagebox.showerror("Erreur", "Nombre d'épisodes invalide (entier 1–500).")
            return
        # En randomisation, les positions sont tirées dans le workspace calibré
        # (toujours atteignable) → on ne valide pick_x/y QUE si randomize=OFF.
        if not self._randomize.get() and not self._validate_reach():
            messagebox.showerror(
                "Position hors portée",
                "La position de pick dépasse la portée du bras "
                f"(max {MAX_REACH_XY:.2f} m en XY).\n\n"
                "Corrige X/Y (champs en rouge) avant de lancer.")
            return
        # Capture la cible + le dossier suivi pour le moniteur de run live.
        self._run_target = n
        self._run_active_raw = self._raw_root.get()
        cmd = self._record_cmd()
        self._run_cmd(cmd, f"Enregistrer {n} démos", kind="record")
        # Mode nuit : minimise l'IHM (CPU libéré) une fois le run bien lancé.
        if self._night_mode.get() and self._runner.running:
            self._night_active = True
            self._status.config(text=f"🌙 Mode nuit — run {n} ép. en cours, IHM masquée.")
            self.after(1200, self._enter_night_hidden)

    def _enter_night_hidden(self) -> None:
        """Minimise l'IHM pendant un run en mode nuit (contenu non rendu → CPU libéré)."""
        if self._night_active and self._runner.running:
            try:
                self.iconify()
            except tk.TclError:
                pass

    def _run_one_test(self) -> None:
        """1 cycle pick&place SANS enregistrement — pour valider le grasp visuellement."""
        # Le test utilise toujours pick_x/pick_y (randomize=false) → on bloque si
        # la position est hors d'atteinte du bras (sinon l'expert refuse le cycle).
        if not self._validate_reach():
            messagebox.showerror(
                "Position hors portée",
                "La position de pick dépasse la portée du bras "
                f"(max {MAX_REACH_XY:.2f} m en XY).\n\n"
                "Corrige X/Y (champs en rouge) avant de lancer le test.")
            return
        # On substitue le recorder/orchestrateur par l'expert seul (sans auto_record)
        cmd = (f"{SIM_ENV} && ros2 launch igus_vla record_demos.launch.py "
               f"num_episodes:=1 "
               # Test = EXACTEMENT 1 essai (sinon, fill_to_target rejouerait le raté).
               f"fill_to_target:=false "
               f"randomize:=false "
               f"pick_x:={self._pick_x.get()} pick_y:={self._pick_y.get()} "
               f"pick_z:={self._pick_z.get()} "
               f"place_x:={self._place_x.get()} place_y:={self._place_y.get()} "
               f"place_z:={self._place_z.get()} "
               f"blend_radius:={self._blend_m()} "
               f"sim_no_calib:=true allow_zero_state:=true "
               f"headless:=false success_filter:=false")
        self._run_cmd(self._with_killall(cmd), "Test 1 cycle (validation grasp)")

    def _run_preview_randomization(self) -> None:
        """APERÇU randomisation : ouvre Gazebo et téléporte la roulette en boucle
        rapide à des positions randomisées, SANS cycle ni enregistrement.
        Sert à valider visuellement la distribution (anneau + exclusion bac)."""
        cmd = (f"{SIM_ENV} && ros2 launch igus_vla preview_randomization.launch.py "
               f"preview_period:=0.5 preview_count:=0 "
               f"place_x:={self._place_x.get()} place_y:={self._place_y.get()} "
               f"headless:=false")
        self._run_cmd(self._with_killall(cmd), "Test randomisation (aperçu)")

    def _run_visibility_sweep(self) -> None:
        """BALAYAGE VISIBILITÉ : place la roulette sur une grille le plus vite
        possible, détecte automatiquement si elle est visible par la caméra, et
        écrit un CSV + carte + zones à exclure (datasets/visibility/)."""
        cmd = (f"{SIM_ENV} && ros2 launch igus_vla visibility_sweep.launch.py "
               f"place_x:={self._place_x.get()} place_y:={self._place_y.get()} "
               f"headless:=false")
        self._run_cmd(self._with_killall(cmd), "Balayage visibilité (auto)")

    def _run_convert(self) -> None:
        """Convertit le dataset CHOISI (toutes caméras) en LeRobotDataset.
        repo_id / dossier de sortie dérivés du nom du dataset → pas d'écrasement,
        et la sélection 📁 de l'IHM pilote la conversion (override --raw-root)."""
        raw = self._raw_root.get().strip()
        raw_path = WORKSPACE / raw if not os.path.isabs(raw) else Path(raw)
        if not raw_path.exists() or not any(raw_path.glob("episode_*")):
            messagebox.showerror(
                "Dataset introuvable",
                f"Aucun dossier episode_* dans :\n{raw_path}\n\n"
                "Choisis un dataset valide avec le bouton 📁.")
            return
        name = Path(raw).name                      # ex. raw_v2
        repo_id = f"dbal67/igus_rebel_{name}"
        out = f"datasets/lerobot_{name}"           # sortie dédiée (pas d'écrasement)
        out_path = WORKSPACE / out
        if out_path.exists() and any(out_path.iterdir()):
            if not messagebox.askyesno(
                    "Sortie déjà existante",
                    f"{out_path}\n existe déjà et n'est pas vide.\n\n"
                    "La conversion refusera d'écraser. Ouvrir le dossier pour le "
                    "supprimer manuellement ?"):
                return
            self._open_folder(out_path)
            return
        cfg = shlex.quote(str(PKG_DIR / "config" / "dataset.yaml"))
        cmd = _vla_python(
            f"igus_vla.to_lerobot_dataset --config {cfg} "
            f"--raw-root {shlex.quote(raw)} "
            f"--repo-id {shlex.quote(repo_id)} "
            f"--root {shlex.quote(out)}")
        self._run_cmd(cmd, f"Convertir {name} → LeRobot ({out})", kind="convert")

    def _run_deploy_sim(self) -> None:
        # venv_python : le launch installé (share/) calcule un .venv/ à côté de
        # lui-même, qui n'existe pas → on lui passe le venv réel (src/igus_vla).
        # device=cuda : inférence SmolVLA temps réel (le défaut cpu est trop lent).
        cmd = (f"{SIM_ENV} && ros2 launch igus_vla vla_deploy.launch.py "
               f"checkpoint:={shlex.quote(self._ckpt.get())} "
               f"device:=cuda "
               f"venv_python:={shlex.quote(str(VENV_PY))}")
        self._run_cmd(cmd, "Déployer VLA (sim)")

    def _run_deploy_real(self) -> None:
        if not messagebox.askyesno("Backend CRI (stub)",
                                   "CRIBackend est un STUB non fonctionnel.\n"
                                   "Lancer quand même ?"):
            return
        cmd = (f"{REAL_ENV} && ros2 launch igus_vla vla_deploy.launch.py "
               f"backend:=cri checkpoint:={shlex.quote(self._ckpt.get())} "
               f"venv_python:={shlex.quote(str(VENV_PY))}")
        self._run_cmd(cmd, "Déployer VLA (réel)")

    # ------------------------------------------------------------------
    #  Stats dataset
    # ------------------------------------------------------------------
    def _refresh_stats_now(self) -> None:
        """Met à jour le label stats une fois (sans reprogrammer le timer)."""
        raw = WORKSPACE / self._raw_root.get()
        s = _dataset_stats(raw)
        if s["total"] == 0:
            txt = "Aucun épisode enregistré."
        else:
            rate = 100 * s["success"] / s["total"] if s["total"] else 0
            txt = (f"Épisodes : {s['total']}  "
                   f"✓ {s['success']}  ✗ {s['fail']}\n"
                   f"Taux succès : {rate:.0f}%")
            if s["last_pick"]:
                txt += (f"\nDernière pick : ({s['last_pick'][0]:.3f}, "
                        f"{s['last_pick'][1]:.3f}) m")
        # Mise à jour du label stats (carte Dataset, colonne gauche)
        try:
            self._stats_lbl.config(text=txt)
        except AttributeError:
            pass

    def _refresh_stats(self) -> None:
        self._refresh_stats_now()
        self.after(3000, self._refresh_stats)

    # ------------------------------------------------------------------
    #  Moniteur de run live (progression gardés/N + taux + causes + état)
    # ------------------------------------------------------------------
    @staticmethod
    def _hms(s: float) -> str:
        s = int(max(0, s))
        return (f"{s // 3600:d}h{(s % 3600) // 60:02d}m" if s >= 3600
                else f"{s // 60:d}m{s % 60:02d}s")

    @staticmethod
    def _read_run_progress(csv_path: Path) -> dict | None:
        """Lit rapport_run_courant.csv → {attempts, saved, ok, causes, elapsed}."""
        import csv as _csv
        res = {"attempts": 0, "saved": 0, "ok": 0, "causes": {}, "elapsed": 0.0}
        try:
            with open(csv_path) as f:
                for row in _csv.DictReader(f):
                    res["attempts"] += 1
                    if row.get("saved") == "1":
                        res["saved"] += 1
                    if row.get("success") == "1":
                        res["ok"] += 1
                    try:
                        res["elapsed"] += float(row.get("episode_s") or 0)
                    except ValueError:
                        pass
                    if row.get("saved") != "1":
                        r = (row.get("reason") or "").strip()
                        if r:
                            res["causes"][r] = res["causes"].get(r, 0) + 1
        except (FileNotFoundError, OSError):
            return None
        return res

    def _set_run_state(self, color: str, label: str) -> None:
        self._state_dot.configure(fg=color)
        self._state_lbl.configure(text=label, fg=color)
        # Miroir dans la barre d'état (LED + libellé compacts).
        dot = getattr(self, "_sb_dot", None)
        if dot is not None and dot.winfo_exists():
            dot.configure(fg=color)
            self._sb_state.configure(text=label, fg=color)

    def _refresh_run_monitor(self) -> None:
        running = self._runner.running
        # Chrono : temps écoulé depuis le début de la tâche, GELÉ à la fin du process.
        if running and self._task_start:
            self._task_elapsed = time.time() - self._task_start
        elapsed = self._task_elapsed

        if self._task_kind == "record":
            self._monitor_record(running, elapsed)
        elif self._task_kind in ("convert", "train", "other") and \
                (running or self._task_total or self._task_pct):
            self._monitor_generic(running, elapsed)
        elif not running:
            self._set_run_state(FG_DIM, "Prêt")
            self._mon_detail.configure(text="Aucune tâche en cours.")
            self._mon_eta.configure(text="")
            self._progress.configure(value=0)
        self.after(1000, self._refresh_run_monitor)

    def _monitor_record(self, running: bool, elapsed: float) -> None:
        """Suivi d'un run d'enregistrement (gardés/N depuis le CSV) + chrono/ETA."""
        raw = self._run_active_raw or self._raw_root.get()
        target = self._run_target or 0
        prog = self._read_run_progress(WORKSPACE / raw / "rapport_run_courant.csv")
        saved = prog["saved"] if prog else 0
        att = prog["attempts"] if prog else 0

        if running:
            self._set_run_state(ACCENT, "Enregistrement — en cours")
        elif target and saved >= target:
            self._set_run_state(OK_GREEN, "Enregistrement — terminé ✓")
        elif target and att > 0 and saved < target:
            self._set_run_state(WARN_YEL, "Enregistrement — arrêté (incomplet)")
        else:
            self._set_run_state(FG_DIM, "Prêt")

        if prog and att > 0:
            rate = 100 * prog["ok"] / att if att else 0
            if target > 0:
                self._progress.configure(mode="determinate", maximum=target,
                                         value=min(saved, target))
            detail = f"Gardés {saved}/{target or '?'} · essais {att} · ✓ {rate:.0f}%"
            if prog["causes"]:
                top = sorted(prog["causes"].items(), key=lambda kv: -kv[1])[:3]
                detail += "\n❌ " + " · ".join(f"{k} ×{v}" for k, v in top)
            self._mon_detail.configure(text=detail)
            # Chrono + ETA (ETA basé sur la cadence de GARDÉS)
            chrono = f"⏱ {self._hms(elapsed)}"
            if running and target and 0 < saved < target and prog["elapsed"] > 0:
                per = prog["elapsed"] / saved
                chrono += f" · ETA ~{self._hms(per * (target - saved))}"
            elif not running:
                chrono += " (fini)"
            self._mon_eta.configure(text=chrono)
        elif not running:
            self._mon_detail.configure(text="Aucun run en cours.")
            self._mon_eta.configure(
                text=f"⏱ {self._hms(elapsed)} (fini)" if elapsed else "")

    def _monitor_generic(self, running: bool, elapsed: float) -> None:
        """Suivi d'une tâche longue quelconque (conversion, entraînement) :
        barre de progression + chrono depuis le début + temps restant (ETA)."""
        label = self._task_label or "Tâche"
        frac = None
        if self._task_total > 0:
            frac = min(1.0, self._task_k / self._task_total)
        elif self._task_pct > 0:
            frac = min(1.0, self._task_pct / 100.0)

        if running:
            self._set_run_state(ACCENT, f"{label} — en cours")
        elif frac is not None and frac >= 0.999:
            self._set_run_state(OK_GREEN, f"{label} — terminé ✓")
        else:
            self._set_run_state(WARN_YEL, f"{label} — arrêté")

        # Barre + détail
        if frac is not None:
            self._progress.configure(mode="determinate", maximum=100, value=frac * 100)
            txt = f"{label} · {frac * 100:.0f}%"
            if self._task_total > 0:
                txt += f"  ({self._task_k}/{self._task_total})"
            self._mon_detail.configure(text=txt)
        else:
            self._progress.configure(mode="determinate", value=0)
            self._mon_detail.configure(text=f"{label} · en cours…")

        # Chrono depuis le début + ETA (temps restant)
        chrono = f"⏱ {self._hms(elapsed)}"
        if running and frac and frac > 0.02:
            chrono += f" · ETA ~{self._hms(elapsed * (1 - frac) / frac)}"
        elif not running:
            chrono += " (fini)"
        self._mon_eta.configure(text=chrono)

    def _browse_dataset(self) -> None:
        """Ouvre l'explorateur de fichiers pour choisir le dossier dataset.
        Stocke un chemin RELATIF au workspace si possible (plus lisible/portable)."""
        init = WORKSPACE / "datasets"
        init.mkdir(parents=True, exist_ok=True)
        chosen = filedialog.askdirectory(
            title="Choisir le dossier dataset (contenant episode_*/)",
            initialdir=str(init))
        if not chosen:
            return
        p = Path(chosen)
        try:
            rel = p.relative_to(WORKSPACE)
            self._raw_root.set(str(rel))
        except ValueError:
            self._raw_root.set(str(p))   # hors workspace → chemin absolu
        self._refresh_stats_now()
        self._status.configure(text=f"📁 Dataset : {self._raw_root.get()}")

    def _open_folder(self, path: Path) -> None:
        """Ouvre un dossier/fichier dans l'application par défaut (xdg-open)."""
        try:
            if path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["xdg-open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._status.configure(text=f"📂 Ouvert : {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ouverture impossible", f"{path}\n\n{exc}")

    def _open_raw_echecs(self) -> None:
        """Ouvre le dossier des épisodes échoués archivés (frère du dataset courant)."""
        raw = WORKSPACE / self._raw_root.get()
        fail = raw.parent / "raw_echecs"
        self._open_folder(fail)

    def _show_stats_popup(self) -> None:
        raw = WORKSPACE / self._raw_root.get()
        s = _dataset_stats(raw)
        msg = (f"Dossier : {raw}\n\n"
               f"Total épisodes : {s['total']}\n"
               f"Succès : {s['success']}\n"
               f"Échecs : {s['fail']}\n"
               f"Taux succès : {100 * s['success'] / s['total']:.0f}%"
               if s['total'] > 0 else
               f"Dossier : {raw}\n\nAucun épisode enregistré.")
        messagebox.showinfo("Stats dataset", msg)

    # ------------------------------------------------------------------
    #  Journal
    # ------------------------------------------------------------------
    def _drain_log(self) -> None:
        try:
            while True:
                line = self._q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        if not self._runner.running:
            txt = self._status.cget("text")
            if txt.startswith("En cours") or txt.startswith("🌙"):
                self._status.config(text="Prêt.")
            # Fin du run en mode nuit → on ré-affiche l'IHM pour voir le résultat.
            if self._night_active:
                self._night_active = False
                try:
                    self.deiconify()
                    self.lift()
                except tk.TclError:
                    pass
        self.after(100, self._drain_log)

    # Mots-clés de classification des lignes de journal (filtre + colorisation)
    _ERR_KW = ("error", "erreur", "❌", "échou", "echou", "timeout", "ratée", "ratee",
               "abandon", "⛔", "traceback", "exception", "refus", "figée", "figee")
    _OK_KW  = ("✓", "✅", "succès", "succes", "sauvé", "sauve", "gardé", "garde", " ok")

    @classmethod
    def _line_matches_filter(cls, line: str, mode: str) -> bool:
        if mode == "tout":
            return True
        low = line.lower()
        if mode == "erreurs":
            return any(k in low for k in cls._ERR_KW)
        if mode == "succes":
            return any(k in low for k in cls._OK_KW)
        return True

    @classmethod
    def _classify_line(cls, line: str) -> str:
        """Tag de colorisation du journal : err / ok / ihm / '' (normal)."""
        stripped = line.lstrip("\n")
        if stripped.startswith("[IHM]") or stripped.startswith("─"):
            return "ihm"
        low = line.lower()
        if any(k in low for k in cls._ERR_KW):
            return "err"
        if any(k in low for k in cls._OK_KW):
            return "ok"
        return ""

    # ── Compaction du journal ─────────────────────────────────────────
    # Le stdout ROS est verbeux : préfixes [INFO] [timestamp] [node] réécrits
    # courts, lignes de plomberie supprimées. Brut via Affichage → « verbeux ».
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    _ROS_RE = re.compile(
        r"^\[(INFO|WARN|ERROR|DEBUG|FATAL)\] \[[\d.]+\] \[([^\]]+)\]:?\s?(.*)$")
    _DROP_KW = ("process started with pid",
                "process has finished cleanly",
                "all log files can be found",
                "default logging verbosity",
                "signal_handler(signum=",
                "deprecation warning",
                "using degraded time")

    def _compact_line(self, line: str) -> str | None:
        """Réécrit une ligne de log en version utile, ou None pour la jeter."""
        line = self._ANSI_RE.sub("", line)
        if self._log_verbose.get():
            return line
        low = line.lower()
        if any(k in low for k in self._DROP_KW):
            return None
        # Bannière de lancement (ProcessRunner) : on garde le titre, pas la
        # commande bash géante (récupérable en mode verbeux).
        if "[IHM] ▶" in line and "[IHM] $" in line:
            mb = re.search(r"\[IHM\] ▶ (.+)", line)
            if mb:
                return f"\n─── ▶ {mb.group(1).strip()} ───\n"
        m = self._ROS_RE.match(line.strip())
        if m:
            lvl, node, msg = m.groups()
            if not msg.strip():
                return None
            node = node.split(".")[-1]
            if lvl in ("ERROR", "FATAL"):
                return f"⛔ {node} · {msg}\n"
            if lvl == "WARN":
                return f"⚠ {node} · {msg}\n"
            return f"{node} · {msg}\n"
        return line

    # Parsing de progression depuis le stdout (conversion, entraînement, …)
    _RE_FRAC_PAREN = re.compile(r"\((\d+)\s*/\s*(\d+)\)")   # ex. "(45/500)"
    _RE_FRAC = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")          # ex. "step 1000/20000"
    _RE_PCT = re.compile(r"(\d{1,3})\s*%")                   # ex. tqdm "38%"

    def _parse_task_progress(self, line: str) -> None:
        """Met à jour k/total/pct de la tâche courante par parsing du stdout.
        Ignoré pour les runs d'enregistrement (suivis via le CSV)."""
        if self._task_kind in ("", "record"):
            return
        m = self._RE_FRAC_PAREN.search(line) or self._RE_FRAC.search(line)
        if m:
            k, tot = int(m.group(1)), int(m.group(2))
            if tot > 0 and k <= tot:
                self._task_k, self._task_total = k, tot
                return
        m = self._RE_PCT.search(line)
        if m:
            p = int(m.group(1))
            if 0 <= p <= 100:
                self._task_pct = float(p)

    def _append_log(self, text: str) -> None:
        # Progression de tâche parsée sur le BRUT (avant toute compaction).
        self._parse_task_progress(text)
        compact = self._compact_line(text)
        if compact is None:
            return
        text = compact
        # Conserve TOUT dans le buffer (pour re-filtrer sans perdre l'historique).
        self._log_buffer.append(text)
        if len(self._log_buffer) > 5000:           # borne mémoire
            self._log_buffer = self._log_buffer[-4000:]
        if self._line_matches_filter(text, self._log_filter.get()):
            tag = self._classify_line(text)
            self._log.configure(state="normal")
            self._log.insert("end", text, tag or ())
            self._log.see("end")
            self._log.configure(state="disabled")

    def _apply_log_filter(self) -> None:
        """Re-rend le journal selon le filtre actif, depuis le buffer complet."""
        mode = self._log_filter.get()
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        for line in self._log_buffer:
            if self._line_matches_filter(line, mode):
                tag = self._classify_line(line)
                self._log.insert("end", line, tag or ())
        self._log.see("end")
        self._log.configure(state="disabled")

    def _copy_log(self) -> None:
        text = self._log.get("1.0", "end").strip()
        if not text:
            self._status.configure(text="Journal vide.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._status.configure(
            text=f"✓ {len(text.splitlines())} lignes copiées dans le presse-papier.")
        self.after(3000, lambda: self._status.configure(text="Prêt."))

    def _clear_log(self) -> None:
        self._log_buffer.clear()
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    # ------------------------------------------------------------------
    #  Runner
    # ------------------------------------------------------------------
    def _run_cmd(self, cmd: str, label: str, kind: str = "other") -> None:
        if self._runner.running:
            messagebox.showwarning("Occupé",
                                   "Un processus tourne déjà. Arrêtez-le d'abord.")
            return
        if self._runner.start(cmd, label):
            self._status.config(text=f"En cours : {label}")
            # Réinitialise le chrono + la progression de tâche (toutes tâches longues).
            self._task_kind = kind
            self._task_label = label
            self._task_start = time.time()
            self._task_elapsed = 0.0
            self._task_k = 0
            self._task_total = 0
            self._task_pct = 0.0

    def _stop(self) -> None:
        self._runner.stop()
        self._status.config(text="Arrêté.")

    # ------------------------------------------------------------------
    #  Plein écran & quitter
    # ------------------------------------------------------------------
    def _set_fullscreen(self, on: bool) -> None:
        self._fullscreen = bool(on)
        try:
            self._fs_var.set(self._fullscreen)
        except tk.TclError:
            pass
        try:
            self.attributes("-fullscreen", self._fullscreen)
        except tk.TclError:
            pass

    def _enter_fullscreen(self) -> None:
        self._set_fullscreen(True)

    def _toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self._fullscreen)

    def _quit(self) -> None:
        if self._runner.running:
            if not messagebox.askyesno("Quitter",
                                       "Un processus tourne. Arrêter et quitter ?"):
                return
            self._runner.stop()
        self._cam_feed.stop()
        self.destroy()

    # ------------------------------------------------------------------
    #  Tooltip
    # ------------------------------------------------------------------
    def _tooltip(self, widget: tk.Widget, text: str) -> None:
        tip: dict = {"win": None}

        def show(_e):
            if tip["win"] is not None:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 2
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(win, text=text, bg="#0b0f14", fg=FG, bd=1, relief="solid",
                     font=("DejaVu Sans", 10), padx=6, pady=3, justify="left").pack()
            tip["win"] = win

        def hide(_e):
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        # add="+" : cohabite avec les binds de survol (RButton, hover…)
        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")


# ============================================================================
def main() -> None:
    app = IhmVla()
    app.mainloop()


if __name__ == "__main__":
    main()
