#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IHM tactile — Bras robot IGUS Rebel (ROS2 Humble)
=================================================

Interface graphique Tkinter pensée pour un écran tactile Raspberry Pi 4
(mode kiosque, plein écran, portrait 720x1280).

Elle permet de piloter le robot « sans toucher un terminal » :
  * connexion robot (bring-up MoveIt CRI) en sous-processus,
  * lancement d'un mode pick&place / dance en sous-processus,
  * contrôle de la pince Schunk EGP25 via service ROS2,
  * pilotage articulaire (6 joints) et positions prédéfinies,
  * sauvegarde de positions personnalisées,
  * lecture en direct des positions articulaires (/joint_states),
  * monitoring RPi (CPU / RAM / température),
  * journal en direct, filtrable et exportable,
  * arrêt d'urgence flottant avec confirmation tactile (hold-to-stop).

PORTABLE : aucun chemin absolu codé en dur. Tous les chemins dérivent de
l'emplacement de ce script ou du répertoire utilisateur.
"""

import faulthandler
import html as html_lib
import json
import math
import os
import queue as _queue
import re
import shlex
import signal
import smtplib
import ssl
import statistics
import subprocess
import sys
import threading
import time as _time
import traceback
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from datetime import datetime, date
from email.message import EmailMessage
from pathlib import Path
from threading import Lock, Thread

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool, Trigger

# Services MoveIt (TRAC-IK) — imports différés si absent (graceful fallback)
try:
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.msg import RobotState, PositionIKRequest
    _HAS_MOVEIT_MSGS = True
except ImportError:
    GetPositionIK = None
    RobotState = None
    PositionIKRequest = None
    _HAS_MOVEIT_MSGS = False

from anomaly_detector import AnomalyDetector, N_NOMINAL as ANOMALIE_N_NOMINAL


# ============================================================================
#  1. PORTABILITÉ — détection des chemins depuis l'emplacement du script
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../ihm
WORKSPACE  = SCRIPT_DIR.parent                        # racine du workspace
ACTIVATE   = WORKSPACE / "activate.bash"               # script de sourcing

# Stockage utilisateur (positions perso, exports, stats, rapports)
# Placé sur le Bureau pour rester visible facilement (sinon ~/.ihm_robot caché).
def _dossier_bureau():
    for nom in ("Bureau", "Desktop"):
        d = Path.home() / nom
        if d.is_dir():
            return d
    return Path.home()

USER_DIR        = _dossier_bureau() / "ihm_robot"
POSITIONS_FILE  = USER_DIR / "positions.json"
EXPORTS_DIR     = USER_DIR / "journaux"
STATS_FILE      = USER_DIR / "stats.json"
RAPPORTS_DIR    = USER_DIR / "rapports"
EMAIL_CONFIG    = USER_DIR / "email_config.json"
LIMITS_FILE     = USER_DIR / "joint_limits.json"
ZONES_FILE      = USER_DIR / "zones.json"
CATALOGUE_FILE  = USER_DIR / "objets_detectables.json"
IA_HISTORIQUE   = USER_DIR / "ia_historique.json"
LAST_JOINTS_FILE = USER_DIR / "last_joints.json"  # mémoire des derniers angles

# ----------------------------------------------------------------------------
#  JOURNAL DE CRASH (live)
# ----------------------------------------------------------------------------
# Tout le journal de l'IHM est écrit au fil de l'eau dans un fichier
# line-buffered, et les exceptions non gérées (thread principal + threads
# Python + signaux fatals C type segfault) sont capturées avec leur trace.
# Objectif : si l'IHM se ferme brutalement, on a quand même les dernières
# secondes du journal et la cause sur disque.
try:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _CRASH_LOG_PATH = EXPORTS_DIR / (
        "journal_live_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    _CRASH_LOG_FILE = open(_CRASH_LOG_PATH, "w", buffering=1)
except Exception:
    _CRASH_LOG_FILE = None
    _CRASH_LOG_PATH = None

_CRASH_LOG_LOCK = Lock()


def _crash_log_write(ligne):
    """Écrit une ligne dans le journal-live (thread-safe, jamais lève)."""
    if _CRASH_LOG_FILE is None:
        return
    try:
        with _CRASH_LOG_LOCK:
            _CRASH_LOG_FILE.write(ligne + "\n")
    except Exception:
        pass


if _CRASH_LOG_FILE is not None:
    # Dump une stack C en cas de SIGSEGV / SIGABRT / SIGFPE / SIGBUS / SIGILL —
    # essentiel pour les crashs de Tk, librealsense, librealsense, rclpy C++.
    try:
        faulthandler.enable(file=_CRASH_LOG_FILE)
    except Exception:
        pass

    _crash_log_write("=== Démarrage IHM %s ===" %
                     datetime.now().isoformat(timespec="seconds"))


def _excepthook_main(exc_type, exc_value, tb):
    _crash_log_write("=== EXCEPTION NON GÉRÉE (thread principal) ===")
    for entry in traceback.format_exception(exc_type, exc_value, tb):
        for ligne in entry.rstrip().splitlines():
            _crash_log_write(ligne)
    _crash_log_write("=== FIN EXCEPTION ===")
    sys.__excepthook__(exc_type, exc_value, tb)


def _excepthook_thread(args):
    name = getattr(args.thread, "name", "?")
    _crash_log_write("=== EXCEPTION THREAD %s ===" % name)
    for entry in traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback):
        for ligne in entry.rstrip().splitlines():
            _crash_log_write(ligne)
    _crash_log_write("=== FIN EXCEPTION THREAD ===")


sys.excepthook = _excepthook_main
threading.excepthook = _excepthook_thread


# Assistant IA — paramètres Ollama (HTTP API locale)
OLLAMA_URL          = "http://localhost:11434/api/chat"
OLLAMA_MODELE       = "llama3.2:3b"
OLLAMA_TIMEOUT_S    = 60

# Topics et services ROS2 pour l'Assistant IA
TOPIC_OBJET_ROBOT   = "/object_position_in_robot"
SERVICE_IK          = "/compute_ik"
GROUPE_PLANIF       = "rebel_arm"     # nom du planning group MoveIt (à ajuster)
TIP_LINK            = "link6"          # link du TCP (à ajuster selon URDF)

# Objet par défaut (modifiable dans Catalogue Objets)
CATALOGUE_DEFAUT = {
    "roulette verte": {
        "classe_yolo": "roulette",
        "couleur": "verte",
        "description": "Roulette de mobilier industriel verte (~5 cm)"
    }
}
SEQUENCES_FILE  = USER_DIR / "sequences.json"
ANOMALIE_DIR    = USER_DIR / "anomalie"   # autoencoder LSTM (profils, modèle)

# Watchdog /joint_states : seuils d'alerte (secondes sans message)
WATCHDOG_WARN_S = 2.0
WATCHDOG_CRIT_S = 5.0

# Courbes temps réel
COURBES_N_SAMPLES = 150     # ~30 s à 5 Hz
COURBES_HAUTEUR   = 140
COULEURS_JOINTS = ["#1976d2", "#2e7d32", "#ef6c00",
                   "#c62828", "#7b1fa2", "#00838f"]

# Limites articulaires par défaut (degrés) — large par défaut
LIMITES_DEFAUT = {"joint%d" % i: (-180.0, 180.0) for i in range(1, 7)}

# ---- Commande vocale (Vosk hors ligne) -------------------------------- #
# Le modèle FR doit être extrait dans ~/Bureau/ihm_robot/vosk-model-fr/
VOSK_MODEL_DIR = USER_DIR / "vosk-model-fr"
VOSK_SAMPLE_RATE = 16000
VOSK_BLOCKSIZE   = 4000  # ~250 ms à 16 kHz

# Mots d'urgence : déclenchent l'arrêt immédiat sans phrase de réveil.
# Garde anti-faux-positif : ne déclenche que si la phrase reconnue est
# courte (≤ 2 mots) ou contient "urgence" qui est moins ambigu.
VOCAL_URGENCE_KEYWORDS = ("urgence", "stop", "arrêt", "arrete", "arrête",
                           "stoppe", "halte")

# Commandes normales : phrases précises pour limiter les faux positifs.
# Ordre = priorité de match (premier qui matche gagne).
VOCAL_COMMANDES = [
    # (action, [phrases à chercher en sous-chaîne, en minuscules])
    ("ouvre_pince",     ["ouvre la pince", "ouvre pince",
                          "ouvrir la pince", "ouverture pince"]),
    ("ferme_pince",     ["ferme la pince", "ferme pince",
                          "fermer la pince", "fermeture pince"]),
    ("home",            ["position initiale", "position home",
                          "retour home", "retour à la maison",
                          "va à la position initiale"]),
    ("connecter",       ["connecte le robot", "connecter le robot",
                          "connexion robot"]),
    ("deconnecter",     ["déconnecte le robot", "deconnecte le robot",
                          "déconnecter le robot"]),
    ("lancer_mode",     ["lance le mode", "lancer le mode",
                          "démarre le mode", "demarre le mode"]),
    ("pause_mode",      ["arrête le mode", "arrete le mode",
                          "stoppe le mode", "pause mode"]),
    ("ouvre_pince",     ["ouvre"]),   # repli court
    ("ferme_pince",     ["ferme"]),   # repli court
]

# Identité affichée dans les rapports
AUTEUR_NOM       = "Daniel BAL"
AUTEUR_FORMATION = "M1 MESI"
AUTEUR_ECOLE     = "Université de Strasbourg"
AUTEUR_PROJET    = "Bras robot IGUS Rebel — Pilotage tactile ROS2"

# Seuils empiriques pour l'évaluation de la fatigue mécanique
SEUIL_FATIGUE_JOINT_SURV   = 10_000   # nb mouvements de joint
SEUIL_FATIGUE_JOINT_CRIT   = 50_000
SEUIL_FATIGUE_PINCE_SURV   = 5_000    # cycles d'ouverture/fermeture
SEUIL_FATIGUE_PINCE_CRIT   = 20_000
SEUIL_REPETABILITE_OK_DEG  = 0.5      # écart-type max acceptable


# ============================================================================
#  2. CONSTANTES — style et configuration
# ============================================================================
INC_DEG_DEFAUT   = 10.0   # incrément par défaut des boutons +/- des joints
VITESSE_DEFAUT   = 5      # durée par défaut d'une trajectoire (secondes)
HOLD_DEL_MS      = 1000   # durée du long-press pour supprimer une position

# Palette de couleurs (identité visuelle de l'IHM existante)
FOND      = "#f4f6f9"   # fond général
CARD      = "#ffffff"   # cartes blanches
BORDURE   = "#dde1ea"   # bordure des cartes
BLEU      = "#1976d2"   # bleu d'action
VERT      = "#2e7d32"   # vert (succès / ouvrir)
VERT_BG   = "#e8f5e9"   # vert clair (fond)
ROUGE     = "#c62828"   # rouge (arrêt / fermer)
ROUGE_BG  = "#ffebee"   # rouge clair (fond)
GRIS      = "#6b7280"   # gris (texte secondaire / inactif)
ORANGE    = "#ef6c00"   # orange (connexion en cours / simu)
JAUNE     = "#f9a825"   # jaune (warning / hold)
TEXTE     = "#1f2937"   # texte principal (ardoise foncée)


def _assombrir(couleur, facteur):
    """Assombrit une couleur '#rrggbb' (facteur < 1). Renvoie la couleur
    inchangée si le format n'est pas un hex 7 caractères (nom Tk, etc.)."""
    try:
        if not (isinstance(couleur, str) and len(couleur) == 7
                and couleur.startswith('#')):
            return couleur
        r = max(0, min(255, int(int(couleur[1:3], 16) * facteur)))
        g = max(0, min(255, int(int(couleur[3:5], 16) * facteur)))
        b = max(0, min(255, int(int(couleur[5:7], 16) * facteur)))
        return "#%02x%02x%02x" % (r, g, b)
    except ValueError:
        return couleur

POLICE = "monospace"

# Rayon de travail max du bras dans le plan (m) : contrainte √(x²+y²) ≤ RAYON_MAX.
# Sert au calculateur de portée (section « Portée XY ») pour donner le y maxi
# atteignable pour un x donné.
RAYON_MAX = 0.60

# Modes disponibles : libellé -> commande ros2
MODES_PHYSIQUE = {
    "Pick & Place IA (caméra YOLO)": "ros2 launch mon_controleur pick_place_complet.launch.py",
    "Pick & Place cartésien":        "ros2 launch mon_controleur pick_place_cartesien.launch.py",
    "Pick & Place optimisé":         "ros2 launch mon_controleur pick_place_optim.launch.py",
    "Robot Dance":                   "ros2 launch mon_controleur robot_dance.launch.py",
}
MODES_SIMU = {
    "Pick & Place IA (caméra YOLO)": "ros2 launch mon_controleur sim_pick_place.launch.py",
    "Pick & Place cartésien":        "ros2 launch mon_controleur sim_pick_place_cartesien.launch.py",
    "Pick & Place optimisé":         "ros2 launch mon_controleur sim_pick_place_optim.launch.py",
    "Robot Dance":                   "ros2 launch mon_controleur sim_robot_dance.launch.py",
}
MODES = MODES_PHYSIQUE
MODE_DANCE = "Robot Dance"
CHOREGRAPHIES = ["all", "shapes", "lemniscate", "joint_dance", "parametric"]

# Détection d'un cycle pick&place terminé dans la sortie d'un launch.
# On reste large : "cycle terminé", "pick&place OK", "place done", etc.
RE_CYCLE = re.compile(
    r'(cycle\s+(termin|complet|done|fini|ok)|'
    r'(pick.?place|place)\s+(termin|complet|done|fini|ok))',
    re.IGNORECASE)

# Détection des lignes "erreur" pour le filtre du journal.
RE_ERREUR = re.compile(
    r'(erreur|error|\bfail|échec|echec|\[ERROR\]|\bWARN\b|warning)',
    re.IGNORECASE)

# Détection d'un cycle échoué (avant abandon / recovery).
RE_ECHEC_CYCLE = re.compile(
    r'(cycle\s+(echec|échec|fail|failed|abort|abandon)|'
    r'(pick|place)\s+(failed|abort|impossible|introuvable|non\s+atteinte))',
    re.IGNORECASE)

# Détection d'un home recovery (retour position sûre après incident).
RE_HOME_RECOVERY = re.compile(
    r'(home.{0,15}recovery|recovery.{0,15}home|retour.{0,15}home|'
    r'going\s+home|recovery\s+pose)',
    re.IGNORECASE)


# ============================================================================
#  3. NŒUD ROS2
# ============================================================================
class RobotController(Node):
    """Nœud ROS2 de l'IHM : trajectoires, pince, et lecture /joint_states."""

    def __init__(self):
        super().__init__('ihm_robot_node')
        self.pub = self.create_publisher(
            JointTrajectory, '/rebel_arm_controller/joint_trajectory', 10)
        self.gripper_client = self.create_client(SetBool, '/gripper/command')
        self.start_client = self.create_client(Trigger, '/start_robot')
        self.stop_client  = self.create_client(Trigger, '/stop_robot')

        # Lecture en direct des positions articulaires
        self._lock = Lock()
        self.last_positions = None  # list[6] en radians, ou None
        self.last_stamp = 0.0
        self.last_wall = 0.0        # _time.monotonic() lors du dernier message
        self._last_save_wall = 0.0  # throttle pour la persistance sur disque
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(JointState, '/joint_states',
                                 self._on_joint_state, qos)

        # Charge les derniers angles sauvegardés sur disque (si présents)
        # → RViz/IHM affichent une position plausible avant tout message live
        self._charger_derniers_angles()

        # Détection d'objets (perception YOLO → coordonnées dans le repère robot)
        self._lock_obj = Lock()
        self.last_object = None     # (x, y, z) en mètres, ou None
        self.last_object_wall = 0.0
        self.create_subscription(PointStamped, TOPIC_OBJET_ROBOT,
                                 self._on_object, 10)

        # Client TRAC-IK (résolution position cartésienne → angles)
        self.ik_client = None
        if _HAS_MOVEIT_MSGS:
            self.ik_client = self.create_client(GetPositionIK, SERVICE_IK)

    def _on_joint_state(self, msg):
        """Callback /joint_states — stocke les positions joint1..joint6."""
        try:
            ordre = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            mapping = dict(zip(msg.name, msg.position))
            positions = [mapping[n] for n in ordre if n in mapping]
            if len(positions) == 6:
                now = _time.monotonic()
                with self._lock:
                    self.last_positions = positions
                    self.last_stamp = self.get_clock().now().nanoseconds * 1e-9
                    self.last_wall = now
                # Persistance throttle : on écrit au plus une fois toutes les 2s
                if now - self._last_save_wall >= 2.0:
                    self._last_save_wall = now
                    self._sauver_derniers_angles(positions)
        except Exception:
            pass

    def _sauver_derniers_angles(self, positions_rad=None):
        """Sauvegarde les derniers angles articulaires sur disque."""
        try:
            if positions_rad is None:
                with self._lock:
                    positions_rad = self.last_positions
            if positions_rad is None or len(positions_rad) != 6:
                return
            USER_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "positions_rad": [float(p) for p in positions_rad],
                "ts": datetime.now().isoformat(timespec='seconds'),
            }
            tmp = LAST_JOINTS_FILE.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump(data, f)
            tmp.replace(LAST_JOINTS_FILE)
        except Exception:
            pass

    def _charger_derniers_angles(self):
        """Charge les derniers angles sauvegardés (pour amorcer last_positions)."""
        try:
            if not LAST_JOINTS_FILE.exists():
                return
            with open(LAST_JOINTS_FILE, 'r') as f:
                data = json.load(f)
            positions = data.get("positions_rad")
            if positions and len(positions) == 6:
                with self._lock:
                    self.last_positions = [float(p) for p in positions]
                    # last_wall reste à 0 → considéré comme "jamais reçu en live"
        except Exception:
            pass

    def get_positions_deg(self):
        """Retourne les 6 positions en degrés, ou None si jamais reçu."""
        with self._lock:
            if self.last_positions is None:
                return None
            return [math.degrees(p) for p in self.last_positions]

    def get_age_s(self):
        """Renvoie l'âge en secondes du dernier message (None si jamais reçu)."""
        with self._lock:
            if self.last_wall <= 0:
                return None
            return _time.monotonic() - self.last_wall

    def _on_object(self, msg):
        """Callback /object_position_in_robot — stocke (x, y, z) en mètres."""
        try:
            with self._lock_obj:
                self.last_object = (float(msg.point.x),
                                     float(msg.point.y),
                                     float(msg.point.z))
                self.last_object_wall = _time.monotonic()
        except Exception:
            pass

    def get_last_object(self, max_age_s=10.0):
        """Renvoie la dernière position d'objet détectée si récente, ou None."""
        with self._lock_obj:
            if self.last_object is None:
                return None
            if _time.monotonic() - self.last_object_wall > max_age_s:
                return None
            return tuple(self.last_object)


# Initialisation ROS2 + spin dans un thread démon
rclpy.init()
robot_node = RobotController()
Thread(target=lambda: rclpy.spin(robot_node), daemon=True).start()


def envoyer_trajectoire(angles_deg, temps=VITESSE_DEFAUT):
    """Publie une trajectoire articulaire (angles en degrés)."""
    msg = JointTrajectory()
    msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    pt = JointTrajectoryPoint()
    pt.positions = [math.radians(float(a)) for a in angles_deg]
    pt.time_from_start.sec = int(max(1, temps))
    msg.points.append(pt)
    robot_node.pub.publish(msg)


# ============================================================================
#  4. UTILITAIRES MONITORING (CPU / RAM / TEMP RPi)
# ============================================================================
class SysMonitor:
    """Lit CPU / RAM / température sans dépendance externe (procfs only)."""

    def __init__(self):
        self._last_cpu = None  # (idle, total)

    def lire(self):
        return {
            "cpu":  self._lire_cpu(),
            "ram":  self._lire_ram(),
            "temp": self._lire_temp(),
        }

    def _lire_cpu(self):
        try:
            with open('/proc/stat', 'r') as f:
                fields = f.readline().split()
            vals = [int(v) for v in fields[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            if self._last_cpu is None:
                self._last_cpu = (idle, total)
                return None
            d_idle = idle - self._last_cpu[0]
            d_total = total - self._last_cpu[1]
            self._last_cpu = (idle, total)
            if d_total <= 0:
                return None
            return 100.0 * (1.0 - d_idle / d_total)
        except Exception:
            return None

    def _lire_ram(self):
        try:
            total = avail = None
            with open('/proc/meminfo', 'r') as f:
                for ligne in f:
                    if ligne.startswith('MemTotal:'):
                        total = int(ligne.split()[1])
                    elif ligne.startswith('MemAvailable:'):
                        avail = int(ligne.split()[1])
                    if total and avail:
                        break
            if total and avail:
                return 100.0 * (1.0 - avail / total)
        except Exception:
            return None
        return None

    def _lire_temp(self):
        for path in ('/sys/class/thermal/thermal_zone0/temp',
                     '/sys/class/hwmon/hwmon0/temp1_input'):
            try:
                with open(path, 'r') as f:
                    return int(f.read().strip()) / 1000.0
            except Exception:
                continue
        return None


# ============================================================================
#  4b. COMMANDE VOCALE (Vosk, hors ligne)
# ============================================================================
class VoiceListener:
    """Écoute audio + reconnaissance hors ligne via Vosk.

    Les imports vosk / sounddevice sont différés pour que l'IHM puisse
    démarrer sans ces dépendances. `disponible()` renvoie un message clair
    expliquant ce qui manque.
    """

    def __init__(self, on_texte, model_dir=VOSK_MODEL_DIR):
        self._on_texte = on_texte          # callback(texte: str)
        self.model_dir = Path(model_dir)
        self._actif = False
        self._thread = None
        self._stop_event = False

    def disponible(self):
        """Renvoie (ok: bool, message: str)."""
        try:
            import vosk          # noqa: F401
        except ImportError:
            return False, ("Module « vosk » introuvable.\n"
                           "Installez-le : pip install vosk")
        try:
            import sounddevice   # noqa: F401
        except ImportError:
            return False, ("Module « sounddevice » introuvable.\n"
                           "Installez-le : pip install sounddevice\n"
                           "(et : sudo apt install libportaudio2)")
        if not self.model_dir.exists():
            return False, ("Modèle Vosk introuvable :\n%s\n\n"
                           "Téléchargez « vosk-model-small-fr » depuis\n"
                           "https://alphacephei.com/vosk/models\n"
                           "puis extrayez-le ici (renommez le dossier en\n"
                           "« vosk-model-fr »)."
                           % self.model_dir)
        return True, "OK"

    def est_actif(self):
        return self._actif

    def demarrer(self):
        ok, msg = self.disponible()
        if not ok:
            return False, msg
        if self._actif:
            return True, "Déjà actif"
        self._stop_event = False
        self._actif = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True, "Écoute démarrée"

    def arreter(self):
        self._stop_event = True
        self._actif = False

    def _loop(self):
        try:
            import vosk
            import sounddevice as sd
        except ImportError:
            self._actif = False
            return

        # Coupe les logs verbeux de Vosk
        try:
            vosk.SetLogLevel(-1)
        except Exception:
            pass

        try:
            model = vosk.Model(str(self.model_dir))
            rec = vosk.KaldiRecognizer(model, VOSK_SAMPLE_RATE)
        except Exception as exc:
            self._on_texte("__erreur__ chargement modèle : %s" % exc)
            self._actif = False
            return

        q = _queue.Queue()

        def _cb(indata, _frames, _t, status):
            if status:
                pass  # ignore underruns
            q.put(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=VOSK_SAMPLE_RATE,
                                    blocksize=VOSK_BLOCKSIZE,
                                    dtype='int16', channels=1,
                                    callback=_cb):
                while not self._stop_event:
                    try:
                        data = q.get(timeout=0.3)
                    except _queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        try:
                            res = json.loads(rec.Result())
                        except Exception:
                            continue
                        texte = (res.get("text") or "").strip().lower()
                        if texte:
                            self._on_texte(texte)
        except Exception as exc:
            self._on_texte("__erreur__ flux audio : %s" % exc)
        finally:
            self._actif = False


# ============================================================================
#  4c. ASSISTANT IA — Zones, Catalogue d'objets, Planner LLM
# ============================================================================
class ZoneManager:
    """Gère les zones de travail nommées (x, y, z) dans le repère robot.

    Persistées dans ~/Bureau/ihm_robot/zones.json.
    """

    def __init__(self):
        self.zones = self._charger()

    def _charger(self):
        try:
            if ZONES_FILE.exists():
                with open(ZONES_FILE, 'r') as f:
                    data = json.load(f)
                out = {}
                for nom, v in data.items():
                    if (isinstance(nom, str) and isinstance(v, dict)
                            and all(k in v for k in ("x", "y", "z"))):
                        out[nom] = {
                            "x": float(v["x"]),
                            "y": float(v["y"]),
                            "z": float(v["z"]),
                            "description": v.get("description", ""),
                        }
                return out
        except Exception:
            pass
        return {}

    def _sauver(self):
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(ZONES_FILE, 'w') as f:
                json.dump(self.zones, f, indent=2)
        except Exception:
            pass

    def ajouter(self, nom, x, y, z, description=""):
        self.zones[nom] = {"x": float(x), "y": float(y),
                            "z": float(z), "description": description}
        self._sauver()

    def supprimer(self, nom):
        if nom in self.zones:
            del self.zones[nom]
            self._sauver()

    def get(self, nom):
        return self.zones.get(nom)

    def lister(self):
        return sorted(self.zones.keys())


class CatalogueObjets:
    """Catalogue des objets que la perception YOLO sait détecter.

    Sert au LLM pour savoir ce qu'il peut demander de saisir. Persistée
    dans ~/Bureau/ihm_robot/objets_detectables.json.
    """

    def __init__(self):
        self.objets = self._charger()

    def _charger(self):
        try:
            if CATALOGUE_FILE.exists():
                with open(CATALOGUE_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        # Première utilisation : on écrit le catalogue par défaut
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(CATALOGUE_FILE, 'w') as f:
                json.dump(CATALOGUE_DEFAUT, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return dict(CATALOGUE_DEFAUT)

    def _sauver(self):
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(CATALOGUE_FILE, 'w') as f:
                json.dump(self.objets, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def ajouter(self, nom, classe_yolo, couleur="", description=""):
        self.objets[nom] = {
            "classe_yolo": classe_yolo,
            "couleur": couleur,
            "description": description,
        }
        self._sauver()

    def supprimer(self, nom):
        if nom in self.objets:
            del self.objets[nom]
            self._sauver()

    def lister(self):
        return sorted(self.objets.keys())


class LLMPlanner:
    """Communication avec Ollama (LLM local) via HTTP.

    Rôle volontairement minimal : à partir d'une commande en français,
    le LLM extrait seulement deux informations — l'objet à saisir
    (« cible ») et la zone de dépôt (« place »). L'IHM les traduit
    ensuite en arguments d'un launch pick & place déjà déterminé.
    Remplir deux champs est une tâche fiable pour un modèle 3B, là où
    composer une séquence d'actions ne l'était pas.
    """

    def __init__(self, zones: ZoneManager, catalogue: CatalogueObjets,
                  url=OLLAMA_URL, modele=OLLAMA_MODELE):
        self.zones = zones
        self.catalogue = catalogue
        self.url = url
        self.modele = modele

    def disponible(self):
        """Renvoie (ok, message) — teste la présence d'Ollama."""
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            modeles = [m.get("name", "") for m in data.get("models", [])]
            if not modeles:
                return False, "Ollama tourne mais aucun modèle installé."
            if not any(self.modele in m for m in modeles):
                return (False,
                        "Modèle « %s » introuvable.\n"
                        "Installez-le : ollama pull %s\n"
                        "Modèles présents : %s"
                        % (self.modele, self.modele, ", ".join(modeles)))
            return True, "OK (modèle : %s)" % self.modele
        except urllib.error.URLError:
            return False, ("Ollama injoignable sur localhost:11434.\n"
                           "Installez Ollama (curl -fsSL "
                           "https://ollama.com/install.sh | sh)\n"
                           "puis : ollama pull %s" % self.modele)
        except Exception as exc:
            return False, "Erreur Ollama : %s" % exc

    def _prompt_systeme(self):
        """Construit le prompt système : extraction cible + place."""
        objets = self.catalogue.lister()
        objets_lib = ", ".join("« %s »" % o for o in objets) or "(aucun)"
        zones_noms = self.zones.lister()
        zones_lib = ", ".join("« %s »" % z for z in zones_noms) or "(aucune)"

        obj_ex  = objets[0] if objets else "objet"
        zone_ex = zones_noms[0] if zones_noms else "zone"

        return (
            "Tu assistes un bras robot qui fait du pick & place : il SAISIT "
            "un objet repéré par caméra et le DÉPOSE dans une zone.\n"
            "Tu reçois une commande en français et tu réponds UNIQUEMENT par "
            "du JSON valide. Aucun texte hors JSON.\n"
            "\n"
            "Ton seul travail : identifier deux choses dans la commande —\n"
            "  • cible : l'objet à saisir\n"
            "  • place : la zone où le déposer\n"
            "\n"
            "== OBJETS SAISISSABLES (valeurs autorisées pour \"cible\") ==\n"
            "  " + objets_lib + "\n"
            "\n"
            "== ZONES DE DÉPÔT (valeurs autorisées pour \"place\") ==\n"
            "  " + zones_lib + "\n"
            "\n"
            "== RÈGLES ==\n"
            "1. \"cible\" doit être EXACTEMENT un des objets ci-dessus, "
            "recopié à l'identique.\n"
            "2. \"place\" doit être EXACTEMENT une des zones ci-dessus, "
            "recopiée à l'identique.\n"
            "3. Si l'objet demandé n'est pas dans la liste, mets \"cible\":\"\".\n"
            "4. Si aucune zone n'est mentionnée ou reconnue, mets "
            "\"place\":\"\".\n"
            "5. N'invente jamais de nom.\n"
            "\n"
            "== EXEMPLE ==\n"
            "commande : « attrape " + obj_ex + " et pose-la dans "
            + zone_ex + " »\n"
            "{\"cible\":\"" + obj_ex + "\",\"place\":\"" + zone_ex + "\","
            "\"comprehension\":\"Saisir " + obj_ex + " et le déposer dans "
            + zone_ex + ".\"}\n"
            "\n"
            "== FORMAT DE TA RÉPONSE (JSON STRICT, RIEN D'AUTRE) ==\n"
            "{\"cible\":\"<objet>\",\"place\":\"<zone>\","
            "\"comprehension\":\"<phrase>\"}\n"
        )

    def planifier(self, commande):
        """Envoie la commande au LLM et renvoie (extraction, brut_str).

        extraction = {"cible": str, "place": str, "comprehension": str}.
        Lève une exception en cas d'erreur HTTP ou JSON invalide.
        """
        prompt = self._prompt_systeme()
        payload = {
            "model": self.modele,
            "stream": False,
            "format": "json",     # force la sortie JSON valide
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",   "content": commande},
            ],
        }
        req = urllib.request.Request(
            self.url, method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        contenu = data.get("message", {}).get("content", "")
        if not contenu:
            raise RuntimeError("Réponse vide d'Ollama")
        # Parse le JSON renvoyé par le LLM
        try:
            extraction = json.loads(contenu)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "JSON invalide du LLM : %s\nContenu : %s"
                % (exc, contenu[:300]))
        if not isinstance(extraction, dict):
            raise RuntimeError("Le LLM n'a pas renvoyé un objet JSON.")
        return extraction, contenu


# ============================================================================
#  5. STATISTIQUES PERSISTANTES
# ============================================================================
class StatsManager:
    """Gère toutes les statistiques d'usage (cycles, fatigue, erreurs, etc.).

    Persistées dans ~/Bureau/ihm_robot/stats.json. Les modifications marquent
    un drapeau « sale » ; l'écriture disque réelle est différée à flush()
    (boucle sysmon 2,5 s + fermeture) pour éviter un json.dump synchrone à
    chaque cycle/mouvement (latence, usure carte SD).
    """

    def __init__(self):
        self._lock = Lock()
        self._dirty = False
        self.data = self._charger()

    # -- chargement / sauvegarde --------------------------------------- #
    def _modele_vide(self):
        return {
            "premier_demarrage": datetime.now().isoformat(timespec='seconds'),
            "temps_fonctionnement_s": 0,
            "session_demarree_iso": None,
            "cycles": {
                "total":       0,
                "echecs":      0,
                "recoveries":  0,
                "par_jour":    {},   # "YYYY-MM-DD" -> int
            },
            "modes": {},             # libellé -> {lancements, duree_s, cycles}
            "joints": {
                "joint%d" % i: {
                    "min_deg": None, "max_deg": None,
                    "mouvements": 0,
                } for i in range(1, 7)
            },
            "pince": {"ouvertures": 0, "fermetures": 0},
            "erreurs": [],           # [{ts, message, contexte}]
            "repetabilite": [],      # [{ts, n, ecarts_std_deg, ...}]
        }

    def _charger(self):
        try:
            if STATS_FILE.exists():
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                # Migration douce : on garantit la présence de toutes les clés
                modele = self._modele_vide()
                for k, v in modele.items():
                    data.setdefault(k, v)
                for k in modele["cycles"]:
                    data["cycles"].setdefault(k, modele["cycles"][k])
                return data
        except Exception:
            pass
        return self._modele_vide()

    def _sauver(self):
        # Appelé sous self._lock : marque seulement ; l'écriture réelle
        # a lieu dans flush() (jamais sous le verrou d'un appelant).
        self._dirty = True

    def flush(self, force=False):
        """Écrit stats.json si des modifications sont en attente."""
        with self._lock:
            if not (self._dirty or force):
                return
            self._dirty = False
            contenu = json.dumps(self.data, indent=2)
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(STATS_FILE, 'w') as f:
                f.write(contenu)
        except Exception:
            pass

    # -- sessions (temps de fonctionnement cumulé) --------------------- #
    def session_start(self):
        with self._lock:
            self.data["session_demarree_iso"] = (
                datetime.now().isoformat(timespec='seconds'))
            self._sauver()
        # Début de session persisté tout de suite (le compteur de temps de
        # fonctionnement repose sur cet horodatage en cas de crash).
        self.flush(force=True)

    def session_stop(self):
        with self._lock:
            iso = self.data.get("session_demarree_iso")
            if iso:
                try:
                    t0 = datetime.fromisoformat(iso)
                    dt = (datetime.now() - t0).total_seconds()
                    if dt > 0:
                        self.data["temps_fonctionnement_s"] += int(dt)
                except Exception:
                    pass
            self.data["session_demarree_iso"] = None
            self._sauver()
        # Fin de session = écriture immédiate garantie (fermeture d'IHM)
        self.flush(force=True)

    def temps_total_s(self):
        """Temps cumulé + temps de la session courante si en cours."""
        with self._lock:
            base = self.data.get("temps_fonctionnement_s", 0)
            iso = self.data.get("session_demarree_iso")
            if iso:
                try:
                    t0 = datetime.fromisoformat(iso)
                    base += int((datetime.now() - t0).total_seconds())
                except Exception:
                    pass
            return base

    # -- cycles --------------------------------------------------------- #
    def cycle_ok(self):
        with self._lock:
            self.data["cycles"]["total"] += 1
            jour = date.today().isoformat()
            d = self.data["cycles"]["par_jour"]
            d[jour] = d.get(jour, 0) + 1
            self._sauver()

    def cycle_echec(self):
        with self._lock:
            self.data["cycles"]["echecs"] += 1
            self._sauver()

    def home_recovery(self):
        with self._lock:
            self.data["cycles"]["recoveries"] += 1
            self._sauver()

    def cycles_jour(self, jour_iso=None):
        with self._lock:
            jour = jour_iso or date.today().isoformat()
            return self.data["cycles"]["par_jour"].get(jour, 0)

    # -- modes ---------------------------------------------------------- #
    def mode_start(self, libelle):
        with self._lock:
            m = self.data["modes"].setdefault(
                libelle, {"lancements": 0, "duree_s": 0, "cycles": 0})
            m["lancements"] += 1
            self._sauver()

    def mode_end(self, libelle, duree_s, cycles_session):
        with self._lock:
            m = self.data["modes"].setdefault(
                libelle, {"lancements": 0, "duree_s": 0, "cycles": 0})
            m["duree_s"] += max(0, int(duree_s))
            m["cycles"] += max(0, int(cycles_session))
            self._sauver()

    # -- joints / pince ------------------------------------------------- #
    def tracker_joints(self, positions_deg):
        """Met à jour min/max et incrémente le compteur de mouvements."""
        if positions_deg is None or len(positions_deg) != 6:
            return
        with self._lock:
            for i, p in enumerate(positions_deg, start=1):
                key = "joint%d" % i
                j = self.data["joints"][key]
                if j["min_deg"] is None or p < j["min_deg"]:
                    j["min_deg"] = float(p)
                if j["max_deg"] is None or p > j["max_deg"]:
                    j["max_deg"] = float(p)

    def joint_mouvement(self):
        """Incrémente le compteur de mouvements pour les 6 joints (1 ordre)."""
        with self._lock:
            for k in self.data["joints"]:
                self.data["joints"][k]["mouvements"] += 1
            self._sauver()

    def pince_action(self, fermer):
        with self._lock:
            cle = "fermetures" if fermer else "ouvertures"
            self.data["pince"][cle] = self.data["pince"].get(cle, 0) + 1
            self._sauver()

    # -- erreurs / répétabilité ---------------------------------------- #
    def ajouter_erreur(self, message, contexte=""):
        with self._lock:
            self.data["erreurs"].append({
                "ts": datetime.now().isoformat(timespec='seconds'),
                "message": message[:300],
                "contexte": contexte[:80],
            })
            # Cap à 200 dernières erreurs
            self.data["erreurs"] = self.data["erreurs"][-200:]
            self._sauver()

    def ajouter_repetabilite(self, resultat):
        with self._lock:
            self.data["repetabilite"].append(resultat)
            self.data["repetabilite"] = self.data["repetabilite"][-50:]
            self._sauver()

    # -- diagnostics --------------------------------------------------- #
    def diagnostic_joints(self):
        """Retourne {joint: ("ok"|"surv"|"crit", message)}."""
        out = {}
        with self._lock:
            for k, j in self.data["joints"].items():
                n = j["mouvements"]
                if n >= SEUIL_FATIGUE_JOINT_CRIT:
                    niv, msg = "crit", "Inspection recommandée"
                elif n >= SEUIL_FATIGUE_JOINT_SURV:
                    niv, msg = "surv", "Surveillance"
                else:
                    niv, msg = "ok", "Nominal"
                out[k] = (niv, msg, n, j["min_deg"], j["max_deg"])
        return out

    def diagnostic_pince(self):
        with self._lock:
            n = (self.data["pince"].get("ouvertures", 0)
                 + self.data["pince"].get("fermetures", 0))
        if n >= SEUIL_FATIGUE_PINCE_CRIT:
            return ("crit", "Inspection ressort/pneumatique recommandée", n)
        if n >= SEUIL_FATIGUE_PINCE_SURV:
            return ("surv", "Surveillance — usure possible", n)
        return ("ok", "Nominal", n)


# ============================================================================
#  6. GÉNÉRATION DE RAPPORT HTML
# ============================================================================
def _fmt_duree(s):
    s = int(s or 0)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return "%dh %02dmin" % (h, m)
    if m:
        return "%dmin %02ds" % (m, sec)
    return "%ds" % sec


def _esc(x):
    """Échappe pour insertion en HTML."""
    return html_lib.escape(str(x), quote=True)


def generer_rapport_html(stats: StatsManager, chemin_sortie: Path,
                          cycles_jour=None):
    """Génère un rapport HTML autonome et l'écrit sur disque.

    Renvoie le chemin du fichier créé.
    """
    d = stats.data
    aujourd_hui = date.today().isoformat()
    if cycles_jour is None:
        cycles_jour = stats.cycles_jour(aujourd_hui)

    total = d["cycles"]["total"]
    echecs = d["cycles"]["echecs"]
    recov = d["cycles"]["recoveries"]
    taux = (100.0 * (total - echecs) / total) if total > 0 else 0.0
    temps = stats.temps_total_s()

    diag_joints = stats.diagnostic_joints()
    diag_pince  = stats.diagnostic_pince()

    # --- Sections dynamiques en HTML --------------------------------- #
    def card(titre, valeur, sous=""):
        return ('<div class="card"><div class="lbl">%s</div>'
                '<div class="val">%s</div>'
                '<div class="sub">%s</div></div>'
                % (_esc(titre), _esc(valeur), _esc(sous)))

    def badge(niv):
        couleurs = {"ok": "#16a34a", "surv": "#d97706", "crit": "#dc2626"}
        libs = {"ok": "✓ OK", "surv": "⚠ Surveillance", "crit": "✗ Critique"}
        return ('<span class="badge" style="background:%s">%s</span>'
                % (couleurs.get(niv, "#6b7280"), libs.get(niv, niv)))

    # Tableau joints
    lignes_joints = []
    for i in range(1, 7):
        k = "joint%d" % i
        niv, msg, n, mn, mx = diag_joints[k]
        plage = ("%.1f° → %.1f°" % (mn, mx)) if mn is not None else "—"
        lignes_joints.append(
            "<tr><td><b>%s</b></td><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>"
            % (_esc(k), badge(niv), n, _esc(plage), _esc(msg)))

    # Tableau modes
    lignes_modes = []
    for libelle, m in sorted(d["modes"].items()):
        lignes_modes.append(
            "<tr><td>%s</td><td>%d</td><td>%s</td><td>%d</td></tr>"
            % (_esc(libelle), m["lancements"],
               _esc(_fmt_duree(m["duree_s"])), m["cycles"]))
    if not lignes_modes:
        lignes_modes = ['<tr><td colspan="4" class="vide">'
                        'Aucun mode lancé pour le moment.</td></tr>']

    # Tableau erreurs récentes (20 dernières)
    lignes_err = []
    for e in d["erreurs"][-20:][::-1]:
        lignes_err.append(
            "<tr><td>%s</td><td>%s</td><td class=\"mono\">%s</td></tr>"
            % (_esc(e["ts"]), _esc(e.get("contexte", "")),
               _esc(e["message"])))
    if not lignes_err:
        lignes_err = ['<tr><td colspan="3" class="vide">'
                      'Aucune erreur enregistrée. 👌</td></tr>']

    # Tableau répétabilité (5 derniers tests)
    lignes_rep = []
    for r in d["repetabilite"][-5:][::-1]:
        ecarts = r.get("ecarts_std_deg") or []
        ecart_max = max(ecarts) if ecarts else 0.0
        ok = ecart_max <= SEUIL_REPETABILITE_OK_DEG
        lignes_rep.append(
            "<tr><td>%s</td><td>%s</td><td>%d</td><td>%.3f°</td>"
            "<td>%s</td></tr>"
            % (_esc(r.get("ts", "")), _esc(r.get("position", "")),
               r.get("n", 0), ecart_max,
               badge("ok") if ok else badge("surv")))
    if not lignes_rep:
        lignes_rep = ['<tr><td colspan="5" class="vide">'
                      'Aucun test de répétabilité enregistré.</td></tr>']

    # Cycles par jour (7 derniers jours)
    par_jour = d["cycles"]["par_jour"]
    derniers = sorted(par_jour.items())[-7:]
    if derniers:
        max_v = max(v for _, v in derniers) or 1
        barres = []
        for j, v in derniers:
            h = int(160 * v / max_v)
            barres.append(
                '<div class="bar"><div class="fill" style="height:%dpx">'
                '</div><div class="val-bar">%d</div>'
                '<div class="lbl-bar">%s</div></div>'
                % (h, v, _esc(j[5:])))
        graph_html = '<div class="graph">' + "".join(barres) + "</div>"
    else:
        graph_html = '<p class="vide">Aucun cycle enregistré.</p>'

    diag_p_niv, diag_p_msg, diag_p_n = diag_pince

    # --- Template HTML ----------------------------------------------- #
    html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport d'exploitation — IGUS Rebel</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #f4f6f9; color: #1f2937; margin: 0; padding: 0;
  }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 32px 24px; }}
  header {{
    background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
    color: white; padding: 32px 24px; border-radius: 16px;
    margin-bottom: 24px;
  }}
  header h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
  header .subtitle {{ opacity: 0.9; font-size: 14px; }}
  header .author {{
    margin-top: 16px; padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.3);
    font-size: 13px; opacity: 0.95;
  }}
  h2 {{
    color: #1976d2; border-bottom: 2px solid #dde1ea;
    padding-bottom: 8px; margin-top: 32px;
  }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin: 16px 0;
  }}
  .card {{
    background: white; border-radius: 12px; padding: 20px;
    border: 1px solid #dde1ea;
  }}
  .card .lbl {{ font-size: 12px; color: #6b7280; text-transform: uppercase;
                letter-spacing: 0.5px; }}
  .card .val {{ font-size: 32px; font-weight: bold; color: #1976d2;
                margin: 4px 0; }}
  .card .sub {{ font-size: 12px; color: #6b7280; }}
  table {{
    width: 100%; border-collapse: collapse; background: white;
    border-radius: 12px; overflow: hidden; border: 1px solid #dde1ea;
  }}
  th {{ background: #f4f6f9; text-align: left; padding: 12px; font-size: 12px;
        text-transform: uppercase; color: #6b7280; letter-spacing: 0.5px; }}
  td {{ padding: 12px; border-top: 1px solid #dde1ea; font-size: 14px; }}
  td.mono {{ font-family: monospace; font-size: 12px; color: #4b5563; }}
  td.vide {{ text-align: center; color: #9ca3af; font-style: italic; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    color: white; font-size: 11px; font-weight: bold;
  }}
  .graph {{
    display: flex; align-items: flex-end; gap: 8px; height: 220px;
    background: white; padding: 16px; border-radius: 12px;
    border: 1px solid #dde1ea; margin: 16px 0;
  }}
  .bar {{
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: flex-end; gap: 4px;
  }}
  .fill {{ width: 80%; background: linear-gradient(180deg, #1976d2, #64b5f6);
           border-radius: 4px 4px 0 0; min-height: 4px; }}
  .val-bar {{ font-size: 12px; font-weight: bold; color: #1976d2; }}
  .lbl-bar {{ font-size: 11px; color: #6b7280; }}
  footer {{
    text-align: center; color: #6b7280; font-size: 12px;
    margin-top: 48px; padding-top: 16px; border-top: 1px solid #dde1ea;
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>Rapport d'exploitation — Bras robot IGUS Rebel</h1>
    <div class="subtitle">{projet}</div>
    <div class="author">
      <b>{nom}</b> — {formation}<br>
      {ecole}<br>
      Rapport généré le {date_rapport}
    </div>
  </header>

  <h2>Synthèse exécutive</h2>
  <div class="grid">
    {c1}{c2}{c3}{c4}
  </div>
  <div class="grid">
    {c5}{c6}{c7}{c8}
  </div>

  <h2>Activité — Cycles pick &amp; place par jour (7 derniers jours)</h2>
  {graph}

  <h2>Statistiques par mode</h2>
  <table>
    <tr><th>Mode</th><th>Lancements</th><th>Durée cumulée</th><th>Cycles</th></tr>
    {modes}
  </table>

  <h2>Diagnostic mécanique — Articulations</h2>
  <p style="color:#6b7280;font-size:13px;">
    Évaluation indicative basée sur le nombre de mouvements ordonnés et la
    plage angulaire utilisée. Surveillance recommandée au-delà de
    {seuil_surv} mouvements, inspection recommandée au-delà de {seuil_crit}.
  </p>
  <table>
    <tr><th>Joint</th><th>État</th><th>Mouvements</th>
        <th>Plage utilisée</th><th>Recommandation</th></tr>
    {joints}
  </table>

  <h2>Diagnostic mécanique — Pince Schunk EGP25</h2>
  <div class="grid">
    <div class="card">
      <div class="lbl">Cycles totaux</div>
      <div class="val">{pince_n}</div>
      <div class="sub">{pince_badge} — {pince_msg}</div>
    </div>
  </div>

  <h2>Tests de répétabilité</h2>
  <table>
    <tr><th>Date</th><th>Position</th><th>N cycles</th>
        <th>Écart-type max</th><th>Résultat</th></tr>
    {rep}
  </table>

  <h2>Erreurs récentes (20 dernières)</h2>
  <table>
    <tr><th>Horodatage</th><th>Contexte</th><th>Message</th></tr>
    {err}
  </table>

  <footer>
    Rapport généré automatiquement par l'IHM tactile IGUS Rebel.<br>
    {nom} · {formation} · {ecole}
  </footer>

</div>
</body>
</html>
""".format(
        projet=_esc(AUTEUR_PROJET),
        nom=_esc(AUTEUR_NOM),
        formation=_esc(AUTEUR_FORMATION),
        ecole=_esc(AUTEUR_ECOLE),
        date_rapport=_esc(datetime.now().strftime("%d/%m/%Y à %H:%M")),
        c1=card("Cycles aujourd'hui", cycles_jour, aujourd_hui),
        c2=card("Cycles total", total, "depuis le démarrage"),
        c3=card("Temps fonctionnement", _fmt_duree(temps), "cumulé"),
        c4=card("Taux de succès", "%.1f %%" % taux,
                "%d échec(s)" % echecs),
        c5=card("Cycles échoués", echecs, "avant abandon/recovery"),
        c6=card("Home recoveries", recov, "retours position sûre"),
        c7=card("Ouvertures pince", d["pince"].get("ouvertures", 0), ""),
        c8=card("Fermetures pince", d["pince"].get("fermetures", 0), ""),
        graph=graph_html,
        modes="".join(lignes_modes),
        joints="".join(lignes_joints),
        seuil_surv=SEUIL_FATIGUE_JOINT_SURV,
        seuil_crit=SEUIL_FATIGUE_JOINT_CRIT,
        pince_n=diag_p_n,
        pince_badge=badge(diag_p_niv),
        pince_msg=_esc(diag_p_msg),
        rep="".join(lignes_rep),
        err="".join(lignes_err),
    )

    RAPPORTS_DIR.mkdir(parents=True, exist_ok=True)
    chemin_sortie.write_text(html, encoding="utf-8")
    return chemin_sortie


# ============================================================================
#  7. APPLICATION IHM
# ============================================================================
class IHMRobot:
    """Application Tkinter principale."""

    def __init__(self):
        # --- État géré ----------------------------------------------------
        self.mode_proc   = None
        self.home_proc   = None        # retour HOME sécurisé via home.launch.py
        self.robotserver_proc = None   # serveur ROS local fournissant /start_robot
        self.connecte    = False
        self.mode_courant = None
        self.mode_simulation = False

        # Réglages dynamiques
        self.vitesse_traj = VITESSE_DEFAUT
        self.inc_deg      = INC_DEG_DEFAUT

        # Compteur de cycles + chrono du mode courant
        self.compteur_cycles = 0
        self.t0_mode = None        # timestamp de démarrage (datetime) ou None

        # Journal : on garde toutes les lignes en mémoire (avec niveau)
        # pour pouvoir filtrer / exporter sans relire le widget.
        self.journal_lignes = []   # [(ligne_str, "info"|"erreur"), ...]
        self.filtre_journal = "tout"  # "tout" / "info" / "erreur"
        # File thread-safe des lignes en attente d'affichage : vidée par lots
        # toutes les 100 ms (_boucle_journal) au lieu d'un after() par ligne —
        # un launch MoveIt/Gazebo crache des centaines de lignes/s.
        self._journal_queue = deque()

        # Positions personnalisées (chargées depuis disque)
        self.positions_perso = self._charger_positions()
        self._hold_del_after = None     # timer pour long-press de suppression
        self._hold_del_target = None    # nom de la position visée

        # Monitoring système
        self.sysmon = SysMonitor()

        # Statistiques persistantes + démarrage de la session
        self.stats = StatsManager()
        self.stats.session_start()

        # Limites articulaires logicielles
        self.limites = self._charger_limites()

        # Buffers pour les courbes temps réel (1 deque par joint, en degrés)
        self.courbes_buffers = [
            deque(maxlen=COURBES_N_SAMPLES) for _ in range(6)]
        self.courbes_visible = True

        # Séquenceur (programmes enchaînant positions / pince / pauses)
        self.sequences = self._charger_sequences()
        self._seq_en_cours = False
        self._seq_annule = False

        # Commande vocale (Vosk) — créée mais inactive par défaut
        self.vocal = VoiceListener(
            on_texte=lambda t: self.root.after(
                0, self._traiter_commande_vocale, t))

        # Assistant IA — zones, catalogue d'objets, planner LLM
        self.zones = ZoneManager()
        self.catalogue = CatalogueObjets()
        self.llm = LLMPlanner(self.zones, self.catalogue)
        self._ia_cmd_courante = None     # commande ros2 launch prête à lancer
        self._ia_en_cours = False

        # Détection d'anomalies par autoencoder LSTM (PyTorch). Le callback
        # bascule sur le thread UI via root.after avant d'arrêter le mode.
        self.anomaly = AnomalyDetector(
            ANOMALIE_DIR,
            on_log=lambda s: self.log(s))
        self.anomaly.set_callback_anomalie(
            lambda err, th: self.root.after(0, self._reagir_anomalie, err, th))
        self._entrainement_en_cours = False
        # Compteurs de session (réinitialisés à chaque lancement de mode)
        self._cycles_session = 0
        # Garde-fous anti-doublon pour la détection texte
        self._derniere_detection_recovery = 0.0
        self._derniere_detection_echec = 0.0
        # État du test de répétabilité
        self._repet_en_cours = False
        self._repet_annule = False

        # --- Fenêtre principale ------------------------------------------
        self.root = tk.Tk()
        self.root.title("IGUS Rebel — IHM tactile")

        # L'interface est dessinée pour un écran 720x1280 PORTRAIT. La fenêtre
        # reste donc TOUJOURS au format portrait (ratio 720:1280) :
        #  • écran portrait (kiosque HMI)  → plein écran ;
        #  • écran paysage (poste de test) → fenêtre portrait centrée, ajustée
        #    à la hauteur de l'écran (jamais étirée en paysage).
        # Le facteur d'échelle est appliqué aux polices via 'tk scaling'.
        ECRAN_REF_W, ECRAN_REF_H = 720, 1280
        ecran_w = self.root.winfo_screenwidth()
        ecran_h = self.root.winfo_screenheight()

        if ecran_h >= ecran_w:
            # Écran déjà portrait : on prend tout l'écran.
            fenetre_w, fenetre_h = ecran_w, ecran_h
            self._plein_ecran_init = True
        else:
            # Écran paysage : fenêtre portrait au ratio 720:1280, hauteur = écran.
            fenetre_h = ecran_h
            fenetre_w = int(fenetre_h * ECRAN_REF_W / ECRAN_REF_H)
            self._plein_ecran_init = False

        self.echelle = min(fenetre_w / ECRAN_REF_W, fenetre_h / ECRAN_REF_H)
        self.echelle = max(0.5, min(self.echelle, 2.0))   # garde-fou
        _base = float(self.root.tk.call('tk', 'scaling'))
        self.root.tk.call('tk', 'scaling', _base * self.echelle)

        pos_x = max(0, (ecran_w - fenetre_w) // 2)
        pos_y = max(0, (ecran_h - fenetre_h) // 2)
        self.root.geometry("%dx%d+%d+%d" % (fenetre_w, fenetre_h, pos_x, pos_y))
        self.root.configure(bg=FOND)
        if self._plein_ecran_init:
            self.root.attributes('-fullscreen', True)
        self.root.minsize(int(480 * self.echelle), int(640 * self.echelle))
        self.root.bind('<Escape>', self._toggle_fullscreen)
        self.root.bind('<F11>', self._toggle_fullscreen)
        self.root.bind('<Control-m>', lambda e: self._reduire())
        # Raccourci clavier d'urgence (alternative au bouton rouge)
        self.root.bind('<Alt-space>', self._declencher_urgence)
        self.root.protocol("WM_DELETE_WINDOW", self.quitter)

        # --- Construction de l'interface ---------------------------------
        # Ordre IMPORTANT : entête (top) + barre d'urgence (bottom) avant
        # la zone défilable qui prend tout l'espace restant.
        self._construire_entete()
        self._construire_arret_urgence()
        self._construire_zone_defilable()
        self._construire_connexion()
        self._construire_modes()
        self._construire_pince()
        self._construire_joints()
        self._construire_courbes()
        self._construire_positions()
        self._construire_portee()
        self._construire_sequenceur()
        self._construire_reglages()
        self._construire_limites()
        self._construire_vocal()
        self._construire_assistant_ia()
        self._construire_zones()
        self._construire_catalogue()
        self._construire_rapports()
        self._construire_anomalie()
        self._construire_journal()

        self.log("IHM démarrée. Workspace : %s" % WORKSPACE)
        self._maj_etat_boutons()

        # Lance RobotServer en arrière-plan : fournit /start_robot et /stop_robot
        # appelés par le bouton « Connecter ». Sans lui, la connexion échoue.
        self._lancer_robotserver()

        # Boucles périodiques (polling thread UI)
        self._boucle_horloge()
        self._boucle_sysmon()
        self._boucle_watchdog()
        self._boucle_position_live()
        self._boucle_chrono()
        self._boucle_journal()
        self._boucle_anomalie()

    # ------------------------------------------------------------------ #
    #  HELPERS DE STYLE                                                   #
    # ------------------------------------------------------------------ #
    def flat_btn(self, parent, texte, cmd, bg=BLEU, fg="#ffffff", fs=13,
                 bold=True, pady=12, padx=10, width=None, state="normal"):
        """Crée un bouton plat (sans relief) au style de l'IHM."""
        police = (POLICE, fs, "bold" if bold else "normal")
        b = tk.Button(parent, text=texte, command=cmd, bg=bg, fg=fg,
                       font=police, relief="flat", bd=0,
                       activebackground=_assombrir(bg, 0.85),
                       activeforeground=fg,
                       pady=pady, padx=padx, state=state,
                       cursor="hand2")
        if width:
            b.configure(width=width)

        # Feedback visuel au survol/appui. Les couleurs sont relues à chaque
        # entrée du curseur (cget) car plusieurs boutons changent de bg
        # dynamiquement (mode simu, vocal, répétabilité, courbes…) ; au
        # départ du curseur on ne restaure que si personne n'a changé la
        # couleur entre-temps.
        def _survol_entree(_e, b=b):
            if b["state"] == "disabled":
                return
            base = b.cget("bg")
            survol = _assombrir(base, 0.92)
            b._survol_pair = (base, survol)
            b.configure(bg=survol, activebackground=_assombrir(base, 0.85))

        def _survol_sortie(_e, b=b):
            pair = getattr(b, "_survol_pair", None)
            b._survol_pair = None
            if pair and b.cget("bg") == pair[1]:
                b.configure(bg=pair[0])

        b.bind("<Enter>", _survol_entree, add="+")
        b.bind("<Leave>", _survol_sortie, add="+")
        return b

    def section(self, parent, titre):
        """Crée une 'card' blanche avec un titre ; renvoie le frame de contenu."""
        carte = tk.Frame(parent, bg=CARD, highlightbackground=BORDURE,
                          highlightthickness=1, bd=0)
        carte.pack(fill="x", padx=14, pady=8)
        tk.Label(carte, text=titre, bg=CARD, fg=TEXTE,
                 font=(POLICE, 13, "bold")).pack(anchor="w", padx=14, pady=(12, 4))
        contenu = tk.Frame(carte, bg=CARD)
        contenu.pack(fill="x", padx=14, pady=(0, 12))
        return contenu

    # ------------------------------------------------------------------ #
    #  EN-TÊTE                                                            #
    # ------------------------------------------------------------------ #
    def _construire_entete(self):
        entete = tk.Frame(self.root, bg=BLEU)
        entete.pack(fill="x", side="top")

        # Contrôles fenêtre (réduire / plein écran ↔ fenêtré / fermer)
        controles = tk.Frame(entete, bg=BLEU)
        controles.pack(side="right", anchor="ne")

        def _btn_fenetre(texte, cmd):
            return tk.Button(controles, text=texte, command=cmd,
                             bg=BLEU, fg="#ffffff", font=(POLICE, 14, "bold"),
                             relief="flat", bd=0, activebackground=BLEU,
                             activeforeground="#ffffff", cursor="hand2",
                             padx=12, pady=6)

        _btn_fenetre("—", self._reduire).pack(side="left")
        # Glyphe initial : ❐ = restaurer (la fenêtre démarre en plein écran)
        self.btn_fenetre = _btn_fenetre("❐", self._toggle_fullscreen)
        self.btn_fenetre.pack(side="left")
        _btn_fenetre("✕", self.quitter).pack(side="left")

        # Bouton de bascule PHYSIQUE / SIMU
        self.btn_mode_simu = tk.Button(
            entete, text="PHYSIQUE", command=self._toggle_simulation,
            bg=VERT, fg="#ffffff", font=(POLICE, 10, "bold"),
            relief="flat", bd=0, activebackground=VERT,
            activeforeground="#ffffff", cursor="hand2",
            padx=12, pady=4)
        self.btn_mode_simu.pack(side="right", anchor="ne",
                                padx=(0, 8), pady=8)

        tk.Label(entete, text="IGUS Rebel", bg=BLEU, fg="#ffffff",
                 font=(POLICE, 20, "bold")).pack(anchor="w", padx=16, pady=(12, 0))
        tk.Label(entete, text="Interface de pilotage tactile", bg=BLEU,
                 fg="#cfe3f7", font=(POLICE, 11)).pack(anchor="w", padx=16)

        # Ligne d'état : voyant + statut + mode
        ligne = tk.Frame(entete, bg=BLEU)
        ligne.pack(fill="x", padx=16, pady=(12, 4))

        self.voyant = tk.Canvas(ligne, width=22, height=22, bg=BLEU,
                                highlightthickness=0)
        self._cercle = self.voyant.create_oval(3, 3, 19, 19, fill=GRIS,
                                               outline="#ffffff")
        self.voyant.pack(side="left")

        self.lbl_statut = tk.Label(ligne, text="Déconnecté", bg=BLEU,
                                   fg="#ffffff", font=(POLICE, 12, "bold"))
        self.lbl_statut.pack(side="left", padx=(8, 0))

        # Horloge à droite de la ligne d'état
        self.lbl_horloge = tk.Label(ligne, text="--:--:--", bg=BLEU,
                                     fg="#ffffff", font=(POLICE, 12, "bold"))
        self.lbl_horloge.pack(side="right")

        self.lbl_mode = tk.Label(entete, text="Mode : —", bg=BLEU,
                                 fg="#cfe3f7", font=(POLICE, 11))
        self.lbl_mode.pack(anchor="w", padx=16)

        # Bandeau monitoring RPi (CPU / RAM / Temp) + watchdog ROS comms
        bandeau_bas = tk.Frame(entete, bg=BLEU)
        bandeau_bas.pack(fill="x", padx=16, pady=(0, 10))
        self.lbl_sysmon = tk.Label(
            bandeau_bas, text="CPU —  RAM —  Temp —", bg=BLEU,
            fg="#cfe3f7", font=(POLICE, 10))
        self.lbl_sysmon.pack(side="left")
        self.lbl_watchdog = tk.Label(
            bandeau_bas, text="● Comms : —", bg=BLEU,
            fg="#cfe3f7", font=(POLICE, 10, "bold"))
        self.lbl_watchdog.pack(side="right")

    def _set_voyant(self, couleur):
        self.voyant.itemconfigure(self._cercle, fill=couleur)

    # ------------------------------------------------------------------ #
    #  BARRE D'ARRÊT D'URGENCE (flottante, en bas, hold-to-stop)          #
    # ------------------------------------------------------------------ #
    def _construire_arret_urgence(self):
        """Bouton ARRÊT D'URGENCE toujours visible (hors zone scrollable)."""
        bandeau = tk.Frame(self.root, bg=ROUGE, height=92)
        bandeau.pack(side="bottom", fill="x")
        bandeau.pack_propagate(False)

        # Clic simple immédiat : un arrêt d'urgence doit être instantané.
        # Raccourci clavier équivalent : Alt+Espace (lié sur la fenêtre).
        self.btn_urgence = tk.Button(
            bandeau, text="⛔  ARRÊT D'URGENCE  (Alt+Espace)",
            bg=ROUGE, fg="#ffffff",
            font=(POLICE, 16, "bold"),
            relief="flat", bd=0,
            activebackground="#7f1d1d", activeforeground="#ffffff",
            cursor="hand2",
            command=self._declencher_urgence)
        self.btn_urgence.pack(fill="both", expand=True, padx=8, pady=8)

    def _declencher_urgence(self, _event=None):
        """Clic (ou Alt+Espace) : arrêt total immédiat (mode + déconnexion)."""
        self.log("⛔ ARRÊT D'URGENCE déclenché.")
        # Arrêt mode + déconnexion (sans confirmation, c'est l'urgence)
        if self.mode_proc is not None and self.mode_proc.poll() is None:
            self.arreter_mode()
        if self.connecte:
            self.deconnecter()

    # ------------------------------------------------------------------ #
    #  ZONE DE CONTENU DÉFILABLE                                          #
    # ------------------------------------------------------------------ #
    def _construire_zone_defilable(self):
        conteneur = tk.Frame(self.root, bg=FOND)
        conteneur.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(conteneur, bg=FOND, highlightthickness=0)
        scroll = ttk.Scrollbar(conteneur, orient="vertical",
                               command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(self.canvas, bg=FOND)
        self._fenetre = self.canvas.create_window((0, 0), window=self.content,
                                                  anchor="nw")

        self.content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._fenetre, width=e.width))

        self.canvas.bind_all("<Button-4>",
                             lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>",
                             lambda e: self.canvas.yview_scroll(2, "units"))
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(
                                 -1 * int(e.delta / 120), "units"))
        self.canvas.bind("<ButtonPress-1>", self._debut_glisser)
        self.canvas.bind("<B1-Motion>", self._glisser)
        self._y_glisser = 0

    def _debut_glisser(self, event):
        self._y_glisser = event.y

    def _glisser(self, event):
        delta = self._y_glisser - event.y
        if abs(delta) > 2:
            self.canvas.yview_scroll(int(delta / 2), "units")
            self._y_glisser = event.y

    # ------------------------------------------------------------------ #
    #  SECTION CONNEXION                                                  #
    # ------------------------------------------------------------------ #
    def _construire_connexion(self):
        c = self.section(self.content, "Connexion robot")

        self.lbl_connexion_info = tk.Label(
            c, text="Via RobotServer (services /start_robot · /stop_robot)",
            bg=CARD, fg=GRIS, font=(POLICE, 10))
        self.lbl_connexion_info.pack(anchor="w", pady=(0, 8))

        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x")

        self.btn_connecter = self.flat_btn(ligne, "Connecter",
                                           self.connecter, bg=VERT)
        self.btn_connecter.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_deconnecter = self.flat_btn(ligne, "Déconnecter",
                                             self.deconnecter, bg=ROUGE,
                                             state="disabled")
        self.btn_deconnecter.pack(side="left", expand=True, fill="x", padx=(6, 0))

    # ------------------------------------------------------------------ #
    #  SECTION MODES                                                      #
    # ------------------------------------------------------------------ #
    def _construire_modes(self):
        c = self.section(self.content, "Modes de fonctionnement")

        tk.Label(c, text="Sélectionner un mode (connexion requise) :",
                 bg=CARD, fg=GRIS, font=(POLICE, 10)).pack(anchor="w", pady=(0, 6))

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        # Habille les widgets ttk aux couleurs de l'IHM (le clam par défaut
        # est gris et jure au milieu des cartes blanches).
        style.configure("Grand.TCombobox", font=(POLICE, 13), padding=8,
                        fieldbackground=CARD, background=CARD,
                        foreground=TEXTE, arrowcolor=BLEU,
                        bordercolor=BORDURE, lightcolor=CARD, darkcolor=CARD)
        style.map("Grand.TCombobox",
                  fieldbackground=[("readonly", CARD)],
                  foreground=[("readonly", TEXTE)])
        style.configure("TCombobox",
                        fieldbackground=CARD, background=CARD,
                        foreground=TEXTE, arrowcolor=BLEU,
                        bordercolor=BORDURE, lightcolor=CARD, darkcolor=CARD)
        style.map("TCombobox",
                  fieldbackground=[("readonly", CARD)],
                  foreground=[("readonly", TEXTE)])
        style.configure("Vertical.TScrollbar",
                        background=BORDURE, troughcolor=FOND,
                        bordercolor=FOND, arrowcolor=GRIS,
                        lightcolor=BORDURE, darkcolor=BORDURE)
        self.root.option_add('*TCombobox*Listbox.font', (POLICE, 13))
        self.root.option_add('*TCombobox*Listbox.background', CARD)
        self.root.option_add('*TCombobox*Listbox.foreground', TEXTE)
        self.root.option_add('*TCombobox*Listbox.selectBackground', BLEU)
        self.root.option_add('*TCombobox*Listbox.selectForeground', CARD)

        self.var_mode = tk.StringVar(value=list(MODES.keys())[0])
        self.combo_mode = ttk.Combobox(c, textvariable=self.var_mode,
                                       values=list(MODES.keys()),
                                       state="readonly",
                                       font=(POLICE, 13),
                                       style="Grand.TCombobox")
        self.combo_mode.pack(fill="x", pady=4, ipady=6)
        self.combo_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

        # Second combobox : chorégraphie (Robot Dance)
        self.frame_choreo = tk.Frame(c, bg=CARD)
        tk.Label(self.frame_choreo, text="Chorégraphie :", bg=CARD, fg=GRIS,
                 font=(POLICE, 10)).pack(anchor="w", pady=(6, 2))
        self.var_choreo = tk.StringVar(value=CHOREGRAPHIES[0])
        self.combo_choreo = ttk.Combobox(self.frame_choreo,
                                         textvariable=self.var_choreo,
                                         values=CHOREGRAPHIES,
                                         state="readonly",
                                         font=(POLICE, 13),
                                         style="Grand.TCombobox")
        self.combo_choreo.pack(fill="x", ipady=6)

        # Boutons Lancer / Arrêter
        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x", pady=(10, 0))

        self.btn_lancer = self.flat_btn(ligne, "▶ Lancer le mode",
                                        self.lancer_mode, bg=BLEU,
                                        state="disabled")
        self.btn_lancer.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.btn_arreter = self.flat_btn(ligne, "■ ARRÊTER",
                                         self.arreter_mode, bg=ROUGE, fs=14,
                                         state="disabled")
        self.btn_arreter.pack(side="left", expand=True, fill="x", padx=(6, 0))

        # Ligne statut + chrono + cycles
        info = tk.Frame(c, bg=CARD)
        info.pack(fill="x", pady=(8, 0))
        self.lbl_mode_statut = tk.Label(info, text="Aucun mode en cours.",
                                        bg=CARD, fg=GRIS, font=(POLICE, 10))
        self.lbl_mode_statut.pack(side="left")

        self.lbl_chrono = tk.Label(info, text="⏱ 00:00:00",
                                   bg=CARD, fg=TEXTE,
                                   font=(POLICE, 10, "bold"))
        self.lbl_chrono.pack(side="right")

        self.lbl_cycles = tk.Label(c, text="Cycles pick & place : 0",
                                   bg=CARD, fg=BLEU,
                                   font=(POLICE, 10, "bold"))
        self.lbl_cycles.pack(anchor="w", pady=(4, 0))

    def _on_mode_change(self, _event=None):
        if self.var_mode.get() == MODE_DANCE:
            self.frame_choreo.pack(fill="x", after=self.combo_mode)
        else:
            self.frame_choreo.pack_forget()

    # ------------------------------------------------------------------ #
    #  SECTION PINCE                                                      #
    # ------------------------------------------------------------------ #
    def _construire_pince(self):
        c = self.section(self.content, "Pince Schunk EGP25")

        tk.Label(c, text="Service ROS2 : /gripper/command", bg=CARD,
                 fg=GRIS, font=(POLICE, 10)).pack(anchor="w", pady=(0, 8))

        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x")

        self.flat_btn(ligne, "OUVRIR", self.ouvrir_pince,
                      bg=VERT).pack(side="left", expand=True, fill="x",
                                    padx=(0, 6))
        self.flat_btn(ligne, "FERMER", self.fermer_pince,
                      bg=ROUGE).pack(side="left", expand=True, fill="x",
                                     padx=(6, 0))

        self.lbl_pince = tk.Label(c, text="État pince : —", bg=CARD,
                                  fg=GRIS, font=(POLICE, 10))
        self.lbl_pince.pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------ #
    #  SECTION JOINTS                                                     #
    # ------------------------------------------------------------------ #
    def joints_grid(self, parent):
        """Crée la grille des 6 joints (consigne + live)."""
        self.entrees_joints = []
        self.labels_live = []

        grille = tk.Frame(parent, bg=CARD)
        grille.pack(fill="x")

        # En-tête
        ent = tk.Frame(grille, bg=CARD)
        ent.pack(fill="x", pady=(0, 4))
        tk.Label(ent, text="", bg=CARD, width=8).pack(side="left")
        tk.Label(ent, text="consigne", bg=CARD, fg=GRIS,
                 font=(POLICE, 9, "bold")).pack(side="left", padx=(78, 0))
        tk.Label(ent, text="actuel", bg=CARD, fg=GRIS,
                 font=(POLICE, 9, "bold")).pack(side="right", padx=(0, 6))

        for i in range(6):
            ligne = tk.Frame(grille, bg=CARD)
            ligne.pack(fill="x", pady=4)

            tk.Label(ligne, text="joint%d" % (i + 1), bg=CARD, fg=TEXTE,
                     font=(POLICE, 12, "bold"), width=8,
                     anchor="w").pack(side="left")

            self.flat_btn(ligne, "−", lambda idx=i: self._inc_joint(idx, -1),
                          bg=GRIS, fs=16, pady=8, padx=4,
                          width=3).pack(side="left", padx=(0, 4))

            entry = tk.Entry(ligne, font=(POLICE, 14), width=7,
                             justify="center", relief="flat",
                             bg=FOND, bd=4)
            entry.insert(0, "0")
            entry.pack(side="left", padx=4)
            self.entrees_joints.append(entry)

            self.flat_btn(ligne, "+", lambda idx=i: self._inc_joint(idx, +1),
                          bg=GRIS, fs=16, pady=8, padx=4,
                          width=3).pack(side="left", padx=(4, 0))

            tk.Label(ligne, text="°", bg=CARD, fg=GRIS,
                     font=(POLICE, 12)).pack(side="left", padx=(4, 0))

            # Affichage live de la position courante
            lbl_live = tk.Label(ligne, text="—", bg=CARD, fg=GRIS,
                                 font=(POLICE, 11, "bold"), width=8,
                                 anchor="e")
            lbl_live.pack(side="right")
            self.labels_live.append(lbl_live)

    def _construire_joints(self):
        c = self.section(self.content, "Pilotage articulaire (6 joints)")
        self.joints_grid(c)

        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x", pady=(10, 0))

        self.flat_btn(ligne, "Envoyer la position", self.envoyer_joints,
                      bg=BLEU).pack(side="left", expand=True, fill="x",
                                    padx=(0, 6))
        self.flat_btn(ligne, "↺ Capturer actuelle", self.capturer_position,
                      bg=GRIS, fs=12).pack(side="left", expand=True, fill="x",
                                           padx=(6, 0))

    def _inc_joint(self, idx, sens):
        try:
            val = float(self.entrees_joints[idx].get())
        except ValueError:
            val = 0.0
        val += sens * self.inc_deg
        self.entrees_joints[idx].delete(0, tk.END)
        self.entrees_joints[idx].insert(0, "%g" % val)

    def envoyer_joints(self):
        try:
            angles = [float(e.get()) for e in self.entrees_joints]
        except ValueError:
            messagebox.showerror("Erreur",
                                 "Les valeurs des joints doivent être numériques.")
            return
        if not self._valider_angles(angles):
            return
        envoyer_trajectoire(angles, temps=self.vitesse_traj)
        self.stats.joint_mouvement()
        self.log("Trajectoire envoyée (vitesse %ds) : %s"
                 % (self.vitesse_traj, angles))

    def capturer_position(self):
        """Copie la position courante du robot dans les Entry de consigne."""
        angles = robot_node.get_positions_deg()
        if angles is None:
            messagebox.showwarning(
                "Position indisponible",
                "Aucune donnée /joint_states reçue pour le moment.\n"
                "Le robot est-il connecté ?")
            return
        for e, a in zip(self.entrees_joints, angles):
            e.delete(0, tk.END)
            e.insert(0, "%.1f" % a)
        self.log("Position actuelle capturée : %s"
                 % ["%.1f" % a for a in angles])

    # ------------------------------------------------------------------ #
    #  SECTION COURBES TEMPS RÉEL                                         #
    # ------------------------------------------------------------------ #
    def _construire_courbes(self):
        c = self.section(self.content, "Courbes — historique des joints (30 s)")

        # En-tête + boutons d'action
        barre = tk.Frame(c, bg=CARD)
        barre.pack(fill="x", pady=(0, 6))
        tk.Label(barre,
                 text="Évolution en degrés. Échelle Y auto-ajustée.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 9)).pack(side="left")
        self.flat_btn(barre, "Effacer", self._effacer_courbes,
                      bg=GRIS, fs=10, pady=4, padx=10).pack(side="right")
        self.btn_courbes_toggle = self.flat_btn(
            barre, "Masquer", self._toggle_courbes,
            bg=BLEU, fs=10, pady=4, padx=10)
        self.btn_courbes_toggle.pack(side="right", padx=(0, 4))

        # Canvas de tracé
        self.canvas_courbes = tk.Canvas(
            c, bg="#1e1e2e", height=COURBES_HAUTEUR,
            highlightthickness=0)
        self.canvas_courbes.pack(fill="x")
        self.canvas_courbes.bind(
            "<Configure>", lambda _e: self._redessiner_courbes())

        # Légende (6 carrés colorés)
        leg = tk.Frame(c, bg=CARD)
        leg.pack(fill="x", pady=(6, 0))
        for i in range(6):
            tk.Frame(leg, bg=COULEURS_JOINTS[i],
                     width=14, height=14).pack(side="left", padx=(8, 4))
            tk.Label(leg, text="joint%d" % (i + 1), bg=CARD, fg=GRIS,
                     font=(POLICE, 9)).pack(side="left")
        self.lbl_courbes_yscale = tk.Label(
            leg, text="", bg=CARD, fg=GRIS,
            font=(POLICE, 9, "italic"))
        self.lbl_courbes_yscale.pack(side="right")

    def _toggle_courbes(self):
        self.courbes_visible = not self.courbes_visible
        if self.courbes_visible:
            self.canvas_courbes.pack(fill="x")
            self.btn_courbes_toggle.configure(text="Masquer", bg=BLEU)
        else:
            self.canvas_courbes.pack_forget()
            self.btn_courbes_toggle.configure(text="Afficher", bg=GRIS)

    def _effacer_courbes(self):
        for buf in self.courbes_buffers:
            buf.clear()
        self._redessiner_courbes()

    def _redessiner_courbes(self):
        if not self.courbes_visible:
            return
        cv = self.canvas_courbes
        # Pas de tracé si le canvas n'est pas affiché (fenêtre réduite,
        # section masquée) — le <Configure> redessinera au retour.
        if not cv.winfo_ismapped():
            return
        cv.delete("all")
        w = max(1, cv.winfo_width())
        h = COURBES_HAUTEUR

        # Recherche min/max sur tous les buffers (auto-scale Y)
        all_vals = [v for buf in self.courbes_buffers for v in buf]
        if not all_vals:
            cv.create_text(w / 2, h / 2,
                           text="En attente de /joint_states…",
                           fill=GRIS, font=(POLICE, 10))
            self.lbl_courbes_yscale.configure(text="")
            return

        y_min, y_max = min(all_vals), max(all_vals)
        if y_max - y_min < 1.0:           # plage trop étroite : on l'élargit
            mid = (y_max + y_min) / 2.0
            y_min, y_max = mid - 1.0, mid + 1.0
        marge = (y_max - y_min) * 0.1
        y_min -= marge
        y_max += marge

        # Grille horizontale (5 lignes)
        for i in range(1, 5):
            y = h * i / 5
            cv.create_line(0, y, w, y, fill="#3a3a4e", width=1)

        # Tracé des 6 courbes
        n_max = max(len(buf) for buf in self.courbes_buffers)
        if n_max < 2:
            self.lbl_courbes_yscale.configure(
                text="Y : %+.1f° / %+.1f°" % (y_min, y_max))
            return

        for i, buf in enumerate(self.courbes_buffers):
            if len(buf) < 2:
                continue
            pts = []
            for k, v in enumerate(buf):
                # Aligne à droite : les buffers de longueur < n_max
                # ont leurs premières valeurs décalées vers la droite
                x_idx = k + (n_max - len(buf))
                x = w * x_idx / (n_max - 1)
                y = h - h * (v - y_min) / (y_max - y_min)
                pts.extend([x, y])
            cv.create_line(*pts, fill=COULEURS_JOINTS[i], width=2,
                           smooth=False)

        self.lbl_courbes_yscale.configure(
            text="Y : %+.1f° → %+.1f°" % (y_min, y_max))

    # ------------------------------------------------------------------ #
    #  SECTION POSITIONS PRÉDÉFINIES + PERSO                              #
    # ------------------------------------------------------------------ #
    def _construire_positions(self):
        c = self.section(self.content, "Positions prédéfinies")
        self.positions_container = c

        # Positions usine (non supprimables)
        self._positions_usine = [
            ("Position initiale", [0, 0, 0, 0, 0, 0]),
            ("Passage porte",     [0, -30, -30, 0, -30, 0]),
        ]

        # Frame pour la liste dynamique (usine + perso)
        self.frame_positions = tk.Frame(c, bg=CARD)
        self.frame_positions.pack(fill="x")
        self._refresh_positions()

        # Ligne d'ajout d'une position perso
        sep = tk.Frame(c, bg=BORDURE, height=1)
        sep.pack(fill="x", pady=8)

        ajout = tk.Frame(c, bg=CARD)
        ajout.pack(fill="x")
        self.entry_nom_position = tk.Entry(
            ajout, font=(POLICE, 12), relief="flat", bg=FOND, bd=4)
        self.entry_nom_position.pack(side="left", expand=True, fill="x",
                                      ipady=6, padx=(0, 6))
        self.entry_nom_position.insert(0, "Nom de la position…")
        self.entry_nom_position.bind(
            "<FocusIn>",
            lambda _e: (self.entry_nom_position.delete(0, tk.END)
                        if self.entry_nom_position.get() == "Nom de la position…"
                        else None))

        self.flat_btn(ajout, "+ Enregistrer", self._sauver_position_perso,
                      bg=VERT, fs=11, pady=8).pack(side="left")

        tk.Label(c, text="Astuce : maintenez un bouton perso 1s pour le supprimer.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 9, "italic")).pack(anchor="w", pady=(8, 0))

    def _refresh_positions(self):
        """Reconstruit la liste des boutons de positions."""
        for w in self.frame_positions.winfo_children():
            w.destroy()

        # Positions usine
        for nom, angles in self._positions_usine:
            self.flat_btn(self.frame_positions, nom,
                          lambda a=angles, n=nom: self._aller_position(a, n),
                          bg=BLEU, fs=12).pack(fill="x", pady=4)

        # Positions perso (avec long-press pour supprimer)
        if self.positions_perso:
            tk.Label(self.frame_positions, text="— Mes positions —",
                     bg=CARD, fg=GRIS,
                     font=(POLICE, 9)).pack(anchor="w", pady=(6, 2))
        for nom in sorted(self.positions_perso.keys()):
            angles = self.positions_perso[nom]
            btn = self.flat_btn(
                self.frame_positions, "★ " + nom,
                lambda a=angles, n=nom: self._aller_position(a, n),
                bg="#4a4e69", fs=12)
            btn.pack(fill="x", pady=4)
            btn.bind("<ButtonPress-1>",
                     lambda _e, n=nom: self._del_press(n), add="+")
            btn.bind("<ButtonRelease-1>",
                     lambda _e: self._del_release(), add="+")

    # Positions « usine » dont la trajectoire doit passer par MoveIt
    # (workspace_scene + go_homez) — sinon l'envoi direct sur
    # /rebel_arm_controller/joint_trajectory court-circuite le contrôle
    # de collision et le bras peut taper la caméra.
    LAUNCH_SECURISE = {
        "position initiale": ("home.launch.py",          "HOME"),
        "passage porte":     ("passage_porte.launch.py", "PASSAGE PORTE"),
    }

    def _launch_securise_pour(self, nom):
        """Retourne (launch_file, label) si `nom` doit passer par MoveIt, sinon None."""
        n = (nom or "").lower()
        if n in self.LAUNCH_SECURISE:
            return self.LAUNCH_SECURISE[n]
        # Tolérance pour les préfixes type « vocal: » / « IA → ».
        for cle, val in self.LAUNCH_SECURISE.items():
            if cle in n:
                return val
        if "initiale" in n or "home" in n:
            return self.LAUNCH_SECURISE["position initiale"]
        return None

    def _aller_position(self, angles, nom):
        if not self._valider_angles(list(angles), origine=nom):
            return
        cible = self._launch_securise_pour(nom)
        if cible is not None and not self.mode_simulation:
            self._aller_position_securisee(nom, cible[0], cible[1])
            return
        for e, a in zip(self.entrees_joints, angles):
            e.delete(0, tk.END)
            e.insert(0, "%g" % a)
        envoyer_trajectoire(angles, temps=self.vitesse_traj)
        self.stats.joint_mouvement()
        self.log("Position « %s » envoyée." % nom)

    def _aller_position_securisee(self, nom, launch_file, label):
        """Mouvement sécurisé via MoveIt (workspace_scene + go_homez)."""
        if not self.connecte:
            messagebox.showwarning(
                "Connexion requise",
                "Connectez d'abord le robot (MoveIt) pour un mouvement sécurisé.")
            return
        if self.mode_proc is not None and self.mode_proc.poll() is None:
            messagebox.showwarning(
                "Mode en cours",
                "Arrêtez le mode en cours avant de lancer ce mouvement.")
            return
        if self.home_proc is not None and self.home_proc.poll() is None:
            self.log("Mouvement sécurisé déjà en cours.")
            return

        self.log("Mouvement sécurisé : %s (workspace_scene + go_homez)..." % nom)
        proc = self._lancer_processus(
            "ros2 launch mon_controleur " + launch_file)
        if proc is None:
            self.lbl_mode_statut.configure(
                text="Échec du lancement (%s)." % label)
            return
        self.home_proc = proc

        Thread(target=self._lecteur_stdout,
               args=(proc, "[%s]" % label), daemon=True).start()

        def _attendre():
            proc.wait()
            if self.home_proc is proc:
                self.home_proc = None
            self.root.after(0, self.log,
                            "Mouvement sécurisé terminé (%s)." % label)
        Thread(target=_attendre, daemon=True).start()
        self.stats.joint_mouvement()

    def _charger_positions(self):
        try:
            if POSITIONS_FILE.exists():
                with open(POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                # Validation : doit être un dict {str: list[6 numbers]}
                ok = {}
                for k, v in data.items():
                    if (isinstance(k, str) and isinstance(v, list)
                            and len(v) == 6
                            and all(isinstance(x, (int, float)) for x in v)):
                        ok[k] = list(v)
                return ok
        except Exception:
            pass
        return {}

    def _sauver_disque(self):
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(self.positions_perso, f, indent=2)
        except Exception as exc:
            self.log("Sauvegarde positions : erreur — %s" % exc)

    def _sauver_position_perso(self):
        nom = self.entry_nom_position.get().strip()
        if not nom or nom == "Nom de la position…":
            messagebox.showwarning("Nom requis",
                                   "Donnez un nom à votre position.")
            return
        try:
            angles = [float(e.get()) for e in self.entrees_joints]
        except ValueError:
            messagebox.showerror("Erreur",
                                 "Les valeurs des joints doivent être numériques.")
            return
        # Si nom déjà utilisé (usine ou perso), demander confirmation
        usines = {n for n, _ in self._positions_usine}
        if nom in usines:
            messagebox.showwarning("Nom réservé",
                                   "Ce nom est réservé à une position usine.")
            return
        if nom in self.positions_perso:
            if not messagebox.askyesno(
                    "Écraser ?",
                    "Une position « %s » existe déjà. Remplacer ?" % nom):
                return
        self.positions_perso[nom] = angles
        self._sauver_disque()
        self._refresh_positions()
        self.entry_nom_position.delete(0, tk.END)
        self.log("Position perso enregistrée : %s = %s" % (nom, angles))

    def _del_press(self, nom):
        """Début long-press sur une position perso."""
        self._hold_del_target = nom
        self._hold_del_after = self.root.after(
            HOLD_DEL_MS, lambda: self._confirmer_suppression(nom))

    def _del_release(self):
        """Relâchement : annule la suppression si pas déclenchée."""
        if self._hold_del_after is not None:
            try:
                self.root.after_cancel(self._hold_del_after)
            except Exception:
                pass
            self._hold_del_after = None
        self._hold_del_target = None

    def _confirmer_suppression(self, nom):
        self._hold_del_after = None
        if nom not in self.positions_perso:
            return
        if messagebox.askyesno("Supprimer ?",
                               "Supprimer la position « %s » ?" % nom):
            del self.positions_perso[nom]
            self._sauver_disque()
            self._refresh_positions()
            self.log("Position perso supprimée : %s" % nom)

    # ------------------------------------------------------------------ #
    #  SECTION SÉQUENCEUR DE POSITIONS                                    #
    # ------------------------------------------------------------------ #
    def _charger_sequences(self):
        try:
            if SEQUENCES_FILE.exists():
                with open(SEQUENCES_FILE, 'r') as f:
                    data = json.load(f)
                # Validation minimale
                out = {}
                for nom, seq in data.items():
                    if (isinstance(nom, str) and isinstance(seq, dict)
                            and isinstance(seq.get("etapes"), list)):
                        out[nom] = {
                            "etapes": [e for e in seq["etapes"]
                                       if isinstance(e, dict)
                                       and e.get("type") in ("move", "pince",
                                                              "pause")],
                            "boucles": int(seq.get("boucles", 1)),
                        }
                return out
        except Exception:
            pass
        return {}

    def _sauver_sequences(self):
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(SEQUENCES_FILE, 'w') as f:
                json.dump(self.sequences, f, indent=2)
        except Exception as exc:
            self.log("Séquences : erreur sauvegarde — %s" % exc)

    # ------------------------------------------------------------------ #
    #  SECTION PORTÉE XY (calculateur y maxi)                            #
    # ------------------------------------------------------------------ #
    def _construire_portee(self):
        """Calculateur : pour un x donné, y maxi tel que √(x²+y²) ≤ RAYON_MAX."""
        c = self.section(self.content, "Portée XY — y maxi")

        tk.Label(
            c,
            text="Contrainte : √(x² + y²) ≤ %.2f m" % RAYON_MAX,
            bg=CARD, fg=GRIS, font=(POLICE, 10)).pack(anchor="w", pady=(0, 8))

        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x")

        tk.Label(ligne, text="x (m) :", bg=CARD, fg=TEXTE,
                 font=(POLICE, 12)).pack(side="left", padx=(0, 6))

        self.entry_portee_x = tk.Entry(
            ligne, font=(POLICE, 12), relief="flat", bg=FOND, bd=4, width=10)
        self.entry_portee_x.pack(side="left", ipady=6, padx=(0, 6))
        self.entry_portee_x.bind("<Return>", lambda _e: self._calculer_portee())

        self.flat_btn(ligne, "Calculer", self._calculer_portee,
                      bg=BLEU, fs=11, pady=8).pack(side="left")

        self.lbl_portee_res = tk.Label(
            c, text="y maxi : —", bg=CARD, fg=TEXTE,
            font=(POLICE, 13, "bold"))
        self.lbl_portee_res.pack(anchor="w", pady=(10, 0))

    def _calculer_portee(self):
        """Calcule y maxi = √(RAYON_MAX² − x²) pour le x saisi."""
        brut = self.entry_portee_x.get().strip().replace(",", ".")
        try:
            x = float(brut)
        except ValueError:
            self.lbl_portee_res.configure(
                text="Entrez une valeur numérique pour x.", fg=ROUGE)
            return

        if abs(x) > RAYON_MAX:
            self.lbl_portee_res.configure(
                text="|x| > %.2f m : hors de portée, aucun y possible." % RAYON_MAX,
                fg=ROUGE)
            return

        y_max = math.sqrt(RAYON_MAX ** 2 - x ** 2)
        self.lbl_portee_res.configure(
            text="y maxi : ± %.3f m  (donc −%.3f ≤ y ≤ %.3f)"
                 % (y_max, y_max, y_max),
            fg=VERT)

    def _construire_sequenceur(self):
        c = self.section(self.content, "Séquenceur de positions")

        tk.Label(c,
                 text="Enchaînez positions, actions pince et pauses ; "
                      "exécutez en boucle.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10)).pack(anchor="w", pady=(0, 8))

        # Sélection de la séquence
        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x")
        self.var_seq = tk.StringVar()
        self.combo_seq = ttk.Combobox(
            ligne, textvariable=self.var_seq, state="readonly",
            font=(POLICE, 12), style="Grand.TCombobox")
        self.combo_seq.pack(side="left", expand=True, fill="x",
                             ipady=4, padx=(0, 6))
        self.combo_seq.bind(
            "<<ComboboxSelected>>", lambda _e: self._maj_apercu_seq())

        # Spinbox nombre de boucles (1, 2, 5, 10, ∞)
        tk.Label(ligne, text="Boucles :", bg=CARD, fg=GRIS,
                 font=(POLICE, 10)).pack(side="left", padx=(4, 4))
        self.var_seq_boucles = tk.StringVar(value="1")
        ttk.Combobox(ligne, textvariable=self.var_seq_boucles,
                     values=("1", "2", "5", "10", "∞"),
                     state="readonly", width=4,
                     font=(POLICE, 11)).pack(side="left")

        # Aperçu (label texte avec résumé des étapes)
        self.lbl_seq_apercu = tk.Label(
            c, text="Aucune séquence sélectionnée.", bg=CARD, fg=GRIS,
            font=(POLICE, 10, "italic"),
            justify="left", anchor="w", wraplength=560)
        self.lbl_seq_apercu.pack(fill="x", pady=(8, 6))

        # Boutons Lancer / Arrêter / Éditer / Nouveau / Supprimer
        b1 = tk.Frame(c, bg=CARD)
        b1.pack(fill="x")
        self.btn_seq_lancer = self.flat_btn(
            b1, "▶ Lancer", self._seq_lancer, bg=VERT, fs=12)
        self.btn_seq_lancer.pack(side="left", expand=True, fill="x",
                                  padx=(0, 4))
        self.btn_seq_arreter = self.flat_btn(
            b1, "■ Arrêter", self._seq_arreter, bg=ROUGE, fs=12,
            state="disabled")
        self.btn_seq_arreter.pack(side="left", expand=True, fill="x",
                                   padx=(4, 0))

        b2 = tk.Frame(c, bg=CARD)
        b2.pack(fill="x", pady=(6, 0))
        self.flat_btn(b2, "+ Nouvelle",
                      lambda: self._editeur_sequence(None),
                      bg=BLEU, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(0, 4))
        self.flat_btn(b2, "✏ Éditer",
                      self._editer_seq_courante,
                      bg=GRIS, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(4, 0))
        self.flat_btn(b2, "🗑 Supprimer",
                      self._supprimer_seq_courante,
                      bg=GRIS, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(4, 0))

        # Statut d'exécution
        self.lbl_seq_statut = tk.Label(
            c, text="Inactif.", bg=CARD, fg=GRIS,
            font=(POLICE, 10))
        self.lbl_seq_statut.pack(anchor="w", pady=(8, 0))

        self._refresh_combo_seq()

    def _refresh_combo_seq(self):
        noms = sorted(self.sequences.keys())
        self.combo_seq.configure(values=noms)
        cur = self.var_seq.get()
        if cur not in noms:
            self.var_seq.set(noms[0] if noms else "")
        self._maj_apercu_seq()

    def _maj_apercu_seq(self):
        nom = self.var_seq.get()
        if not nom or nom not in self.sequences:
            self.lbl_seq_apercu.configure(
                text="Aucune séquence. Cliquez « + Nouvelle » pour créer.")
            return
        seq = self.sequences[nom]
        lignes = []
        for i, e in enumerate(seq["etapes"], start=1):
            lignes.append("%2d. %s" % (i, self._fmt_etape(e)))
        txt = ("Séquence « %s » — %d étape(s), %d boucle(s) :\n"
               % (nom, len(seq["etapes"]), seq.get("boucles", 1))
               + "\n".join(lignes) if lignes
               else "Séquence vide.")
        self.lbl_seq_apercu.configure(text=txt)

    def _fmt_etape(self, e):
        t = e.get("type")
        if t == "move":
            ang = e.get("angles", [])
            return "Move → [%s]°" % ", ".join("%g" % a for a in ang)
        if t == "pince":
            return "Pince : %s" % ("fermer" if e.get("fermer") else "ouvrir")
        if t == "pause":
            return "Pause %gs" % e.get("duree", 1.0)
        return "?"

    def _editer_seq_courante(self):
        nom = self.var_seq.get()
        if not nom:
            messagebox.showinfo("Aucune séquence",
                                "Aucune séquence sélectionnée à éditer.")
            return
        self._editeur_sequence(nom)

    def _supprimer_seq_courante(self):
        nom = self.var_seq.get()
        if not nom or nom not in self.sequences:
            return
        if not messagebox.askyesno("Supprimer ?",
                                    "Supprimer la séquence « %s » ?" % nom):
            return
        del self.sequences[nom]
        self._sauver_sequences()
        self._refresh_combo_seq()
        self.log("Séquence supprimée : %s" % nom)

    # ---- Exécution --------------------------------------------------- #
    def _seq_lancer(self):
        nom = self.var_seq.get()
        if not nom or nom not in self.sequences:
            return
        if self._seq_en_cours:
            return
        if not self.connecte and not self.mode_simulation:
            messagebox.showwarning("Connexion requise",
                                   "Connectez d'abord le robot.")
            return

        seq = self.sequences[nom]
        if not seq["etapes"]:
            messagebox.showinfo("Séquence vide",
                                "Aucune étape à exécuter.")
            return

        # Validation préalable des limites pour tous les move
        for e in seq["etapes"]:
            if e["type"] == "move":
                if not self._valider_angles(e["angles"],
                                             origine="séquence « %s »" % nom):
                    return

        boucles_str = self.var_seq_boucles.get()
        boucles = -1 if boucles_str == "∞" else int(boucles_str)

        self._seq_en_cours = True
        self._seq_annule = False
        self.btn_seq_lancer.configure(state="disabled")
        self.btn_seq_arreter.configure(state="normal")
        self.log("Séquence « %s » lancée (boucles : %s)"
                 % (nom, boucles_str))
        Thread(target=self._seq_worker,
               args=(nom, seq, boucles), daemon=True).start()

    def _seq_arreter(self):
        self._seq_annule = True
        self.lbl_seq_statut.configure(text="Arrêt en cours…", fg=ORANGE)

    def _seq_worker(self, nom, seq, boucles_demande):
        """Thread d'exécution de la séquence."""
        etapes = seq["etapes"]
        boucle = 0
        try:
            while not self._seq_annule:
                boucle += 1
                if boucles_demande > 0 and boucle > boucles_demande:
                    break
                for idx, etape in enumerate(etapes, start=1):
                    if self._seq_annule:
                        break
                    libelle = "Boucle %d • étape %d/%d — %s" % (
                        boucle, idx, len(etapes), self._fmt_etape(etape))
                    self.root.after(0,
                                    lambda l=libelle:
                                    self.lbl_seq_statut.configure(
                                        text=l, fg=BLEU))
                    ok = self._seq_executer_etape(etape)
                    if not ok and not self._seq_annule:
                        self.root.after(0, self.log,
                                        "Séquence : étape %d échouée." % idx)
                        # On poursuit malgré l'échec (best effort)
        finally:
            self.root.after(0, self._seq_fin, nom, boucle)

    def _seq_executer_etape(self, etape):
        t = etape.get("type")
        if t == "move":
            angles = list(etape["angles"])
            envoyer_trajectoire(angles, temps=self.vitesse_traj)
            self.stats.joint_mouvement()
            # Attente : durée trajectoire + 1s stabilisation
            duree = self.vitesse_traj + 1.0
            tic = _time.monotonic()
            while _time.monotonic() - tic < duree:
                if self._seq_annule:
                    return False
                _time.sleep(0.1)
            return True
        if t == "pince":
            fermer = bool(etape.get("fermer"))
            return self._seq_pince(fermer)
        if t == "pause":
            duree = float(etape.get("duree", 1.0))
            tic = _time.monotonic()
            while _time.monotonic() - tic < duree:
                if self._seq_annule:
                    return False
                _time.sleep(0.1)
            return True
        return False

    def _seq_pince(self, fermer):
        """Appel synchrone au service /gripper/command depuis un thread worker."""
        client = robot_node.gripper_client
        if not client.wait_for_service(timeout_sec=2.0):
            return False
        req = SetBool.Request()
        req.data = fermer
        future = client.call_async(req)
        tic = _time.monotonic()
        while not future.done():
            if self._seq_annule:
                return False
            if _time.monotonic() - tic > 5.0:
                return False
            _time.sleep(0.1)
        try:
            res = future.result()
            if res is not None and res.success:
                self.stats.pince_action(fermer)
                return True
        except Exception:
            pass
        return False

    def _seq_fin(self, nom, boucles_realisees):
        self._seq_en_cours = False
        self.btn_seq_lancer.configure(state="normal")
        self.btn_seq_arreter.configure(state="disabled")
        if self._seq_annule:
            self.lbl_seq_statut.configure(
                text="Séquence annulée après %d boucle(s)." % (
                    boucles_realisees - 1),
                fg=ORANGE)
            self.log("Séquence « %s » annulée." % nom)
        else:
            self.lbl_seq_statut.configure(
                text="Séquence terminée (%d boucle(s))." % (
                    boucles_realisees - 1),
                fg=VERT)
            self.log("Séquence « %s » terminée." % nom)

    # ---- Éditeur de séquence (dialog modal) -------------------------- #
    def _editeur_sequence(self, nom_existant):
        """Ouvre l'éditeur. Si nom_existant=None : création."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Éditeur de séquence")
        dlg.configure(bg=FOND)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("%dx%d" % (int(600 * self.echelle), int(680 * self.echelle)))

        # Données courantes (copie pour pouvoir annuler)
        if nom_existant and nom_existant in self.sequences:
            travail = {
                "nom": nom_existant,
                "etapes": [dict(e) for e in
                            self.sequences[nom_existant]["etapes"]],
            }
        else:
            travail = {"nom": "", "etapes": []}

        # En-tête : nom de la séquence
        tk.Label(dlg, text="Nom de la séquence",
                 bg=FOND, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w", padx=20,
                                                   pady=(16, 2))
        e_nom = tk.Entry(dlg, font=(POLICE, 12), relief="flat",
                         bg="white", bd=4)
        e_nom.insert(0, travail["nom"])
        e_nom.pack(fill="x", padx=20, ipady=4, pady=(0, 12))

        # Liste des étapes (Listbox + scrollbar)
        tk.Label(dlg, text="Étapes",
                 bg=FOND, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w", padx=20)

        cadre_lst = tk.Frame(dlg, bg=FOND)
        cadre_lst.pack(fill="both", expand=True, padx=20, pady=4)
        sb = ttk.Scrollbar(cadre_lst, orient="vertical")
        lst = tk.Listbox(cadre_lst, font=(POLICE, 10),
                         yscrollcommand=sb.set,
                         relief="flat", bg="white", height=12)
        sb.config(command=lst.yview)
        sb.pack(side="right", fill="y")
        lst.pack(side="left", fill="both", expand=True)

        def _refresh_lst():
            lst.delete(0, tk.END)
            for i, e in enumerate(travail["etapes"], start=1):
                lst.insert(tk.END, "%2d. %s" % (i, self._fmt_etape(e)))

        _refresh_lst()

        # Boutons de gestion des étapes
        bt = tk.Frame(dlg, bg=FOND)
        bt.pack(fill="x", padx=20, pady=(4, 8))

        def _monter():
            sel = lst.curselection()
            if not sel or sel[0] == 0:
                return
            i = sel[0]
            travail["etapes"][i-1], travail["etapes"][i] = (
                travail["etapes"][i], travail["etapes"][i-1])
            _refresh_lst()
            lst.selection_set(i-1)

        def _descendre():
            sel = lst.curselection()
            if not sel or sel[0] >= len(travail["etapes"]) - 1:
                return
            i = sel[0]
            travail["etapes"][i+1], travail["etapes"][i] = (
                travail["etapes"][i], travail["etapes"][i+1])
            _refresh_lst()
            lst.selection_set(i+1)

        def _retirer():
            sel = lst.curselection()
            if not sel:
                return
            del travail["etapes"][sel[0]]
            _refresh_lst()

        def _ajouter():
            etape = self._dialog_ajouter_etape(dlg)
            if etape:
                travail["etapes"].append(etape)
                _refresh_lst()

        tk.Button(bt, text="+ Ajouter étape", command=_ajouter,
                  bg=VERT, fg="white", relief="flat",
                  font=(POLICE, 10, "bold"), padx=10, pady=6).pack(
            side="left", padx=(0, 4))
        tk.Button(bt, text="▲", command=_monter,
                  bg=GRIS, fg="white", relief="flat",
                  font=(POLICE, 10), padx=10, pady=6).pack(side="left",
                                                            padx=2)
        tk.Button(bt, text="▼", command=_descendre,
                  bg=GRIS, fg="white", relief="flat",
                  font=(POLICE, 10), padx=10, pady=6).pack(side="left",
                                                            padx=2)
        tk.Button(bt, text="🗑 Retirer", command=_retirer,
                  bg=ROUGE, fg="white", relief="flat",
                  font=(POLICE, 10), padx=10, pady=6).pack(side="left",
                                                            padx=4)

        # Boutons OK / Annuler
        bf = tk.Frame(dlg, bg=FOND)
        bf.pack(fill="x", padx=20, pady=(0, 16))

        def _enregistrer():
            nom = e_nom.get().strip()
            if not nom:
                messagebox.showwarning("Nom requis", "Donnez un nom.",
                                       parent=dlg)
                return
            if not travail["etapes"]:
                messagebox.showwarning("Vide",
                                       "Au moins une étape requise.",
                                       parent=dlg)
                return
            # Si renommage : retire l'ancien
            if nom_existant and nom_existant != nom:
                self.sequences.pop(nom_existant, None)
            if nom in self.sequences and nom != nom_existant:
                if not messagebox.askyesno(
                        "Écraser ?",
                        "« %s » existe déjà. Remplacer ?" % nom,
                        parent=dlg):
                    return
            self.sequences[nom] = {
                "etapes": travail["etapes"],
                "boucles": self.sequences.get(nom, {}).get("boucles", 1),
            }
            self._sauver_sequences()
            self._refresh_combo_seq()
            self.var_seq.set(nom)
            self._maj_apercu_seq()
            self.log("Séquence enregistrée : %s (%d étapes)"
                     % (nom, len(travail["etapes"])))
            dlg.destroy()

        tk.Button(bf, text="Annuler", command=dlg.destroy,
                  bg=GRIS, fg="white", relief="flat",
                  font=(POLICE, 11), padx=20, pady=8).pack(side="left")
        tk.Button(bf, text="Enregistrer", command=_enregistrer,
                  bg=VERT, fg="white", relief="flat",
                  font=(POLICE, 11, "bold"), padx=20, pady=8).pack(
            side="right")

    def _dialog_ajouter_etape(self, parent):
        """Sous-dialog : choix du type + paramètres. Renvoie etape ou None."""
        dlg = tk.Toplevel(parent)
        dlg.title("Ajouter une étape")
        dlg.configure(bg=FOND)
        dlg.transient(parent)
        dlg.grab_set()

        resultat = {"etape": None}

        tk.Label(dlg, text="Type d'étape",
                 bg=FOND, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w", padx=20,
                                                   pady=(16, 4))
        var_type = tk.StringVar(value="move")
        for val, lib in (("move", "Mouvement vers une position"),
                          ("pince", "Action pince"),
                          ("pause", "Pause")):
            tk.Radiobutton(dlg, text=lib, variable=var_type, value=val,
                           bg=FOND, fg=TEXTE, selectcolor="white",
                           font=(POLICE, 10),
                           command=lambda: _maj_form()).pack(
                anchor="w", padx=20)

        # Formulaire dynamique selon type
        frame_form = tk.Frame(dlg, bg=FOND)
        frame_form.pack(fill="x", padx=20, pady=(12, 0))

        var_pos = tk.StringVar()
        var_pince = tk.StringVar(value="ouvrir")
        var_pause = tk.DoubleVar(value=1.0)

        def _maj_form():
            for w in frame_form.winfo_children():
                w.destroy()
            t = var_type.get()
            if t == "move":
                tk.Label(frame_form,
                         text="Position (consigne actuelle ou enregistrée)",
                         bg=FOND, fg=TEXTE,
                         font=(POLICE, 10, "bold")).pack(anchor="w")
                noms = ["[consigne actuelle des Entry]"]
                noms += [n for n, _ in self._positions_usine]
                noms += sorted(self.positions_perso.keys())
                if not var_pos.get():
                    var_pos.set(noms[0])
                ttk.Combobox(frame_form, textvariable=var_pos,
                             values=noms, state="readonly",
                             font=(POLICE, 10)).pack(fill="x", pady=(0, 8))
            elif t == "pince":
                tk.Label(frame_form, text="Action",
                         bg=FOND, fg=TEXTE,
                         font=(POLICE, 10, "bold")).pack(anchor="w")
                for val, lib in (("ouvrir", "Ouvrir la pince"),
                                  ("fermer", "Fermer la pince")):
                    tk.Radiobutton(frame_form, text=lib,
                                   variable=var_pince, value=val,
                                   bg=FOND, font=(POLICE, 10),
                                   selectcolor="white").pack(anchor="w")
            elif t == "pause":
                tk.Label(frame_form, text="Durée (secondes, 0.5 → 30)",
                         bg=FOND, fg=TEXTE,
                         font=(POLICE, 10, "bold")).pack(anchor="w")
                tk.Scale(frame_form, from_=0.5, to=30, resolution=0.5,
                         orient="horizontal", variable=var_pause,
                         bg=FOND, length=300,
                         highlightthickness=0).pack(fill="x")

        _maj_form()

        def _ok():
            t = var_type.get()
            if t == "move":
                nom = var_pos.get()
                if nom == "[consigne actuelle des Entry]":
                    try:
                        ang = [float(e.get()) for e in self.entrees_joints]
                    except ValueError:
                        messagebox.showerror(
                            "Erreur",
                            "Les Entry de joints contiennent des valeurs "
                            "non numériques.", parent=dlg)
                        return
                else:
                    src = dict(self._positions_usine)
                    src.update(self.positions_perso)
                    if nom not in src:
                        messagebox.showerror(
                            "Inconnue",
                            "Position introuvable.", parent=dlg)
                        return
                    ang = list(src[nom])
                resultat["etape"] = {"type": "move", "angles": ang}
            elif t == "pince":
                resultat["etape"] = {
                    "type": "pince",
                    "fermer": var_pince.get() == "fermer"}
            elif t == "pause":
                resultat["etape"] = {
                    "type": "pause", "duree": float(var_pause.get())}
            dlg.destroy()

        bf = tk.Frame(dlg, bg=FOND)
        bf.pack(fill="x", padx=20, pady=16)
        tk.Button(bf, text="Annuler", command=dlg.destroy,
                  bg=GRIS, fg="white", relief="flat",
                  font=(POLICE, 11), padx=20, pady=8).pack(side="left")
        tk.Button(bf, text="Ajouter", command=_ok,
                  bg=VERT, fg="white", relief="flat",
                  font=(POLICE, 11, "bold"), padx=20, pady=8).pack(
            side="right")

        dlg.wait_window()
        return resultat["etape"]

    # ------------------------------------------------------------------ #
    #  SECTION RÉGLAGES (vitesse + incrément)                             #
    # ------------------------------------------------------------------ #
    def _construire_reglages(self):
        c = self.section(self.content, "Réglages de pilotage")

        # Vitesse (durée d'une trajectoire)
        tk.Label(c, text="Vitesse — durée d'une trajectoire (s)",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w")
        self.var_vitesse = tk.IntVar(value=VITESSE_DEFAUT)
        self.scale_vitesse = tk.Scale(
            c, from_=1, to=10, orient="horizontal",
            variable=self.var_vitesse, bg=CARD, fg=TEXTE,
            troughcolor=FOND, highlightthickness=0, relief="flat",
            length=200, font=(POLICE, 10),
            command=self._on_vitesse)
        self.scale_vitesse.pack(fill="x")

        # Incrément des boutons +/-
        tk.Label(c, text="Incrément des boutons +/- (degrés)",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w", pady=(8, 0))
        self.var_inc = tk.DoubleVar(value=INC_DEG_DEFAUT)
        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x")
        for val in (1.0, 5.0, 10.0, 30.0):
            tk.Radiobutton(
                ligne, text="%g°" % val, variable=self.var_inc, value=val,
                bg=CARD, fg=TEXTE, selectcolor=FOND,
                font=(POLICE, 11),
                command=self._on_inc).pack(side="left", expand=True)

    def _on_vitesse(self, _val=None):
        self.vitesse_traj = int(self.var_vitesse.get())

    def _on_inc(self):
        self.inc_deg = float(self.var_inc.get())

    # ------------------------------------------------------------------ #
    #  SECTION LIMITES ARTICULAIRES (soft limits)                         #
    # ------------------------------------------------------------------ #
    def _charger_limites(self):
        try:
            if LIMITS_FILE.exists():
                with open(LIMITS_FILE, 'r') as f:
                    data = json.load(f)
                limites = dict(LIMITES_DEFAUT)
                for k, v in data.items():
                    if (k in limites and isinstance(v, list)
                            and len(v) == 2
                            and all(isinstance(x, (int, float)) for x in v)):
                        limites[k] = (float(v[0]), float(v[1]))
                return limites
        except Exception:
            pass
        return dict(LIMITES_DEFAUT)

    def _sauver_limites(self):
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(LIMITS_FILE, 'w') as f:
                json.dump({k: list(v) for k, v in self.limites.items()},
                          f, indent=2)
        except Exception as exc:
            self.log("Limites : erreur sauvegarde — %s" % exc)

    def _valider_angles(self, angles, origine="commande"):
        """Vérifie que les angles respectent les limites.

        Si dépassement, demande confirmation à l'utilisateur (UI thread).
        Renvoie True si on peut envoyer, False sinon.
        """
        violations = []
        for i, a in enumerate(angles, start=1):
            key = "joint%d" % i
            mn, mx = self.limites.get(key, (-180.0, 180.0))
            if a < mn - 1e-3 or a > mx + 1e-3:
                violations.append("%s : %+.1f° hors [%+.0f, %+.0f]"
                                  % (key, a, mn, mx))
        if not violations:
            return True
        msg = ("Hors des limites configurées (%s) :\n\n" % origine
               + "\n".join(violations)
               + "\n\nEnvoyer quand même ?\n"
               + "(Vous pouvez ajuster les limites dans la section dédiée.)")
        return messagebox.askyesno("Limites articulaires", msg)

    def _construire_limites(self):
        c = self.section(self.content, "Limites articulaires (soft limits)")

        tk.Label(c,
                 text="Plage acceptée pour chaque joint. Une commande hors "
                      "plage demande confirmation avant envoi.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10), justify="left").pack(
            anchor="w", pady=(0, 8))

        self.entrees_limites = {}  # joint -> (entry_min, entry_max)
        grille = tk.Frame(c, bg=CARD)
        grille.pack(fill="x")

        # En-tête
        ent = tk.Frame(grille, bg=CARD)
        ent.pack(fill="x", pady=(0, 4))
        tk.Label(ent, text="", bg=CARD, width=8).pack(side="left")
        tk.Label(ent, text="min (°)", bg=CARD, fg=GRIS,
                 font=(POLICE, 9, "bold"),
                 width=8).pack(side="left", padx=(8, 0))
        tk.Label(ent, text="max (°)", bg=CARD, fg=GRIS,
                 font=(POLICE, 9, "bold"),
                 width=8).pack(side="left", padx=(8, 0))

        for i in range(1, 7):
            ligne = tk.Frame(grille, bg=CARD)
            ligne.pack(fill="x", pady=3)
            tk.Label(ligne, text="joint%d" % i, bg=CARD, fg=TEXTE,
                     font=(POLICE, 11, "bold"), width=8,
                     anchor="w").pack(side="left")

            key = "joint%d" % i
            mn, mx = self.limites[key]
            e_min = tk.Entry(ligne, font=(POLICE, 11), width=8,
                              justify="center", relief="flat",
                              bg=FOND, bd=4)
            e_min.insert(0, "%g" % mn)
            e_min.pack(side="left", padx=(8, 0))

            e_max = tk.Entry(ligne, font=(POLICE, 11), width=8,
                              justify="center", relief="flat",
                              bg=FOND, bd=4)
            e_max.insert(0, "%g" % mx)
            e_max.pack(side="left", padx=(8, 0))

            self.entrees_limites[key] = (e_min, e_max)

        ligne_btns = tk.Frame(c, bg=CARD)
        ligne_btns.pack(fill="x", pady=(10, 0))
        self.flat_btn(ligne_btns, "Appliquer & sauvegarder",
                      self._appliquer_limites,
                      bg=VERT, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(0, 4))
        self.flat_btn(ligne_btns, "Réinitialiser",
                      self._reset_limites,
                      bg=GRIS, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(4, 0))

    def _appliquer_limites(self):
        nouvelles = {}
        for key, (e_min, e_max) in self.entrees_limites.items():
            try:
                mn = float(e_min.get())
                mx = float(e_max.get())
            except ValueError:
                messagebox.showerror("Valeur invalide",
                                     "%s : min/max doivent être numériques." % key)
                return
            if mn >= mx:
                messagebox.showerror(
                    "Valeur invalide",
                    "%s : min (%g) doit être strictement inférieur à max (%g)."
                    % (key, mn, mx))
                return
            nouvelles[key] = (mn, mx)
        self.limites = nouvelles
        self._sauver_limites()
        self.log("Limites articulaires mises à jour.")

    def _reset_limites(self):
        if not messagebox.askyesno("Réinitialiser ?",
                                    "Restaurer les limites par défaut "
                                    "(−180° / +180° par joint) ?"):
            return
        self.limites = dict(LIMITES_DEFAUT)
        self._sauver_limites()
        for key, (e_min, e_max) in self.entrees_limites.items():
            mn, mx = self.limites[key]
            e_min.delete(0, tk.END); e_min.insert(0, "%g" % mn)
            e_max.delete(0, tk.END); e_max.insert(0, "%g" % mx)
        self.log("Limites articulaires réinitialisées.")

    # ------------------------------------------------------------------ #
    #  SECTION COMMANDE VOCALE (Vosk hors ligne)                          #
    # ------------------------------------------------------------------ #
    def _construire_vocal(self):
        c = self.section(self.content, "Commande vocale (hors ligne, Vosk)")

        tk.Label(c,
                 text="Reconnaissance locale, sans cloud. Dites « stop » ou "
                      "« arrêt » pour l'urgence ; phrases longues pour les "
                      "autres commandes.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10), wraplength=560,
                 justify="left").pack(anchor="w", pady=(0, 8))

        self.btn_vocal = self.flat_btn(
            c, "🎤 Activer l'écoute", self._toggle_vocal, bg=BLEU)
        self.btn_vocal.pack(fill="x", pady=(0, 6))

        self.lbl_vocal_statut = tk.Label(
            c, text="Écoute désactivée.", bg=CARD, fg=GRIS,
            font=(POLICE, 10, "bold"))
        self.lbl_vocal_statut.pack(anchor="w")

        self.lbl_vocal_dernier = tk.Label(
            c, text="", bg=CARD, fg=TEXTE,
            font=(POLICE, 10, "italic"), wraplength=560,
            justify="left", anchor="w")
        self.lbl_vocal_dernier.pack(anchor="w", fill="x", pady=(2, 8))

        # Liste des commandes (aide-mémoire)
        aide = (
            "🚨 Urgence : « stop » | « arrêt » | « urgence »\n"
            "🤖 Robot : « connecte le robot » | « déconnecte le robot »\n"
            "▶ Modes : « lance le mode » | « arrête le mode »\n"
            "🦾 Pince : « ouvre la pince » | « ferme la pince »\n"
            "🏠 Home : « position initiale » | « retour home »"
        )
        tk.Label(c, text=aide, bg=CARD, fg=GRIS,
                 font=(POLICE, 9), justify="left",
                 anchor="w").pack(anchor="w", fill="x")

    def _toggle_vocal(self):
        if self.vocal.est_actif():
            self.vocal.arreter()
            self.btn_vocal.configure(text="🎤 Activer l'écoute", bg=BLEU)
            self.lbl_vocal_statut.configure(
                text="Écoute désactivée.", fg=GRIS)
            self.log("Commande vocale : écoute arrêtée.")
            return

        # Démarrage (peut prendre 1-2s pour charger le modèle)
        self.lbl_vocal_statut.configure(
            text="Chargement du modèle…", fg=ORANGE)
        self.root.update_idletasks()
        ok, msg = self.vocal.demarrer()
        if not ok:
            self.lbl_vocal_statut.configure(
                text="Indisponible.", fg=ROUGE)
            messagebox.showerror("Commande vocale indisponible", msg)
            return
        self.btn_vocal.configure(text="🔴 Arrêter l'écoute", bg=ROUGE)
        self.lbl_vocal_statut.configure(
            text="🎤 Écoute active.", fg=VERT)
        self.log("Commande vocale : écoute active.")

    def _traiter_commande_vocale(self, texte):
        """Reçoit le texte reconnu (déjà dans le thread UI)."""
        # Messages d'erreur internes du listener
        if texte.startswith("__erreur__"):
            self.lbl_vocal_statut.configure(
                text="Erreur : %s" % texte.replace("__erreur__", "").strip(),
                fg=ROUGE)
            self.btn_vocal.configure(text="🎤 Activer l'écoute", bg=BLEU)
            return

        self.lbl_vocal_dernier.configure(text="Reçu : « %s »" % texte)
        self.log("Vocal : « %s »" % texte)

        # 1) URGENCE — garde anti-faux-positif : "urgence" passe toujours ;
        # les autres mots ne déclenchent que si la phrase est courte (≤ 2 mots).
        if "urgence" in texte:
            self.log("⛔ Commande vocale URGENCE.")
            self._declencher_urgence()
            return
        mots = texte.split()
        if len(mots) <= 2:
            for k in VOCAL_URGENCE_KEYWORDS:
                if k in mots:
                    self.log("⛔ Commande vocale URGENCE (« %s »)." % k)
                    self._declencher_urgence()
                    return

        # 2) Commandes normales — match par sous-chaîne, premier qui gagne
        for action, phrases in VOCAL_COMMANDES:
            for p in phrases:
                if p in texte:
                    self._executer_commande_vocale(action, p)
                    return

    def _executer_commande_vocale(self, action, phrase):
        """Dispatch d'une action vocale reconnue."""
        self.log("Vocal : action « %s » (déclenchée par « %s »)"
                 % (action, phrase))

        if action == "ouvre_pince":
            self.ouvrir_pince()
        elif action == "ferme_pince":
            self.fermer_pince()
        elif action == "home":
            # Cherche la position usine "initiale"
            for nom, angles in self._positions_usine:
                if "initiale" in nom.lower() or "home" in nom.lower():
                    self._aller_position(angles, "vocal: " + nom)
                    return
            self.log("Vocal : aucune position « home » trouvée.")
        elif action == "lancer_mode":
            self.lancer_mode()
        elif action == "pause_mode":
            self.arreter_mode()
        elif action == "connecter":
            self.connecter()
        elif action == "deconnecter":
            self.deconnecter()

    # ------------------------------------------------------------------ #
    #  ASSISTANT IA — Section UI                                          #
    # ------------------------------------------------------------------ #
    def _construire_assistant_ia(self):
        c = self.section(self.content,
                          "🧠 Assistant IA — Pilotage par langage naturel")

        tk.Label(c,
                 text="Décrivez en français l'objet à saisir et la zone où "
                      "le déposer. Le LLM identifie les deux, vous validez, "
                      "puis le robot lance le pick & place.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10), wraplength=560,
                 justify="left").pack(anchor="w", pady=(0, 8))

        # Champ de commande
        tk.Label(c, text="Commande :",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w")
        ligne_cmd = tk.Frame(c, bg=CARD)
        ligne_cmd.pack(fill="x")
        self.entry_ia = tk.Entry(ligne_cmd, font=(POLICE, 12),
                                  relief="flat", bg=FOND, bd=4)
        self.entry_ia.pack(side="left", expand=True, fill="x",
                            ipady=6, padx=(0, 6))
        self.entry_ia.bind("<Return>", lambda _e: self._ia_planifier())
        self.flat_btn(ligne_cmd, "🎤 Dicter",
                      self._ia_dicter, bg=GRIS, fs=10,
                      pady=6, padx=10).pack(side="left", padx=(0, 4))
        self.flat_btn(ligne_cmd, "Planifier",
                      self._ia_planifier, bg=BLEU, fs=11,
                      pady=6, padx=14).pack(side="left")

        # Exemples cliquables
        ex = tk.Frame(c, bg=CARD)
        ex.pack(fill="x", pady=(6, 4))
        tk.Label(ex, text="Exemples :", bg=CARD, fg=GRIS,
                 font=(POLICE, 9)).pack(side="left")
        for exemple in (
                "Attrape la roulette verte et pose-la dans la boîte",
                "Mets la roulette verte sur le tapis",
                "Range la roulette verte dans la boîte"):
            b = tk.Label(ex, text="« %s »"
                          % (exemple[:35] + "…" if len(exemple) > 35
                             else exemple),
                          bg=FOND, fg=BLEU,
                          font=(POLICE, 8, "italic"),
                          cursor="hand2", padx=4, pady=2)
            b.pack(side="left", padx=2)
            b.bind("<Button-1>",
                   lambda _e, t=exemple: (self.entry_ia.delete(0, tk.END),
                                            self.entry_ia.insert(0, t)))

        # Zone d'affichage de l'interprétation
        tk.Label(c, text="Interprétation :",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w", pady=(8, 2))

        cadre = tk.Frame(c, bg=CARD)
        cadre.pack(fill="x")
        scroll = ttk.Scrollbar(cadre, orient="vertical")
        self.text_ia_plan = tk.Text(cadre, height=10, font=(POLICE, 10),
                                     bg="#fafbfc", fg=TEXTE,
                                     relief="flat", bd=4, wrap="word",
                                     yscrollcommand=scroll.set,
                                     state="disabled")
        scroll.config(command=self.text_ia_plan.yview)
        scroll.pack(side="right", fill="y")
        self.text_ia_plan.pack(side="left", fill="both", expand=True)
        self.text_ia_plan.tag_configure("titre",
                                        font=(POLICE, 10, "bold"),
                                        foreground=BLEU)
        self.text_ia_plan.tag_configure("warn", foreground=ROUGE)
        self.text_ia_plan.tag_configure("ok", foreground=VERT)
        self.text_ia_plan.tag_configure("step",
                                        font=(POLICE, 10),
                                        foreground=TEXTE)

        # Boutons d'exécution
        ligne_btn = tk.Frame(c, bg=CARD)
        ligne_btn.pack(fill="x", pady=(8, 0))
        self.btn_ia_exec = self.flat_btn(
            ligne_btn, "▶ Lancer le pick & place",
            self._ia_executer, bg=VERT, fs=12, state="disabled")
        self.btn_ia_exec.pack(side="left", expand=True, fill="x",
                               padx=(0, 4))
        self.btn_ia_stop = self.flat_btn(
            ligne_btn, "■ Annuler",
            self._ia_annuler, bg=ROUGE, fs=12, state="disabled")
        self.btn_ia_stop.pack(side="left", expand=True, fill="x",
                               padx=(4, 0))

        # Statut
        self.lbl_ia_statut = tk.Label(c, text="Prêt.",
                                       bg=CARD, fg=GRIS,
                                       font=(POLICE, 10))
        self.lbl_ia_statut.pack(anchor="w", pady=(6, 0))

        # Affichage des ressources connues
        self.lbl_ia_resources = tk.Label(
            c, text="", bg=CARD, fg=GRIS,
            font=(POLICE, 9, "italic"),
            wraplength=560, justify="left", anchor="w")
        self.lbl_ia_resources.pack(anchor="w", fill="x", pady=(4, 0))
        self._ia_maj_resources()

    def _ia_maj_resources(self):
        zones = self.zones.lister()
        objets = self.catalogue.lister()
        positions = ([n for n, _ in self._positions_usine]
                     + list(self.positions_perso.keys()))
        txt = ("📍 Zones : %s\n"
               "🎯 Objets YOLO : %s\n"
               "🤖 Positions : %s"
               % (", ".join(zones) or "(aucune — créez-en !)",
                  ", ".join(objets),
                  ", ".join(positions)))
        self.lbl_ia_resources.configure(text=txt)

    def _ia_dicter(self):
        """Active Vosk pour dicter une commande dans le champ texte."""
        if not self.vocal.est_actif():
            self._toggle_vocal()
            if not self.vocal.est_actif():
                return
        messagebox.showinfo(
            "Dictée activée",
            "Parlez votre commande maintenant. Elle apparaîtra dans le "
            "champ texte via la commande vocale active.\n\n"
            "Cliquez « Planifier » ensuite.")

    def _ia_planifier(self):
        commande = self.entry_ia.get().strip()
        if not commande:
            return
        if self._ia_en_cours:
            messagebox.showwarning(
                "Exécution en cours",
                "Annulez l'exécution courante avant de planifier.")
            return

        # Vérifie qu'Ollama est dispo
        ok, msg = self.llm.disponible()
        if not ok:
            messagebox.showerror("Ollama indisponible", msg)
            return

        self.lbl_ia_statut.configure(
            text="Analyse de la commande (LLM)…", fg=ORANGE)
        self._afficher_plan_texte("Analyse de la commande en cours…\n", "step")
        self.btn_ia_exec.configure(state="disabled")
        self._ia_cmd_courante = None
        self.root.update_idletasks()

        def _worker():
            try:
                extraction, brut = self.llm.planifier(commande)
                self.root.after(0, self._ia_plan_recu, commande, extraction)
            except Exception as exc:
                self.root.after(0, self._ia_plan_erreur, str(exc))

        Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _resoudre_nom(valeur, candidats):
        """Associe `valeur` à un nom connu : exact, sinon sous-chaîne,
        sinon un mot significatif en commun. Renvoie None si rien."""
        v = (valeur or "").strip().lower()
        if not v:
            return None
        for c in candidats:
            if c.lower() == v:
                return c
        for c in candidats:
            cl = c.lower()
            if v in cl or cl in v:
                return c
        mots_v = {m for m in v.split() if len(m) > 2}
        for c in candidats:
            mots_c = {m for m in c.lower().split() if len(m) > 2}
            if mots_v & mots_c:
                return c
        return None

    def _ia_plan_recu(self, commande, extraction):
        """Résout cible+place, affiche l'aperçu, prépare la commande launch."""
        self._ia_cmd_courante = None

        # Aperçu : on repart d'une zone de texte vierge.
        self.text_ia_plan.configure(state="normal")
        self.text_ia_plan.delete("1.0", tk.END)
        self.text_ia_plan.configure(state="disabled")
        self._afficher_plan_texte("Commande : « %s »\n\n" % commande, "titre")
        comp = (extraction.get("comprehension") or "").strip()
        if comp:
            self._afficher_plan_texte("Compréhension : %s\n" % comp, "step")

        # Le LLM n'extrait que 2 champs ; l'IHM les rattache aux
        # ressources réellement connues (résolution tolérante).
        cible_brut = (extraction.get("cible") or "").strip()
        place_brut = (extraction.get("place") or "").strip()
        objet = self._resoudre_nom(cible_brut, self.catalogue.lister())
        zone  = self._resoudre_nom(place_brut, self.zones.lister())

        problemes = []
        if not objet:
            problemes.append(
                "Objet à saisir non reconnu : « %s ». Objets connus : %s."
                % (cible_brut or "(vide)",
                   ", ".join(self.catalogue.lister()) or "(aucun)"))
        if not zone:
            problemes.append(
                "Zone de dépôt non reconnue : « %s ». Zones connues : %s."
                % (place_brut or "(vide)",
                   ", ".join(self.zones.lister()) or "(aucune)"))

        if problemes:
            self._afficher_plan_texte("\n⚠ Problèmes détectés :\n", "warn")
            for p in problemes:
                self._afficher_plan_texte("  - " + p + "\n", "warn")
            self.lbl_ia_statut.configure(
                text="Commande incomprise — précisez l'objet et la zone.",
                fg=ROUGE)
            self.btn_ia_exec.configure(state="disabled")
            return

        # cible → classe YOLO ; place → coordonnées de dépôt.
        classe = self.catalogue.objets.get(objet, {}).get("classe_yolo", "")
        z = self.zones.get(zone)
        if not classe:
            self._afficher_plan_texte(
                "\n⚠ L'objet « %s » n'a pas de classe YOLO renseignée "
                "dans le catalogue.\n" % objet, "warn")
            self.lbl_ia_statut.configure(
                text="Classe YOLO manquante pour cet objet.", fg=ROUGE)
            self.btn_ia_exec.configure(state="disabled")
            return

        commande_launch = (
            "ros2 launch mon_controleur pick_place_complet.launch.py "
            "target_class:=%s place_x:=%.3f place_y:=%.3f place_z:=%.3f"
            % (shlex.quote(classe), z["x"], z["y"], z["z"]))
        self._ia_cmd_courante = commande_launch

        self._afficher_plan_texte(
            "\n✔ Objet à saisir : « %s »  (classe YOLO : %s)\n"
            % (objet, classe), "ok")
        self._afficher_plan_texte(
            "✔ Zone de dépôt : « %s »  → x=%.2f y=%.2f z=%.2f m\n"
            % (zone, z["x"], z["y"], z["z"]), "ok")
        self._afficher_plan_texte(
            "\nLaunch qui sera exécuté :\n", "titre")
        self._afficher_plan_texte("  " + commande_launch + "\n", "step")

        self.lbl_ia_statut.configure(
            text="Prêt. Cliquez « ▶ Lancer le pick & place » pour exécuter.",
            fg=VERT)
        self.btn_ia_exec.configure(state="normal")
        self.log("IA : commande comprise — objet=%s classe=%s zone=%s"
                 % (objet, classe, zone))

    def _ia_plan_erreur(self, msg):
        self.lbl_ia_statut.configure(text="Erreur LLM.", fg=ROUGE)
        self._afficher_plan_texte("\n❌ Erreur : " + msg, "warn")
        self.log("IA : erreur — %s" % msg)

    def _afficher_plan_texte(self, texte, tag=None):
        self.text_ia_plan.configure(state="normal")
        if tag:
            self.text_ia_plan.insert(tk.END, texte, tag)
        else:
            self.text_ia_plan.insert(tk.END, texte)
        self.text_ia_plan.see(tk.END)
        self.text_ia_plan.configure(state="disabled")

    def _ia_executer(self):
        """Lance le pick & place via le launch déjà déterminé."""
        if not self._ia_cmd_courante:
            return
        if self._ia_en_cours:
            return
        if self.mode_proc is not None and self.mode_proc.poll() is None:
            messagebox.showwarning(
                "Mode en cours",
                "Arrêtez le mode en cours avant de lancer le pick & place IA.")
            return
        if not self.connecte:
            messagebox.showwarning(
                "Connexion requise",
                "Connectez d'abord le robot (MoveIt).")
            return

        self.log("IA : lancement — %s" % self._ia_cmd_courante)
        self.mode_proc = self._lancer_processus(self._ia_cmd_courante)
        if self.mode_proc is None:
            self.lbl_ia_statut.configure(
                text="Échec du lancement du launch.", fg=ROUGE)
            return

        self._ia_en_cours = True
        self.mode_courant = "Assistant IA"
        self.lbl_mode.configure(text="Mode : Assistant IA")
        self.lbl_mode_statut.configure(text="Pick & place IA en cours...")
        self.btn_ia_exec.configure(state="disabled")
        self.btn_ia_stop.configure(state="normal")
        self.lbl_ia_statut.configure(
            text="Pick & place IA en cours…", fg=BLEU)

        Thread(target=self._lecteur_stdout,
               args=(self.mode_proc, "[IA]"), daemon=True).start()
        Thread(target=self._ia_surveiller, daemon=True).start()
        self._maj_etat_boutons()

    def _ia_annuler(self):
        """Arrête le launch pick & place IA en cours."""
        if self.mode_proc is None or self.mode_proc.poll() is not None:
            return
        self.lbl_ia_statut.configure(text="Arrêt en cours…", fg=ORANGE)
        self.log("IA : arrêt demandé.")
        proc = self.mode_proc
        Thread(target=lambda: self._arreter_processus(proc, "IA"),
               daemon=True).start()

    def _ia_surveiller(self):
        """Attend la fin du launch puis réinitialise l'onglet IA."""
        proc = self.mode_proc
        proc.wait()
        if self.mode_proc is proc:
            self.mode_proc = None
        self.root.after(0, self._ia_fin_execution)

    def _ia_fin_execution(self):
        self._ia_en_cours = False
        self.btn_ia_exec.configure(state="normal")
        self.btn_ia_stop.configure(state="disabled")
        self.lbl_ia_statut.configure(
            text="Pick & place IA terminé.", fg=VERT)
        if self.mode_courant == "Assistant IA":
            self.mode_courant = None
            self.lbl_mode.configure(text="Mode : —")
            self.lbl_mode_statut.configure(text="Mode terminé.")
        self.log("IA : exécution terminée.")
        self._maj_etat_boutons()


    # ------------------------------------------------------------------ #
    #  SECTION ZONES DE TRAVAIL                                           #
    # ------------------------------------------------------------------ #
    def _construire_zones(self):
        c = self.section(self.content, "📍 Zones de travail (coordonnées XYZ)")

        tk.Label(c,
                 text="Mémorise des emplacements (« la boîte », « le tapis »…) "
                      "que l'Assistant IA peut référencer.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10), wraplength=560,
                 justify="left").pack(anchor="w", pady=(0, 8))

        # Liste des zones
        self.frame_zones = tk.Frame(c, bg=CARD)
        self.frame_zones.pack(fill="x")
        self._refresh_zones_ui()

        # Ajout d'une zone à partir de la position courante (lecture XYZ)
        sep = tk.Frame(c, bg=BORDURE, height=1)
        sep.pack(fill="x", pady=8)

        tk.Label(c, text="Nouvelle zone (saisie manuelle)",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 10, "bold")).pack(anchor="w")

        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x", pady=4)
        self.entry_zone_nom = tk.Entry(ligne, font=(POLICE, 11),
                                        relief="flat", bg=FOND, bd=4)
        self.entry_zone_nom.insert(0, "nom de la zone")
        self.entry_zone_nom.pack(side="left", expand=True, fill="x",
                                  ipady=4, padx=(0, 4))

        def _mini(libelle):
            tk.Label(ligne, text=libelle, bg=CARD, fg=GRIS,
                     font=(POLICE, 10)).pack(side="left", padx=(4, 2))
            e = tk.Entry(ligne, font=(POLICE, 11), width=6,
                          relief="flat", bg=FOND, bd=4, justify="center")
            e.pack(side="left")
            return e

        self.entry_zone_x = _mini("x (m)")
        self.entry_zone_y = _mini("y (m)")
        self.entry_zone_z = _mini("z (m)")
        for e in (self.entry_zone_x, self.entry_zone_y, self.entry_zone_z):
            e.insert(0, "0.00")

        ligne_btn = tk.Frame(c, bg=CARD)
        ligne_btn.pack(fill="x", pady=(6, 0))
        self.flat_btn(ligne_btn, "+ Enregistrer la zone",
                      self._zone_ajouter, bg=VERT, fs=11).pack(
            side="left", expand=True, fill="x", padx=(0, 4))
        self.flat_btn(ligne_btn,
                      "📡 Lire la dernière détection",
                      self._zone_depuis_detection, bg=GRIS, fs=10).pack(
            side="left", expand=True, fill="x", padx=(4, 0))

    def _refresh_zones_ui(self):
        for w in self.frame_zones.winfo_children():
            w.destroy()
        zones = self.zones.lister()
        if not zones:
            tk.Label(self.frame_zones,
                     text="(aucune zone enregistrée)",
                     bg=CARD, fg=GRIS,
                     font=(POLICE, 10, "italic")).pack(anchor="w")
            return
        for nom in zones:
            z = self.zones.get(nom)
            ligne = tk.Frame(self.frame_zones, bg=CARD)
            ligne.pack(fill="x", pady=2)
            tk.Label(ligne,
                     text="📍 %s" % nom,
                     bg=CARD, fg=TEXTE,
                     font=(POLICE, 11, "bold"),
                     anchor="w").pack(side="left", padx=(0, 8))
            tk.Label(ligne,
                     text="x=%.2f  y=%.2f  z=%.2f m"
                          % (z["x"], z["y"], z["z"]),
                     bg=CARD, fg=GRIS,
                     font=(POLICE, 10)).pack(side="left")
            tk.Button(ligne, text="🗑", bg=CARD, fg=ROUGE,
                       relief="flat", bd=0, cursor="hand2",
                       font=(POLICE, 11),
                       command=lambda n=nom:
                       self._zone_supprimer(n)).pack(side="right")

    def _zone_ajouter(self):
        nom = self.entry_zone_nom.get().strip()
        if not nom or nom == "nom de la zone":
            messagebox.showwarning("Nom requis", "Donnez un nom à la zone.")
            return
        try:
            x = float(self.entry_zone_x.get())
            y = float(self.entry_zone_y.get())
            z = float(self.entry_zone_z.get())
        except ValueError:
            messagebox.showerror("Valeur invalide",
                                 "x, y, z doivent être numériques (mètres).")
            return
        self.zones.ajouter(nom, x, y, z)
        self.entry_zone_nom.delete(0, tk.END)
        self._refresh_zones_ui()
        self._ia_maj_resources()
        self.log("Zone enregistrée : %s = (%.3f, %.3f, %.3f)m"
                 % (nom, x, y, z))

    def _zone_supprimer(self, nom):
        if not messagebox.askyesno("Supprimer ?",
                                    "Supprimer la zone « %s » ?" % nom):
            return
        self.zones.supprimer(nom)
        self._refresh_zones_ui()
        self._ia_maj_resources()

    def _zone_depuis_detection(self):
        """Pré-remplit x/y/z avec la dernière détection YOLO."""
        obj = robot_node.get_last_object(max_age_s=60.0)
        if obj is None:
            messagebox.showinfo(
                "Aucune détection",
                "Aucune détection récente sur /object_position_in_robot.\n"
                "Lancez d'abord la perception YOLO.")
            return
        self.entry_zone_x.delete(0, tk.END)
        self.entry_zone_x.insert(0, "%.3f" % obj[0])
        self.entry_zone_y.delete(0, tk.END)
        self.entry_zone_y.insert(0, "%.3f" % obj[1])
        self.entry_zone_z.delete(0, tk.END)
        self.entry_zone_z.insert(0, "%.3f" % obj[2])
        self.log("Zone : XYZ pré-remplis depuis la dernière détection.")

    # ------------------------------------------------------------------ #
    #  SECTION CATALOGUE D'OBJETS DÉTECTABLES                             #
    # ------------------------------------------------------------------ #
    def _construire_catalogue(self):
        c = self.section(self.content,
                          "🎯 Catalogue d'objets détectables par YOLO")

        tk.Label(c,
                 text="Liste des objets que la perception YOLO est entraînée "
                      "à reconnaître. L'Assistant IA s'en sert pour valider "
                      "les demandes.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10), wraplength=560,
                 justify="left").pack(anchor="w", pady=(0, 8))

        self.frame_catalogue = tk.Frame(c, bg=CARD)
        self.frame_catalogue.pack(fill="x")
        self._refresh_catalogue_ui()

        sep = tk.Frame(c, bg=BORDURE, height=1)
        sep.pack(fill="x", pady=8)

        tk.Label(c, text="Ajouter un objet",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 10, "bold")).pack(anchor="w")

        grille = tk.Frame(c, bg=CARD)
        grille.pack(fill="x", pady=4)

        def _champ(parent, libelle, width=15):
            tk.Label(parent, text=libelle, bg=CARD, fg=GRIS,
                     font=(POLICE, 9)).pack(anchor="w")
            e = tk.Entry(parent, font=(POLICE, 11), width=width,
                          relief="flat", bg=FOND, bd=4)
            e.pack(fill="x", ipady=3)
            return e

        self.entry_obj_nom = _champ(grille,
                                     "Nom (utilisé par le LLM) :")
        self.entry_obj_yolo = _champ(grille,
                                      "Classe YOLO (interne) :")
        self.entry_obj_couleur = _champ(grille,
                                         "Couleur (optionnel) :")

        self.flat_btn(c, "+ Ajouter au catalogue",
                      self._catalogue_ajouter, bg=VERT, fs=11).pack(
            fill="x", pady=(6, 0))

    def _refresh_catalogue_ui(self):
        for w in self.frame_catalogue.winfo_children():
            w.destroy()
        for nom in self.catalogue.lister():
            o = self.catalogue.objets[nom]
            ligne = tk.Frame(self.frame_catalogue, bg=CARD)
            ligne.pack(fill="x", pady=2)
            tk.Label(ligne, text="🎯 %s" % nom,
                     bg=CARD, fg=TEXTE,
                     font=(POLICE, 11, "bold")).pack(side="left",
                                                       padx=(0, 8))
            tk.Label(ligne,
                     text="(classe yolo : %s)"
                          % o.get("classe_yolo", "?"),
                     bg=CARD, fg=GRIS,
                     font=(POLICE, 10)).pack(side="left")
            tk.Button(ligne, text="🗑", bg=CARD, fg=ROUGE,
                       relief="flat", bd=0, cursor="hand2",
                       font=(POLICE, 11),
                       command=lambda n=nom:
                       self._catalogue_supprimer(n)).pack(side="right")

    def _catalogue_ajouter(self):
        nom = self.entry_obj_nom.get().strip()
        yolo = self.entry_obj_yolo.get().strip()
        couleur = self.entry_obj_couleur.get().strip()
        if not nom or not yolo:
            messagebox.showwarning(
                "Champs requis",
                "Le nom et la classe YOLO sont obligatoires.")
            return
        self.catalogue.ajouter(nom, yolo, couleur, "")
        self.entry_obj_nom.delete(0, tk.END)
        self.entry_obj_yolo.delete(0, tk.END)
        self.entry_obj_couleur.delete(0, tk.END)
        self._refresh_catalogue_ui()
        self._ia_maj_resources()
        self.log("Catalogue : objet « %s » ajouté." % nom)

    def _catalogue_supprimer(self, nom):
        if not messagebox.askyesno(
                "Supprimer ?",
                "Retirer « %s » du catalogue ?" % nom):
            return
        self.catalogue.supprimer(nom)
        self._refresh_catalogue_ui()
        self._ia_maj_resources()

    # ------------------------------------------------------------------ #
    #  SECTION RAPPORTS & DIAGNOSTICS                                     #
    # ------------------------------------------------------------------ #
    def _construire_rapports(self):
        c = self.section(self.content, "Rapports & diagnostics")

        tk.Label(c,
                 text="Suivi : cycles, fatigue mécanique, erreurs, répétabilité.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10)).pack(anchor="w", pady=(0, 8))

        # Cartes synthèse (rafraîchies périodiquement)
        cards = tk.Frame(c, bg=CARD)
        cards.pack(fill="x", pady=(0, 8))

        def _mini_card(parent, titre):
            box = tk.Frame(parent, bg=FOND, highlightbackground=BORDURE,
                            highlightthickness=1)
            box.pack(side="left", expand=True, fill="x", padx=2)
            tk.Label(box, text=titre, bg=FOND, fg=GRIS,
                     font=(POLICE, 9)).pack(pady=(6, 0))
            val = tk.Label(box, text="—", bg=FOND, fg=BLEU,
                           font=(POLICE, 14, "bold"))
            val.pack(pady=(0, 6))
            return val

        self.lbl_card_jour   = _mini_card(cards, "Aujourd'hui")
        self.lbl_card_total  = _mini_card(cards, "Total")
        self.lbl_card_heures = _mini_card(cards, "Fonct.")
        self.lbl_card_taux   = _mini_card(cards, "Succès")

        # Test de répétabilité
        sep1 = tk.Frame(c, bg=BORDURE, height=1)
        sep1.pack(fill="x", pady=8)

        tk.Label(c, text="Test de répétabilité",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w")
        tk.Label(c,
                 text="Aller-retours entre deux positions ; calcule "
                      "l'écart-type des positions atteintes.",
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 9)).pack(anchor="w", pady=(0, 4))

        self.lbl_repet_statut = tk.Label(
            c, text="Aucun test en cours.", bg=CARD, fg=GRIS,
            font=(POLICE, 9))
        self.lbl_repet_statut.pack(anchor="w")

        self.btn_repet = self.flat_btn(
            c, "▶ Lancer test de répétabilité", self._dialog_repetabilite,
            bg=BLEU, fs=11)
        self.btn_repet.pack(fill="x", pady=(6, 8))

        # Génération de rapport
        sep2 = tk.Frame(c, bg=BORDURE, height=1)
        sep2.pack(fill="x", pady=4)

        tk.Label(c, text="Rapport d'exploitation",
                 bg=CARD, fg=TEXTE,
                 font=(POLICE, 11, "bold")).pack(anchor="w", pady=(4, 0))

        ligne = tk.Frame(c, bg=CARD)
        ligne.pack(fill="x", pady=6)
        self.flat_btn(ligne, "📄 Générer & ouvrir",
                      self._generer_et_ouvrir_rapport,
                      bg=VERT, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(0, 4))
        self.flat_btn(ligne, "✉ Envoyer par email",
                      self._envoyer_rapport_email,
                      bg=BLEU, fs=11).pack(side="left", expand=True,
                                            fill="x", padx=(4, 0))

        self.flat_btn(c, "⚙ Configurer l'email…",
                      self._dialog_config_email,
                      bg=GRIS, fs=10, pady=8).pack(fill="x")

        self.lbl_rapport_info = tk.Label(
            c, text="Rapports dans : %s" % RAPPORTS_DIR,
            bg=CARD, fg=GRIS, font=(POLICE, 9, "italic"))
        self.lbl_rapport_info.pack(anchor="w", pady=(6, 0))

        # Rafraîchit régulièrement les cartes
        self._boucle_cards_rapports()

    def _boucle_cards_rapports(self):
        try:
            d = self.stats.data
            total = d["cycles"]["total"]
            echecs = d["cycles"]["echecs"]
            taux = (100.0 * (total - echecs) / total) if total > 0 else 0.0
            self.lbl_card_jour.configure(text=str(self.stats.cycles_jour()))
            self.lbl_card_total.configure(text=str(total))
            self.lbl_card_heures.configure(
                text=_fmt_duree(self.stats.temps_total_s()))
            self.lbl_card_taux.configure(text="%.0f%%" % taux)
        except Exception:
            pass
        self.root.after(5000, self._boucle_cards_rapports)

    # ------------------------------------------------------------------ #
    #  TEST DE RÉPÉTABILITÉ                                               #
    # ------------------------------------------------------------------ #
    def _dialog_repetabilite(self):
        """Petite boîte de dialogue : choix de 2 positions + N cycles."""
        if self._repet_en_cours:
            self.lbl_repet_statut.configure(
                text="Test déjà en cours…", fg=ORANGE)
            return
        if not self.connecte and not self.mode_simulation:
            messagebox.showwarning("Connexion requise",
                                   "Connectez d'abord le robot (MoveIt).")
            return

        # Collecte des positions disponibles (usine + perso)
        positions = dict(self._positions_usine)
        positions.update(self.positions_perso)
        if len(positions) < 2:
            messagebox.showwarning(
                "Positions insuffisantes",
                "Il faut au moins 2 positions enregistrées pour un test.")
            return

        # Dialog modal Tkinter
        dlg = tk.Toplevel(self.root)
        dlg.title("Test de répétabilité")
        dlg.configure(bg=FOND)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="Test de répétabilité",
                 bg=FOND, fg=TEXTE,
                 font=(POLICE, 13, "bold")).pack(padx=20, pady=(16, 4))
        tk.Label(dlg, text="Le robot effectuera N aller-retours entre A et B.",
                 bg=FOND, fg=GRIS,
                 font=(POLICE, 10)).pack(padx=20, pady=(0, 12))

        noms = sorted(positions.keys())

        def _ligne(parent, libelle):
            tk.Label(parent, text=libelle, bg=FOND, fg=TEXTE,
                     font=(POLICE, 10, "bold")).pack(anchor="w", padx=20)

        _ligne(dlg, "Position A")
        var_a = tk.StringVar(value=noms[0])
        ttk.Combobox(dlg, textvariable=var_a, values=noms,
                     state="readonly", font=(POLICE, 11)).pack(
            fill="x", padx=20, pady=(0, 8))

        _ligne(dlg, "Position B")
        var_b = tk.StringVar(value=noms[1] if len(noms) > 1 else noms[0])
        ttk.Combobox(dlg, textvariable=var_b, values=noms,
                     state="readonly", font=(POLICE, 11)).pack(
            fill="x", padx=20, pady=(0, 8))

        _ligne(dlg, "Nombre de cycles A↔B")
        var_n = tk.IntVar(value=5)
        tk.Scale(dlg, from_=2, to=20, orient="horizontal",
                 variable=var_n, bg=FOND, length=240,
                 highlightthickness=0).pack(padx=20, pady=(0, 8))

        _ligne(dlg, "Stabilisation par mouvement (secondes)")
        var_t = tk.IntVar(value=3)
        tk.Scale(dlg, from_=1, to=8, orient="horizontal",
                 variable=var_t, bg=FOND, length=240,
                 highlightthickness=0).pack(padx=20, pady=(0, 12))

        btns = tk.Frame(dlg, bg=FOND)
        btns.pack(fill="x", padx=20, pady=(0, 16))

        def _lancer():
            a, b = var_a.get(), var_b.get()
            if a == b:
                messagebox.showwarning("A = B",
                                       "Choisissez deux positions différentes.",
                                       parent=dlg)
                return
            dlg.destroy()
            self._lancer_repetabilite(
                a, positions[a], b, positions[b],
                var_n.get(), var_t.get())

        tk.Button(btns, text="Annuler", command=dlg.destroy,
                  bg=GRIS, fg="white", relief="flat",
                  font=(POLICE, 11), padx=20, pady=8).pack(side="left")
        tk.Button(btns, text="▶ Démarrer", command=_lancer,
                  bg=VERT, fg="white", relief="flat",
                  font=(POLICE, 11, "bold"), padx=20, pady=8).pack(
            side="right")

    def _lancer_repetabilite(self, nom_a, ang_a, nom_b, ang_b, n, t_stab):
        """Démarre le test en arrière-plan."""
        self._repet_en_cours = True
        self._repet_annule = False
        self.btn_repet.configure(text="■ Annuler le test",
                                  command=self._annuler_repetabilite,
                                  bg=ROUGE)
        self.lbl_repet_statut.configure(
            text="Test en cours : 0/%d…" % n, fg=BLEU)
        self.log("Répétabilité : %d cycles entre « %s » et « %s »…"
                 % (n, nom_a, nom_b))

        def _worker():
            import time as _t
            mesures_a, mesures_b = [], []
            try:
                for i in range(n):
                    if self._repet_annule:
                        break
                    # → A
                    envoyer_trajectoire(ang_a, temps=self.vitesse_traj)
                    self.stats.joint_mouvement()
                    _t.sleep(self.vitesse_traj + t_stab)
                    pos = robot_node.get_positions_deg()
                    if pos is not None:
                        mesures_a.append(pos)
                    # → B
                    if self._repet_annule:
                        break
                    envoyer_trajectoire(ang_b, temps=self.vitesse_traj)
                    self.stats.joint_mouvement()
                    _t.sleep(self.vitesse_traj + t_stab)
                    pos = robot_node.get_positions_deg()
                    if pos is not None:
                        mesures_b.append(pos)
                    self.root.after(0, lambda k=i+1:
                                    self.lbl_repet_statut.configure(
                                        text="Test en cours : %d/%d…" % (k, n)))
            finally:
                self.root.after(0, self._fin_repetabilite,
                                nom_a, mesures_a, nom_b, mesures_b, n)

        Thread(target=_worker, daemon=True).start()

    def _annuler_repetabilite(self):
        self._repet_annule = True
        self.lbl_repet_statut.configure(text="Annulation…", fg=ORANGE)

    def _fin_repetabilite(self, nom_a, mes_a, nom_b, mes_b, n_demande):
        self._repet_en_cours = False
        self.btn_repet.configure(text="▶ Lancer test de répétabilité",
                                  command=self._dialog_repetabilite,
                                  bg=BLEU)

        if self._repet_annule and not mes_a and not mes_b:
            self.lbl_repet_statut.configure(text="Test annulé.", fg=GRIS)
            self.log("Répétabilité : test annulé.")
            return

        # Écart-type par joint, pour A puis B
        def _ecarts(mesures):
            if len(mesures) < 2:
                return [0.0] * 6
            return [statistics.stdev(m[i] for m in mesures) for i in range(6)]

        std_a = _ecarts(mes_a)
        std_b = _ecarts(mes_b)
        max_std = max(std_a + std_b) if (std_a or std_b) else 0.0

        resultat = {
            "ts": datetime.now().isoformat(timespec='seconds'),
            "position": "%s ↔ %s" % (nom_a, nom_b),
            "n": min(len(mes_a), len(mes_b)),
            "n_demande": n_demande,
            "ecarts_std_deg": [max(a, b) for a, b in zip(std_a, std_b)],
            "std_par_position": {nom_a: std_a, nom_b: std_b},
        }
        self.stats.ajouter_repetabilite(resultat)

        verdict = ("OK ✓" if max_std <= SEUIL_REPETABILITE_OK_DEG
                   else "À surveiller ⚠")
        self.lbl_repet_statut.configure(
            text="Test terminé — écart max %.3f° — %s" % (max_std, verdict),
            fg=VERT if max_std <= SEUIL_REPETABILITE_OK_DEG else ORANGE)
        self.log("Répétabilité : %d/%d cycles — écart-type max %.3f° (%s)"
                 % (resultat["n"], n_demande, max_std, verdict))

    # ------------------------------------------------------------------ #
    #  GÉNÉRATION & ENVOI DU RAPPORT                                      #
    # ------------------------------------------------------------------ #
    def _generer_et_ouvrir_rapport(self):
        try:
            chemin = self._generer_rapport()
            webbrowser.open(chemin.as_uri())
            self.log("Rapport généré → %s" % chemin)
        except Exception as exc:
            self.log("Rapport : erreur — %s" % exc)
            messagebox.showerror("Génération impossible", str(exc))

    def _generer_rapport(self):
        nom = "rapport_%s.html" % datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin = RAPPORTS_DIR / nom
        return generer_rapport_html(self.stats, chemin)

    def _parse_destinataires(self, brut):
        """Découpe une chaîne de destinataires (virgule ou point-virgule)."""
        if not brut:
            return []
        morceaux = re.split(r'[,;]', brut)
        return [m.strip() for m in morceaux if m.strip()]

    def _envoyer_rapport_email(self):
        cfg = self._charger_config_email()
        if cfg is None:
            if messagebox.askyesno(
                    "Configuration manquante",
                    "Aucune configuration email enregistrée.\n"
                    "Voulez-vous la configurer maintenant ?"):
                self._dialog_config_email()
            return

        destinataires = self._parse_destinataires(cfg.get("destinataire", ""))
        if not destinataires:
            messagebox.showerror(
                "Destinataire manquant",
                "Aucun destinataire valide dans la configuration.")
            return

        # Génère le rapport
        try:
            chemin = self._generer_rapport()
        except Exception as exc:
            messagebox.showerror("Génération impossible", str(exc))
            return

        # Envoi en arrière-plan
        liste_aff = ", ".join(destinataires)
        self.log("Envoi du rapport par email à %s…" % liste_aff)

        def _worker():
            try:
                self._envoyer_mail(cfg, chemin, destinataires)
                self.root.after(0, lambda: messagebox.showinfo(
                    "Email envoyé",
                    "Rapport envoyé à :\n%s" % liste_aff))
                self.root.after(0, self.log,
                                "Email : envoi réussi (%d destinataire(s))."
                                % len(destinataires))
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: messagebox.showerror(
                    "Échec de l'envoi", msg))
                self.root.after(0, self.log,
                                "Email : échec — %s" % msg)

        Thread(target=_worker, daemon=True).start()

    def _envoyer_mail(self, cfg, chemin_html, destinataires=None):
        """Construit et envoie le mail via SMTP (TLS).

        `destinataires` : liste d'adresses. Si None, on parse cfg["destinataire"].
        """
        if destinataires is None:
            destinataires = self._parse_destinataires(
                cfg.get("destinataire", ""))
        if not destinataires:
            raise ValueError("aucun destinataire")

        msg = EmailMessage()
        msg["Subject"] = ("[IGUS Rebel] Rapport d'exploitation — %s"
                          % datetime.now().strftime("%d/%m/%Y %H:%M"))
        msg["From"] = cfg["expediteur"]
        msg["To"] = ", ".join(destinataires)

        d = self.stats.data
        msg.set_content(
            "Bonjour,\n\n"
            "Veuillez trouver en pièce jointe le rapport d'exploitation\n"
            "du bras robot IGUS Rebel.\n\n"
            "Résumé :\n"
            " - Cycles aujourd'hui : %d\n"
            " - Cycles total       : %d\n"
            " - Échecs / Recoveries : %d / %d\n"
            " - Temps cumulé       : %s\n\n"
            "Le rapport HTML complet est en pièce jointe.\n\n"
            "Cordialement,\n"
            "%s\n%s — %s\n"
            % (self.stats.cycles_jour(),
               d["cycles"]["total"],
               d["cycles"]["echecs"], d["cycles"]["recoveries"],
               _fmt_duree(self.stats.temps_total_s()),
               AUTEUR_NOM, AUTEUR_FORMATION, AUTEUR_ECOLE))

        with open(chemin_html, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="text", subtype="html",
                            filename=chemin_html.name)

        ctx = ssl.create_default_context()
        host = cfg["smtp_host"]
        port = int(cfg.get("smtp_port", 587))
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as s:
                s.login(cfg["smtp_user"], cfg["smtp_password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(context=ctx)
                s.login(cfg["smtp_user"], cfg["smtp_password"])
                s.send_message(msg)

    # ---- Configuration SMTP ------------------------------------------ #
    def _charger_config_email(self):
        try:
            if EMAIL_CONFIG.exists():
                with open(EMAIL_CONFIG, 'r') as f:
                    cfg = json.load(f)
                # Validation minimale
                champs = ("smtp_host", "smtp_user", "smtp_password",
                          "expediteur", "destinataire")
                if all(cfg.get(c) for c in champs):
                    return cfg
        except Exception:
            pass
        return None

    def _sauver_config_email(self, cfg):
        try:
            USER_DIR.mkdir(parents=True, exist_ok=True)
            with open(EMAIL_CONFIG, 'w') as f:
                json.dump(cfg, f, indent=2)
            # Permissions restrictives (mot de passe SMTP)
            try:
                os.chmod(EMAIL_CONFIG, 0o600)
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Sauvegarde impossible", str(exc))

    def _dialog_config_email(self):
        """Boîte de dialogue de configuration SMTP."""
        cfg = self._charger_config_email() or {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "",
            "smtp_password": "",
            "expediteur": "",
            "destinataire": "",
        }

        dlg = tk.Toplevel(self.root)
        dlg.title("Configuration email (SMTP)")
        dlg.configure(bg=FOND)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="Configuration SMTP", bg=FOND, fg=TEXTE,
                 font=(POLICE, 13, "bold")).pack(padx=20, pady=(16, 4))
        tk.Label(dlg,
                 text="Pour Gmail : utilisez un « mot de passe d'application ».",
                 bg=FOND, fg=GRIS,
                 font=(POLICE, 9, "italic")).pack(padx=20, pady=(0, 12))

        entries = {}

        def _champ(libelle, cle, secret=False):
            tk.Label(dlg, text=libelle, bg=FOND, fg=TEXTE,
                     font=(POLICE, 10, "bold")).pack(anchor="w", padx=20)
            e = tk.Entry(dlg, font=(POLICE, 11), relief="flat", bg="white",
                         bd=4, show="•" if secret else "")
            e.insert(0, str(cfg.get(cle, "")))
            e.pack(fill="x", padx=20, pady=(0, 8), ipady=4)
            entries[cle] = e

        _champ("Serveur SMTP",       "smtp_host")
        _champ("Port (587 ou 465)",   "smtp_port")
        _champ("Utilisateur SMTP",    "smtp_user")
        _champ("Mot de passe SMTP",   "smtp_password", secret=True)
        _champ("Expéditeur (From)",   "expediteur")
        _champ("Destinataires (séparés par , ou ;)", "destinataire")

        btns = tk.Frame(dlg, bg=FOND)
        btns.pack(fill="x", padx=20, pady=(8, 16))

        def _sauver():
            new = {k: e.get().strip() for k, e in entries.items()}
            try:
                new["smtp_port"] = int(new.get("smtp_port") or 587)
            except ValueError:
                messagebox.showerror("Port invalide",
                                     "Le port SMTP doit être numérique.",
                                     parent=dlg)
                return
            manquant = [k for k in ("smtp_host", "smtp_user", "smtp_password",
                                     "expediteur", "destinataire")
                        if not new.get(k)]
            if manquant:
                messagebox.showwarning(
                    "Champs requis",
                    "Renseignez : " + ", ".join(manquant), parent=dlg)
                return
            self._sauver_config_email(new)
            self.log("Configuration email enregistrée.")
            dlg.destroy()

        tk.Button(btns, text="Annuler", command=dlg.destroy,
                  bg=GRIS, fg="white", relief="flat",
                  font=(POLICE, 11), padx=20, pady=8).pack(side="left")
        tk.Button(btns, text="Enregistrer", command=_sauver,
                  bg=VERT, fg="white", relief="flat",
                  font=(POLICE, 11, "bold"), padx=20, pady=8).pack(
            side="right")

    # ------------------------------------------------------------------ #
    #  SECTION JOURNAL (filtrable + exportable)                           #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  DÉTECTION D'ANOMALIES (autoencoder LSTM)                           #
    # ------------------------------------------------------------------ #
    def _construire_anomalie(self):
        c = self.section(self.content,
                         "Détection d'anomalies (IA) — autoencoder LSTM")

        tk.Label(c,
                 text=("Apprend %d profils de cycle nominaux puis détecte "
                       "les écarts (collision, surcharge, fatigue) en "
                       "temps réel." % ANOMALIE_N_NOMINAL),
                 bg=CARD, fg=GRIS,
                 font=(POLICE, 10), wraplength=620,
                 justify="left").pack(anchor="w", pady=(0, 6))

        self.lbl_anomalie_statut = tk.Label(
            c, text="Statut : —", bg=CARD, fg=TEXTE,
            font=(POLICE, 11, "bold"))
        self.lbl_anomalie_statut.pack(anchor="w")

        self.lbl_anomalie_meta = tk.Label(
            c, text="", bg=CARD, fg=GRIS, font=(POLICE, 9))
        self.lbl_anomalie_meta.pack(anchor="w")

        self.lbl_anomalie_err = tk.Label(
            c, text="Erreur de reconstruction : —",
            bg=CARD, fg=GRIS, font=(POLICE, 10))
        self.lbl_anomalie_err.pack(anchor="w", pady=(4, 4))

        # Ligne de boutons
        ligne_btns = tk.Frame(c, bg=CARD)
        ligne_btns.pack(fill="x", pady=(4, 4))
        self.btn_anomalie_train = self.flat_btn(
            ligne_btns, "🧠 Entraîner le modèle",
            self._lancer_entrainement_anomalie,
            bg=BLEU, fs=11)
        self.btn_anomalie_train.pack(side="left", expand=True,
                                      fill="x", padx=(0, 4))
        self.btn_anomalie_reset = self.flat_btn(
            ligne_btns, "♻ Réinitialiser",
            self._reset_anomalie, bg=GRIS, fs=11)
        self.btn_anomalie_reset.pack(side="left", expand=True,
                                      fill="x", padx=(4, 0))

        # Surveillance + arrêt auto
        ligne_surv = tk.Frame(c, bg=CARD)
        ligne_surv.pack(fill="x", pady=(6, 2))
        self.var_anomalie_surv = tk.BooleanVar(value=False)
        self.chk_anomalie_surv = tk.Checkbutton(
            ligne_surv, text="Surveillance active",
            variable=self.var_anomalie_surv,
            command=self._toggle_surveillance_anomalie,
            bg=CARD, font=(POLICE, 10),
            activebackground=CARD, anchor="w")
        self.chk_anomalie_surv.pack(side="left")

        self.var_anomalie_arret = tk.BooleanVar(value=True)
        self.chk_anomalie_arret = tk.Checkbutton(
            ligne_surv, text="Arrêt auto du mode sur anomalie",
            variable=self.var_anomalie_arret,
            command=self._toggle_arret_auto_anomalie,
            bg=CARD, font=(POLICE, 10),
            activebackground=CARD, anchor="w")
        self.chk_anomalie_arret.pack(side="left", padx=(12, 0))

    def _toggle_surveillance_anomalie(self):
        self.anomaly.surveillance_active = bool(self.var_anomalie_surv.get())
        etat = "activée" if self.anomaly.surveillance_active else "désactivée"
        self.log("Anomalie IA : surveillance %s." % etat)

    def _toggle_arret_auto_anomalie(self):
        self.anomaly.arret_auto = bool(self.var_anomalie_arret.get())

    def _lancer_entrainement_anomalie(self):
        if self._entrainement_en_cours:
            return
        st = self.anomaly.statut()
        if st == "torch_absent":
            messagebox.showerror(
                "PyTorch manquant",
                "PyTorch n'est pas installé.\n\n"
                "Installez-le sur la machine :\n"
                "    pip install torch numpy\n\n"
                "Puis redémarrez l'IHM.")
            return
        if st == "collecte":
            n = self.anomaly.nb_profils()
            messagebox.showwarning(
                "Collecte insuffisante",
                "Il faut %d cycles nominaux pour entraîner.\n"
                "Collectés : %d / %d.\n\n"
                "Lancez un mode pick & place et laissez-le tourner."
                % (ANOMALIE_N_NOMINAL, n, ANOMALIE_N_NOMINAL))
            return
        if not messagebox.askyesno(
                "Entraînement",
                "Lancer l'entraînement de l'autoencoder LSTM ?\n"
                "(Cela peut prendre 30 s à 2 min sur RPi.)"):
            return

        self._entrainement_en_cours = True
        self.btn_anomalie_train.configure(
            state="disabled", text="Entraînement…")
        self.log("Anomalie IA : entraînement démarré.")

        def _progress(epoch, total, loss):
            self.root.after(
                0, self.lbl_anomalie_meta.configure,
                {"text": "Epoch %d/%d — loss %.5f" % (epoch, total, loss)})

        def _travail():
            ok, msg = self.anomaly.entrainer(on_progress=_progress)
            self.root.after(0, self._fin_entrainement_anomalie, ok, msg)

        Thread(target=_travail, daemon=True).start()

    def _fin_entrainement_anomalie(self, ok, msg):
        self._entrainement_en_cours = False
        self.btn_anomalie_train.configure(
            state="normal", text="🧠 Entraîner le modèle")
        if ok:
            self.log("Anomalie IA : %s" % msg)
            messagebox.showinfo("Entraînement terminé", msg)
            # Active automatiquement la surveillance après entraînement
            self.var_anomalie_surv.set(True)
            self.anomaly.surveillance_active = True
        else:
            self.log("Anomalie IA : ÉCHEC — %s" % msg)
            messagebox.showerror("Entraînement échoué", msg)

    def _reset_anomalie(self):
        if not messagebox.askyesno(
                "Réinitialiser ?",
                "Supprimer le modèle ET les profils collectés ?\n"
                "Cette action est irréversible."):
            return
        try:
            self.anomaly.reset(supprimer_profils=True)
            self.var_anomalie_surv.set(False)
            self.lbl_anomalie_err.configure(
                text="Erreur de reconstruction : —", fg=GRIS)
            self.log("Anomalie IA : modèle et profils réinitialisés.")
        except Exception as exc:
            self.log("Anomalie IA : reset KO — %s" % exc)

    def _reagir_anomalie(self, err, threshold):
        """Callback déclenché depuis le détecteur (déjà dans le thread UI
        grâce au root.after positionné lors de l'init)."""
        self.log("⛔ ANOMALIE IA : err=%.5f > seuil=%.5f — arrêt du mode."
                 % (err, threshold))
        try:
            self.arreter_mode()
        except Exception:
            pass
        try:
            messagebox.showwarning(
                "Anomalie détectée",
                "Profil de mouvement atypique détecté par l'autoencoder.\n\n"
                "Erreur de reconstruction : %.5f\n"
                "Seuil : %.5f\n\n"
                "Le mode a été arrêté automatiquement.\n"
                "Vérifiez le robot avant de relancer." % (err, threshold))
        except Exception:
            pass

    def _boucle_anomalie(self):
        try:
            st = self.anomaly.statut()
            if st == "torch_absent":
                self.lbl_anomalie_statut.configure(
                    text="Statut : PyTorch indisponible", fg=ROUGE)
                self.lbl_anomalie_meta.configure(
                    text="pip install torch numpy puis redémarrez l'IHM.")
                self.btn_anomalie_train.configure(state="disabled")
                self.chk_anomalie_surv.configure(state="disabled")
                self.var_anomalie_surv.set(False)
            elif st == "collecte":
                n = self.anomaly.nb_profils()
                self.lbl_anomalie_statut.configure(
                    text="Statut : Collecte %d / %d profils nominaux"
                         % (n, ANOMALIE_N_NOMINAL),
                    fg=ORANGE)
                self.lbl_anomalie_meta.configure(
                    text="Lancez un mode pick & place pour enregistrer "
                         "les cycles nominaux.")
                if not self._entrainement_en_cours:
                    self.btn_anomalie_train.configure(state="disabled")
                self.chk_anomalie_surv.configure(state="disabled")
                self.var_anomalie_surv.set(False)
            elif st == "pret_a_entrainer":
                self.lbl_anomalie_statut.configure(
                    text="Statut : Prêt à entraîner (%d profils)"
                         % self.anomaly.nb_profils(),
                    fg=BLEU)
                if not self._entrainement_en_cours:
                    self.btn_anomalie_train.configure(state="normal")
                self.chk_anomalie_surv.configure(state="disabled")
                self.var_anomalie_surv.set(False)
            elif st == "entraine":
                meta = self.anomaly.meta()
                trained_at = meta.get("trained_at", "—")
                n = meta.get("n_profiles", 0)
                threshold = meta.get("threshold", 0.0)
                self.lbl_anomalie_statut.configure(
                    text="Statut : Modèle entraîné", fg=VERT)
                self.lbl_anomalie_meta.configure(
                    text="Entraîné le %s sur %d profils — seuil=%.5f"
                         % (trained_at, n, threshold))
                if not self._entrainement_en_cours:
                    self.btn_anomalie_train.configure(
                        state="normal", text="🧠 Réentraîner")
                self.chk_anomalie_surv.configure(state="normal")

            # Erreur courante
            err, thresh = self.anomaly.derniere_erreur()
            if err is None:
                self.lbl_anomalie_err.configure(
                    text="Erreur de reconstruction : —", fg=GRIS)
            else:
                if thresh and err > thresh:
                    coul = ROUGE
                    etat = " (ANOMALIE)"
                elif thresh:
                    coul = VERT
                    etat = " (OK)"
                else:
                    coul = GRIS
                    etat = ""
                self.lbl_anomalie_err.configure(
                    text="Erreur de reconstruction : %.5f / seuil %.5f%s"
                         % (err, thresh or 0.0, etat),
                    fg=coul)
        except Exception:
            pass
        self.root.after(1000, self._boucle_anomalie)

    def _construire_journal(self):
        c = self.section(self.content, "Journal")

        # Barre de filtres
        barre = tk.Frame(c, bg=CARD)
        barre.pack(fill="x", pady=(0, 6))

        self.var_filtre = tk.StringVar(value="tout")
        for val, lib in (("tout", "Tout"),
                         ("info", "Infos"),
                         ("erreur", "Erreurs")):
            tk.Radiobutton(
                barre, text=lib, variable=self.var_filtre, value=val,
                bg=CARD, fg=TEXTE, selectcolor=FOND,
                font=(POLICE, 10),
                command=self._on_filtre_journal).pack(side="left", padx=(0, 8))

        self.flat_btn(barre, "Effacer", self._effacer_journal,
                      bg=GRIS, fs=10, pady=4,
                      padx=10).pack(side="right", padx=(4, 0))
        self.flat_btn(barre, "Exporter", self._exporter_journal,
                      bg=BLEU, fs=10, pady=4,
                      padx=10).pack(side="right")

        cadre = tk.Frame(c, bg=CARD)
        cadre.pack(fill="x")

        scroll = ttk.Scrollbar(cadre, orient="vertical")
        self.journal = tk.Text(cadre, height=9, font=(POLICE, 9),
                               bg="#1e1e2e", fg="#d6deeb", relief="flat",
                               bd=0, wrap="word", state="disabled",
                               yscrollcommand=scroll.set)
        scroll.config(command=self.journal.yview)
        scroll.pack(side="right", fill="y")
        self.journal.pack(side="left", fill="both", expand=True)
        self.journal.tag_configure("erreur", foreground="#ff7b7b")

    def log(self, texte):
        """Ajoute une ligne horodatée au journal (thread-safe via after)."""
        ligne = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), texte)
        niveau = "erreur" if RE_ERREUR.search(texte) else "info"
        # Écrit sur disque AVANT de planifier l'affichage Tk : si Tk crashe,
        # la ligne est déjà sauvegardée (line-buffered + flush automatique).
        _crash_log_write(ligne)
        self._journal_queue.append((ligne, niveau))

    def _boucle_journal(self):
        """Vide la file du journal par lots : une seule écriture Tk par
        100 ms quel que soit le débit des sous-processus (l'ancien
        after() par ligne gelait l'UI au lancement des modes)."""
        lot = []
        q = self._journal_queue
        while True:
            try:
                lot.append(q.popleft())
            except IndexError:
                break
        if lot:
            self.journal_lignes.extend(lot)
            # Cap mémoire : 1000 lignes
            if len(self.journal_lignes) > 1000:
                self.journal_lignes = self.journal_lignes[-1000:]
            visibles = [(l, n) for l, n in lot if self._ligne_visible(n)]
            if visibles:
                self._ecrire_lot_dans_journal(visibles)
        self.root.after(100, self._boucle_journal)

    def _ligne_visible(self, niveau):
        f = self.filtre_journal
        if f == "tout":
            return True
        if f == "erreur":
            return niveau == "erreur"
        if f == "info":
            return niveau != "erreur"
        return True

    def _ecrire_lot_dans_journal(self, lignes):
        self.journal.configure(state="normal")
        for ligne, niveau in lignes:
            if niveau == "erreur":
                self.journal.insert(tk.END, ligne + "\n", "erreur")
            else:
                self.journal.insert(tk.END, ligne + "\n")
        # Limite à ~500 lignes affichées
        nb = int(self.journal.index('end-1c').split('.')[0])
        if nb > 500:
            self.journal.delete("1.0", "%d.0" % (nb - 500))
        self.journal.see(tk.END)
        self.journal.configure(state="disabled")

    def _on_filtre_journal(self):
        self.filtre_journal = self.var_filtre.get()
        # Reconstruit le widget en fonction du nouveau filtre
        self.journal.configure(state="normal")
        self.journal.delete("1.0", tk.END)
        for ligne, niveau in self.journal_lignes[-500:]:
            if self._ligne_visible(niveau):
                if niveau == "erreur":
                    self.journal.insert(tk.END, ligne + "\n", "erreur")
                else:
                    self.journal.insert(tk.END, ligne + "\n")
        self.journal.see(tk.END)
        self.journal.configure(state="disabled")

    def _effacer_journal(self):
        if not messagebox.askyesno("Effacer ?",
                                   "Vider le journal de l'écran ?\n"
                                   "(L'historique mémoire est conservé.)"):
            return
        self.journal.configure(state="normal")
        self.journal.delete("1.0", tk.END)
        self.journal.configure(state="disabled")

    def _exporter_journal(self):
        try:
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            nom = "journal_%s.txt" % datetime.now().strftime("%Y%m%d_%H%M%S")
            chemin = EXPORTS_DIR / nom
            with open(chemin, 'w') as f:
                for ligne, _ in self.journal_lignes:
                    f.write(ligne + "\n")
            messagebox.showinfo("Export réussi",
                                "Journal exporté :\n%s" % chemin)
            self.log("Journal exporté → %s" % chemin)
        except Exception as exc:
            self.log("Export journal : erreur — %s" % exc)
            messagebox.showerror("Échec",
                                 "Export impossible : %s" % exc)

    # ------------------------------------------------------------------ #
    #  BOUCLES PÉRIODIQUES                                                #
    # ------------------------------------------------------------------ #
    def _boucle_horloge(self):
        self.lbl_horloge.configure(
            text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._boucle_horloge)

    def _boucle_sysmon(self):
        d = self.sysmon.lire()
        parts = []
        if d["cpu"] is not None:
            parts.append("CPU %3.0f%%" % d["cpu"])
        else:
            parts.append("CPU —")
        if d["ram"] is not None:
            parts.append("RAM %3.0f%%" % d["ram"])
        else:
            parts.append("RAM —")
        if d["temp"] is not None:
            parts.append("Temp %4.1f°C" % d["temp"])
        else:
            parts.append("Temp —")
        self.lbl_sysmon.configure(text="  ".join(parts))
        # Écriture différée des stats (dirty-flag) : au plus une toutes
        # les 2,5 s au lieu d'un json.dump par événement.
        self.stats.flush()
        self.root.after(2500, self._boucle_sysmon)

    def _boucle_position_live(self):
        angles = robot_node.get_positions_deg()
        if angles is None:
            for lbl in self.labels_live:
                lbl.configure(text="—", fg=GRIS)
        else:
            for lbl, a in zip(self.labels_live, angles):
                lbl.configure(text="%+6.1f°" % a, fg=TEXTE)
            # Met à jour min/max pour le diagnostic de fatigue
            self.stats.tracker_joints(angles)
            # Alimente les courbes temps réel ; redessin 1 tick sur 2
            # (2,5 Hz suffisent à l'œil, le tracé complet coûte ~900
            # segments recréés à chaque passage)
            for buf, a in zip(self.courbes_buffers, angles):
                buf.append(a)
            self._courbes_tick = getattr(self, '_courbes_tick', 0) + 1
            if self._courbes_tick % 2 == 0:
                self._redessiner_courbes()
            # Alimente le détecteur d'anomalies (le module gère lui-même
            # l'état mode_actif et l'accumulation cycle par cycle)
            self.anomaly.enregistrer_echantillon(angles)
        self.root.after(200, self._boucle_position_live)

    def _boucle_watchdog(self):
        """Surveille la fraîcheur des messages /joint_states."""
        age = robot_node.get_age_s()
        if age is None:
            # Aucun message reçu : muet en mode déconnecté, alerte sinon
            if self.connecte or self.mode_simulation:
                self.lbl_watchdog.configure(
                    text="⚠ Comms : silence", fg="#ffeb3b")
            else:
                self.lbl_watchdog.configure(
                    text="● Comms : —", fg="#cfe3f7")
        elif age > WATCHDOG_CRIT_S:
            self.lbl_watchdog.configure(
                text="⛔ Comms : %.0fs SILENCE" % age, fg="#ff5252")
            # On loggue une seule fois par décrochage
            if not getattr(self, '_watchdog_alerte', False):
                self._watchdog_alerte = True
                self.log("Watchdog : aucun /joint_states depuis %.1fs" % age)
        elif age > WATCHDOG_WARN_S:
            self.lbl_watchdog.configure(
                text="⚠ Comms : %.1fs" % age, fg="#ffeb3b")
        else:
            self.lbl_watchdog.configure(
                text="● Comms : OK", fg="#a5d6a7")
            self._watchdog_alerte = False
        self.root.after(500, self._boucle_watchdog)

    def _boucle_chrono(self):
        if self.t0_mode is not None:
            dt = datetime.now() - self.t0_mode
            s = int(dt.total_seconds())
            self.lbl_chrono.configure(
                text="⏱ %02d:%02d:%02d" % (s // 3600, (s // 60) % 60, s % 60),
                fg=VERT)
        # Sinon : on laisse la dernière valeur affichée (durée totale)
        self.root.after(500, self._boucle_chrono)

    # ------------------------------------------------------------------ #
    #  GESTION DES SOUS-PROCESSUS ROS2                                    #
    # ------------------------------------------------------------------ #
    def _construire_script(self, commande):
        if ACTIVATE.exists():
            return 'source "%s" && exec %s' % (ACTIVATE, commande)
        return ('source /opt/ros/humble/setup.bash && '
                'source "%s/install/setup.bash" && exec %s'
                % (WORKSPACE, commande))

    def _lancer_processus(self, commande):
        script = self._construire_script(commande)
        try:
            proc = subprocess.Popen(
                ['bash', '-lc', script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setsid)
            return proc
        except Exception as exc:
            self.log("ERREUR lancement : %s" % exc)
            return None

    def _lecteur_stdout(self, proc, prefixe, on_ligne=None):
        try:
            for ligne in iter(proc.stdout.readline, ''):
                ligne = ligne.rstrip()
                if not ligne:
                    continue
                niveau = "erreur" if RE_ERREUR.search(ligne) else "info"
                horod = "[%s]" % datetime.now().strftime("%H:%M:%S")
                texte = "%s   %s %s" % (horod, prefixe, ligne)
                self._journal_queue.append((texte, niveau))
                # Détection de cycle terminé → incrémente le compteur
                if RE_CYCLE.search(ligne):
                    self.root.after(0, self._incrementer_cycle)
                # Détection d'un échec de cycle (debounce 2s pour éviter
                # de compter plusieurs fois la même alerte)
                if RE_ECHEC_CYCLE.search(ligne):
                    t = datetime.now().timestamp()
                    if t - self._derniere_detection_echec > 2.0:
                        self._derniere_detection_echec = t
                        self.stats.cycle_echec()
                # Détection d'un home recovery
                if RE_HOME_RECOVERY.search(ligne):
                    t = datetime.now().timestamp()
                    if t - self._derniere_detection_recovery > 5.0:
                        self._derniere_detection_recovery = t
                        self.stats.home_recovery()
                # Erreur → trace dans l'historique
                if niveau == "erreur":
                    self.stats.ajouter_erreur(
                        ligne, self.mode_courant or prefixe)
                if on_ligne is not None:
                    on_ligne(ligne)
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _incrementer_cycle(self):
        self.compteur_cycles += 1
        self._cycles_session += 1
        self.stats.cycle_ok()
        self.lbl_cycles.configure(
            text="Cycles pick & place : %d" % self.compteur_cycles)
        # Détection d'anomalies : analyse le profil du cycle qui vient
        # de se terminer. En mode collecte, le profil est sauvegardé ; en
        # mode surveillance, l'erreur de reconstruction est calculée et,
        # si elle dépasse le seuil, le callback _reagir_anomalie déclenche
        # l'arrêt automatique du mode.
        try:
            res = self.anomaly.cycle_termine()
        except Exception as exc:
            self.log("Anomalie IA : erreur évaluation — %s" % exc)
            res = None
        if res is not None:
            kind, val = res
            if kind == "collect":
                if int(val) % 10 == 0 or int(val) >= ANOMALIE_N_NOMINAL:
                    self.log("Anomalie IA : collecte %d/%d cycles nominaux."
                             % (int(val), ANOMALIE_N_NOMINAL))
            elif kind == "ok":
                pass   # silencieux : c'est le cas normal
            elif kind == "anomalie":
                # Le callback est déjà déclenché par le détecteur si
                # arret_auto est activé ; on log de toute façon.
                self.log("Anomalie IA : cycle ATYPIQUE (err=%.5f)." % val)
            elif kind == "trop_court":
                pass

    def _arreter_processus(self, proc, nom):
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        self.log("Arrêt de %s en cours..." % nom)
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=8)
            self.log("%s arrêté." % nom)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=4)
            self.log("%s arrêté (SIGTERM)." % nom)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=3)
            self.log("%s tué (SIGKILL)." % nom)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  ROBOTSERVER (auto-lancé en arrière-plan)                           #
    # ------------------------------------------------------------------ #
    def _lancer_robotserver(self):
        if self.robotserver_proc is not None and self.robotserver_proc.poll() is None:
            return
        # Une session IHM précédente a pu laisser un RobotServer orphelin.
        # S'il en reste plusieurs, chacun répond à /start_robot et lance son
        # propre demo.launch.py → plusieurs RViz/MoveIt en parallèle. On les
        # élimine avant d'en démarrer un seul. Le motif "mon_controleur.*
        # robotserver" ne matche pas l'IHM elle-même (ihm/ihm_robot.py).
        subprocess.run(
            ["pkill", "-9", "-f", "mon_controleur.*robotserver"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.log("Démarrage de RobotServer (services /start_robot · /stop_robot)...")
        self.robotserver_proc = self._lancer_processus(
            "ros2 run mon_controleur robotserver")
        if self.robotserver_proc is None:
            self.log("ERREUR : RobotServer n'a pas démarré — "
                     "la connexion ne pourra pas aboutir.")
            return
        Thread(target=self._lecteur_stdout,
               args=(self.robotserver_proc, "[RobotServer]"),
               daemon=True).start()

    # ------------------------------------------------------------------ #
    #  CONNEXION ROBOT                                                    #
    # ------------------------------------------------------------------ #
    def connecter(self):
        if self.connecte:
            return
        self.log("Connexion au robot (service /start_robot)...")
        self._set_voyant(ORANGE)
        self.lbl_statut.configure(text="Connexion en cours...")
        self._maj_etat_boutons()

        def _travail():
            client = robot_node.start_client
            if not client.wait_for_service(timeout_sec=3.0):
                self.root.after(0, self._connexion_echouee,
                                "service /start_robot indisponible "
                                "(le RobotServer tourne-t-il ?)")
                return
            future = client.call_async(Trigger.Request())
            future.add_done_callback(
                lambda fut: self.root.after(0, self._reponse_start, fut))

        Thread(target=_travail, daemon=True).start()

    def _reponse_start(self, future):
        try:
            res = future.result()
        except Exception as exc:
            self._connexion_echouee("erreur RPC : %s" % exc)
            return
        if res is None:
            self._connexion_echouee("pas de réponse du serveur")
            return
        if not res.success:
            self._connexion_echouee(res.message or "service a échoué")
            return
        # /start_robot a juste lancé demo.launch.py (Popen). Le bring-up CRI
        # met 10-20 s à publier /joint_states (handshake TCP + spawn des
        # controllers). On ne marque "Connecté" qu'après réception effective
        # d'un message frais, sinon le bouton Mode devient cliquable trop tôt
        # et le mode timeoute à 5 s sur /joint_states.
        self.log("Robot lancé : %s" % (res.message or ""))
        self.lbl_statut.configure(text="Initialisation ROS2…")
        self._connexion_attente_debut = _time.monotonic()
        self._attendre_robot_pret()

    def _attendre_robot_pret(self):
        """Polling 200 ms : attend un /joint_states frais avant 'Connecté'."""
        # Si l'utilisateur a annulé entretemps (déconnecté), on stoppe.
        if self.lbl_statut.cget("text") not in ("Initialisation ROS2…",
                                                "Connexion en cours..."):
            return
        now = _time.monotonic()
        with robot_node._lock:
            last_wall = robot_node.last_wall
            positions = robot_node.last_positions
        frais = (last_wall > 0.0) and (now - last_wall < 1.5)
        reference = positions is not None and any(abs(p) > 1e-3 for p in positions)
        if frais and reference:
            self.connecte = True
            self._set_voyant(VERT)
            self.lbl_statut.configure(text="Connecté")
            self.log("Robot prêt — /joint_states reçu.")
            self._maj_etat_boutons()
            return
        if now - self._connexion_attente_debut > 30.0:
            if frais and not reference:
                motif = ("/joint_states reçu mais à 0,0,0,0,0,0 — "
                         "robot non référencé. Référencez via le pendant CRI.")
            else:
                motif = ("aucun /joint_states reçu après 30 s — "
                         "le driver CRI n'a pas démarré (câble / IP / alim ?).")
            self._connexion_echouee(motif)
            return
        self.root.after(200, self._attendre_robot_pret)

    def _connexion_echouee(self, motif):
        self.connecte = False
        self._set_voyant(ROUGE)
        self.lbl_statut.configure(text="Échec de connexion")
        self.log("Connexion : échec — %s" % motif)
        self._maj_etat_boutons()

    def deconnecter(self):
        self.log("Déconnexion demandée...")
        self._set_voyant(ORANGE)
        self.lbl_statut.configure(text="Déconnexion en cours...")
        # Mémorise immédiatement les derniers angles connus pour la
        # prochaine reconnexion (avant que /joint_states ne se taise)
        try:
            robot_node._sauver_derniers_angles()
        except Exception:
            pass

        def _travail():
            self._arreter_processus(self.mode_proc, "mode")
            self.mode_proc = None
            self.root.after(0, self._on_mode_fini, False)
            client = robot_node.stop_client
            if not client.wait_for_service(timeout_sec=3.0):
                self.root.after(0, self._fin_deconnexion,
                                "service /stop_robot indisponible")
                return
            future = client.call_async(Trigger.Request())
            future.add_done_callback(
                lambda fut: self.root.after(0, self._reponse_stop, fut))

        Thread(target=_travail, daemon=True).start()

    def _reponse_stop(self, future):
        try:
            res = future.result()
            msg = res.message if res is not None else "pas de réponse"
            self.log("Déconnexion : %s" % msg)
        except Exception as exc:
            self.log("Déconnexion : erreur — %s" % exc)
        self._fin_deconnexion(None)

    def _fin_deconnexion(self, motif):
        if motif:
            self.log("Déconnexion : %s" % motif)
        self.connecte = False
        self._set_voyant(GRIS)
        self.lbl_statut.configure(text="Déconnecté")
        self._maj_etat_boutons()

    # ------------------------------------------------------------------ #
    #  GESTION DES MODES                                                  #
    # ------------------------------------------------------------------ #
    def lancer_mode(self):
        if not self.mode_simulation and not self.connecte:
            messagebox.showwarning("Connexion requise",
                                   "Connectez d'abord le robot (MoveIt).")
            return
        if self.mode_proc is not None and self.mode_proc.poll() is None:
            return

        libelle = self.var_mode.get()
        table = MODES_SIMU if self.mode_simulation else MODES_PHYSIQUE
        commande = table[libelle]
        if libelle == MODE_DANCE:
            commande += " dance:=%s" % self.var_choreo.get()

        suffixe = " [SIMU]" if self.mode_simulation else ""
        self.log("Lancement du mode : %s%s" % (libelle, suffixe))
        self.mode_proc = self._lancer_processus(commande)
        if self.mode_proc is None:
            self.lbl_mode_statut.configure(text="Échec du lancement du mode.")
            return

        self.mode_courant = libelle
        self.compteur_cycles = 0
        self._cycles_session = 0
        self.lbl_cycles.configure(text="Cycles pick & place : 0")
        self.t0_mode = datetime.now()
        self.stats.mode_start(libelle)
        self.anomaly.mode_demarre()
        self.lbl_mode.configure(text="Mode : %s" % libelle)
        self.lbl_mode_statut.configure(text="Mode en cours d'exécution...")

        Thread(target=self._lecteur_stdout,
               args=(self.mode_proc, "[Mode]"), daemon=True).start()
        Thread(target=self._surveiller_mode, daemon=True).start()
        self._maj_etat_boutons()

    def _surveiller_mode(self):
        proc = self.mode_proc
        proc.wait()
        if self.mode_proc is proc:
            self.mode_proc = None
            self.root.after(0, self._on_mode_fini, True)

    def _on_mode_fini(self, termine_seul):
        # Fige le chrono : on calcule la durée finale puis on libère t0_mode
        duree_session = 0
        if self.t0_mode is not None:
            dt = datetime.now() - self.t0_mode
            duree_session = int(dt.total_seconds())
            self.lbl_chrono.configure(
                text="⏱ %02d:%02d:%02d (terminé)"
                     % (duree_session // 3600,
                        (duree_session // 60) % 60, duree_session % 60),
                fg=GRIS)
            self.t0_mode = None

        # Persiste les stats du mode (durée + cycles de la session)
        if self.mode_courant:
            self.stats.mode_end(self.mode_courant,
                                duree_session, self._cycles_session)
        self.anomaly.mode_arrete()

        self.mode_courant = None
        self.lbl_mode.configure(text="Mode : —")
        if termine_seul:
            self.lbl_mode_statut.configure(text="Mode terminé.")
            self.log("Mode terminé. Cycles : %d" % self.compteur_cycles)
        else:
            self.lbl_mode_statut.configure(text="Aucun mode en cours.")
        self._maj_etat_boutons()

    def arreter_mode(self):
        if self.mode_proc is None or self.mode_proc.poll() is not None:
            return
        self.log("ARRÊT du mode demandé.")
        proc = self.mode_proc
        self.mode_proc = None

        def _travail():
            self._arreter_processus(proc, "mode")
            self.root.after(0, self._on_mode_fini, False)

        Thread(target=_travail, daemon=True).start()
        self._maj_etat_boutons()

    # ------------------------------------------------------------------ #
    #  CONTRÔLE DE LA PINCE                                               #
    # ------------------------------------------------------------------ #
    def ouvrir_pince(self):
        self._commande_pince(False, "ouverture")

    def fermer_pince(self):
        self._commande_pince(True, "fermeture")

    def _commande_pince(self, fermer, label):
        def _travail():
            client = robot_node.gripper_client
            if not client.wait_for_service(timeout_sec=2.0):
                self.root.after(0, self.lbl_pince.configure,
                                {"text": "État pince : service indisponible",
                                 "fg": ROUGE})
                self.log("Pince : service /gripper/command indisponible.")
                return
            req = SetBool.Request()
            req.data = fermer
            future = client.call_async(req)
            future.add_done_callback(
                lambda fut: self.root.after(0, self._reponse_pince, fut, label))
            self.log("Pince : %s demandée." % label)

        Thread(target=_travail, daemon=True).start()

    def _reponse_pince(self, future, label):
        try:
            res = future.result()
            if res is not None and res.success:
                etat = "fermée" if label == "fermeture" else "ouverte"
                self.lbl_pince.configure(text="État pince : %s" % etat, fg=VERT)
                self.stats.pince_action(label == "fermeture")
                self.log("Pince : %s confirmée." % label)
            else:
                msg = res.message if res is not None else "pas de réponse"
                self.lbl_pince.configure(
                    text="État pince : échec (%s)" % msg, fg=ROUGE)
                self.log("Pince : échec de %s — %s" % (label, msg))
        except Exception as exc:
            self.lbl_pince.configure(text="État pince : erreur", fg=ROUGE)
            self.log("Pince : erreur — %s" % exc)

    # ------------------------------------------------------------------ #
    #  MISE À JOUR DE L'ÉTAT DES BOUTONS                                  #
    # ------------------------------------------------------------------ #
    def _maj_etat_boutons(self):
        mode_actif = (self.mode_proc is not None
                      and self.mode_proc.poll() is None)
        simu = self.mode_simulation

        self.btn_connecter.configure(
            state="disabled" if (simu or self.connecte) else "normal")
        self.btn_deconnecter.configure(
            state="normal" if (not simu and self.connecte) else "disabled")

        pret = (simu or self.connecte) and not mode_actif
        self.btn_lancer.configure(state="normal" if pret else "disabled")
        self.btn_arreter.configure(
            state="normal" if mode_actif else "disabled")
        self.combo_mode.configure(
            state="disabled" if mode_actif else "readonly")
        self.combo_choreo.configure(
            state="disabled" if mode_actif else "readonly")

        toggle_bloque = mode_actif or self.connecte
        self.btn_mode_simu.configure(
            state="disabled" if toggle_bloque else "normal")

    # ------------------------------------------------------------------ #
    #  BASCULE PHYSIQUE / SIMULATION                                      #
    # ------------------------------------------------------------------ #
    def _toggle_simulation(self):
        if self.mode_proc is not None and self.mode_proc.poll() is None:
            messagebox.showwarning(
                "Mode en cours",
                "Arrêtez le mode en cours avant de changer de cible.")
            return
        if self.connecte:
            messagebox.showwarning(
                "Robot connecté",
                "Déconnectez le robot réel avant de passer en simulation.")
            return

        self.mode_simulation = not self.mode_simulation
        if self.mode_simulation:
            self.btn_mode_simu.configure(text="SIMU", bg=ORANGE,
                                         activebackground=ORANGE)
            self.lbl_statut.configure(text="Simulation (Gazebo)")
            self._set_voyant(ORANGE)
            self.lbl_connexion_info.configure(
                text="Mode simulation : connexion non requise "
                     "(les launch sim_* démarrent Gazebo + MoveIt).")
            self.log("Bascule en mode SIMULATION (Gazebo Ignition).")
        else:
            self.btn_mode_simu.configure(text="PHYSIQUE", bg=VERT,
                                         activebackground=VERT)
            self.lbl_statut.configure(text="Déconnecté")
            self._set_voyant(GRIS)
            self.lbl_connexion_info.configure(
                text="Via RobotServer (services /start_robot · /stop_robot)")
            self.log("Bascule en mode PHYSIQUE (robot réel).")
        self._maj_etat_boutons()

    # ------------------------------------------------------------------ #
    #  PLEIN ÉCRAN / FERMETURE                                            #
    # ------------------------------------------------------------------ #
    def _toggle_fullscreen(self, _event=None):
        actuel = bool(self.root.attributes('-fullscreen'))
        self.root.attributes('-fullscreen', not actuel)
        # ⛶ = passer en plein écran ; ❐ = revenir en fenêtré
        self.btn_fenetre.configure(text="⛶" if actuel else "❐")

    def _reduire(self):
        """Réduit la fenêtre dans la barre des tâches."""
        # Le plein écran bloque l'iconification sur certains gestionnaires
        # de fenêtres : on le désactive d'abord.
        if bool(self.root.attributes('-fullscreen')):
            self.root.attributes('-fullscreen', False)
            self.btn_fenetre.configure(text="⛶")
        self.root.iconify()

    def quitter(self):
        if not messagebox.askyesno("Quitter",
                                   "Quitter l'IHM et arrêter le robot ?"):
            return
        self.log("Fermeture de l'IHM...")
        # Mémorise les derniers angles avant de couper le robot
        try:
            robot_node._sauver_derniers_angles()
        except Exception:
            pass

        def _travail():
            self._arreter_processus(self.mode_proc, "mode")
            self.mode_proc = None
            self._arreter_processus(self.home_proc, "HOME")
            self.home_proc = None
            if self.connecte:
                try:
                    client = robot_node.stop_client
                    if client.wait_for_service(timeout_sec=1.0):
                        client.call_async(Trigger.Request())
                        import time as _t
                        _t.sleep(0.5)
                except Exception:
                    pass
            self._arreter_processus(self.robotserver_proc, "RobotServer")
            self.robotserver_proc = None
            self.root.after(0, self._fermeture_finale)

        Thread(target=_travail, daemon=True).start()

    def _fermeture_finale(self):
        try:
            self.vocal.arreter()
        except Exception:
            pass
        try:
            self.stats.session_stop()
        except Exception:
            pass
        try:
            robot_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------------ #
    #  BOUCLE PRINCIPALE                                                  #
    # ------------------------------------------------------------------ #
    def run(self):
        self.root.mainloop()


# ============================================================================
#  6. POINT D'ENTRÉE
# ============================================================================
if __name__ == '__main__':
    app = IHMRobot()
    app.run()
