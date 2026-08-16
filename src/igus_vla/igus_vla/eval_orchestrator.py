#!/usr/bin/env python3
"""
eval_orchestrator.py — Harnais d'ÉVALUATION chiffrée d'une politique VLA.

POURQUOI CE NŒUD
================
Le projet n'avait aucun protocole d'évaluation : chaque déploiement était jugé
« à l'œil » sur un ou deux essais, avec une position d'objet différente à chaque
fois. Résultat : impossible d'attribuer un progrès (ou une régression) à une
modification — c'est exactement ce qui a coûté la régression v2 non détectée
(cf. archives/diagnostics/DIAG_REGRESSION_V2.md §4.5 : « N = 20 essais, mêmes 20
positions pour tous les modèles, seedées ; sans ça, impossible d'attribuer les
progrès »).

Ce nœud matérialise ce protocole : il déroule N épisodes dans une simulation
DÉJÀ LANCÉE (jamais redémarrée entre les épisodes — le seul reset est celui de
la machine à états de la politique, et la téléportation de l'objet), sur des
positions de pick tirées de façon REPRODUCTIBLE à partir d'un seed maître.

Boucle par épisode :
    1. position de pick déterministe (seed maître → seed d'épisode) ;
    2. pince ouverte, puis téléportation de l'objet (`/object_position_in_world`)
       et ATTENTE DE CONFIRMATION sur `/gripper/object_pose` ;
    3. appel de `/policy/reset` (ré-arme la politique : retour HOME puis exécution) ;
    4. attente de `/policy/episode_result` (garde-temps orchestrateur plus généreux
       que le `episode_timeout_s` du nœud de politique) ;
    5. consignation de la ligne de résultat (sonde CSV + rapport réécrit), épisode
       suivant.

CE QUI EST MESURÉ (et pourquoi)
==============================
* **Première fermeture de pince** (`dx, dy, dz, dxy, dist, attached`, lus sur
  `/gripper/grasp_result`) : c'est LA mesure d'erreur qui pilote le diagnostic.
  Séparer `dxy` (erreur latérale) de `dz` (erreur verticale) tranche entre deux
  causes très différentes : un `dz` systématiquement négatif = la pince ferme trop
  haut ⇒ problème de SUIVI de trajectoire (cf. le bug `action_duration` 0,8 s qui
  fermait 11 cm trop haut) ; un `dxy` important et corrélé à la position de l'objet
  = déficit d'ANCRAGE VISUEL (la politique ne voit pas où est l'objet).
* **Nombre total de fermetures** : une politique qui ferme, rouvre et referme est
  un mode d'échec distinct d'une politique qui ferme une fois au mauvais endroit.
* **% du temps avec joint5 < 0** : signature du « poignet retourné » identifiée en
  v2 (62 % du temps sur le run du 2026-07-01), invisible dans un verdict binaire.
* **Pose finale de l'objet** (vérité-terrain publiée par le shim) + distance au bac :
  verdict de dépose INDÉPENDANT de l'auto-évaluation de la politique.

SORTIES
=======
* sonde télémétrie `episode.csv` dans le dossier du run (`IGUS_RUN_ID` /
  `IGUS_TELEMETRY_DIR`, partagé avec les sondes des autres nœuds) ;
* `positions.json` (dossier du run) : les positions jouées, rejouables telles
  quelles par une campagne ultérieure via le paramètre `positions_file` ;
* `rapport_eval_courant.md` / `.csv` sous `report_root`, RÉÉCRITS après CHAQUE
  épisode (une campagne dure 20-30 min : une interruption ne doit rien coûter) ;
* `historique_evals.md` / `.csv` sous `report_root`, en APPEND en fin de campagne
  (tout l'historique des modèles évalués dans un seul fichier comparable).

CONTRAT AVEC LES AUTRES NŒUDS
=============================
consommé  : `/policy/episode_result` (std_msgs/String JSON : verdict, reason,
            duration_s), `/gripper/grasp_result` (std_msgs/String JSON : dx, dy,
            dz, dist, dxy, attached, radius), `/gripper/object_pose` (PoseStamped),
            `/joint_states` (sensor_msgs/JointState)
produit   : `/object_position_in_world` (PointStamped) — le shim téléporte l'objet
services  : `/policy/reset` (Trigger), `/gripper/command` (SetBool, data=False=ouvrir)

RÈGLE : ÉCHOUER FRANCHEMENT. Un service absent, un objet qui ne se téléporte pas,
un résultat qui n'arrive jamais → message explicite et arrêt. Jamais d'attente
silencieuse : une campagne muette pendant deux heures est le pire des résultats.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from datetime import datetime
from typing import Any, Optional, Sequence

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from igus_vla.telemetry import Probe, run_dir, run_id, write_meta

# ── Garde-fous d'abandon ──────────────────────────────────────────────────────
# Au-delà de ce nb d'épisodes consécutifs SANS aucun résultat de politique, on
# abandonne : la pile est cassée (politique morte, contrôleurs sourds), inutile de
# jouer 20 épisodes vides pendant une heure. Même esprit que ABORT_TIMEOUTS de
# record_orchestrator.py.
ABORT_TIMEOUTS = 3
# Idem pour les téléportations : si l'objet ne bouge plus SIM VIVANTE, quelque
# chose de systématique est cassé (shim mort, service set_pose HS). Une sim
# FIGÉE, elle, est détectée à part (horloge /clock immobile) et fait abandonner
# immédiatement avec la vraie cause — le 2026-08-14 au matin, les « 3 téléports
# refusés » étaient en réalité un gz server gelé, et le message accusait la
# pince. Un échec isolé de téléportation marque l'épisode `invalide_teleport`
# et CONTINUE : plus jamais d'effondrement en cascade sur un incident ponctuel.
ABORT_TELEPORTS = 6

# ── Zone accessible de tirage (géométrie v3 validée) ──────────────────────────
# Constantes IDENTIQUES à record_orchestrator._sample_object_xy (lignes 198-241) :
# anneau 0,15–0,54 m, exclusion du bac, de l'ombre de colonne et du bord avant-droit
# (~98-100 % d'IK, mesuré par visibility_sweep le 2026-06-27, cf. ZONE_ACCESSIBLE_V3).
# Voir `sample_pick_xy` pour la stratégie « canonique d'abord, copie en secours ».
ZONE_VERSION = "anneau_v3_2026-06-27"
ZONE_R_MIN = 0.15
ZONE_R_MAX = 0.54
ZONE_Y_MIN = -0.52
ZONE_BIN_HALF = 0.092        # demi-côté du bac (0.184 / 2)
ZONE_BIN_MARGIN = 0.13       # marge pince/approche autour des parois
ZONE_MAX_TRIES = 30
ZONE_FALLBACK = (0.35, -0.20)

# Version du format de positions.json — permet de refuser proprement un fichier
# écrit par une version future du harnais plutôt que de mal l'interpréter.
POSITIONS_FORMAT = 1


# ══════════════════════════════════════════════════════════════════════════════
# Tirage des positions de pick
# ══════════════════════════════════════════════════════════════════════════════

def _sample_pick_xy_local(seed: int, place_x: float, place_y: float) -> tuple[float, float]:
    """Copie FIDÈLE de `record_orchestrator._sample_object_xy` (branche randomisée).

    Sert de secours si l'import du module canonique échoue (voir `sample_pick_xy`).
    Le générateur est un `random.Random(seed)` : même graine ⇒ mêmes tirages que la
    version canonique, qui seede le Mersenne Twister global avec la même valeur.
    """
    rng = random.Random(seed)
    keepout = ZONE_BIN_HALF + ZONE_BIN_MARGIN     # 0.222 m de part et d'autre du centre
    for _ in range(ZONE_MAX_TRIES):
        # Tirage polaire dans l'anneau : r ~ √U[r_min², r_max²] (uniformité en surface)
        r = math.sqrt(rng.uniform(ZONE_R_MIN ** 2, ZONE_R_MAX ** 2))
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        if y < ZONE_Y_MIN:                        # pas derrière la caméra front
            continue
        if abs(x - place_x) < keepout and abs(y - place_y) < keepout:
            continue                              # dans/contre le bac ⇒ saisie impossible
        angle = math.degrees(math.atan2(y, x))
        if angle <= -140.0 or angle >= 162.0:     # ombre de la colonne du bras
            continue
        if 12.0 <= angle <= 48.0 and r > 0.47:    # bord de cadre avant-droit
            continue
        return (x, y)
    return ZONE_FALLBACK


def sample_pick_xy(seed: int, place_x: float, place_y: float) -> tuple[float, float]:
    """Tire une position de pick dans la zone accessible, de façon déterministe.

    CHOIX ASSUMÉ (consigne « importe la fonction ou reproduis-la à l'identique ») :
    on appelle **en priorité la fonction canonique** de `record_orchestrator`, celle
    qui a produit les datasets — c'est la seule façon d'être certain qu'évaluation et
    collecte échantillonnent la MÊME zone. Comme il s'agit d'une méthode d'instance
    (elle lit `self.randomize`, `self.place_x`, `self.place_y`) et qu'on ne peut pas
    instancier ce nœud ici (il ouvre des services et démarre sa propre boucle), on
    l'appelle en lui passant un objet-doublure qui porte ces trois attributs.
    Le déterminisme vient de `random.seed()` : la fonction canonique tire sur le
    Mersenne Twister global.

    Si l'import échoue (module renommé, méthode refactorée), on retombe sur
    `_sample_pick_xy_local`, copie fidèle des mêmes constantes. Le nom de la voie
    réellement empruntée est consigné dans `positions.json` et dans le meta du run :
    une divergence de géométrie doit rester visible, jamais silencieuse.

    NOTE POUR PLUS TARD : la vraie correction est d'extraire cette géométrie dans un
    module partagé (`igus_vla/zone.py`) dont les deux orchestrateurs dépendraient.
    Elle n'a pas été faite ici parce que `record_orchestrator.py` est hors du
    périmètre de cette tâche (fichier partagé avec d'autres chantiers en cours).
    """
    try:
        from igus_vla.record_orchestrator import RecordOrchestrator

        class _ZoneStub:
            """Doublure portant les seuls attributs lus par `_sample_object_xy`."""
            randomize = True
            pick_x = 0.0
            pick_y = 0.0

        stub = _ZoneStub()
        stub.place_x = float(place_x)   # type: ignore[attr-defined]
        stub.place_y = float(place_y)   # type: ignore[attr-defined]
        random.seed(seed)
        x, y = RecordOrchestrator._sample_object_xy(stub)  # type: ignore[arg-type]
        return (float(x), float(y))
    except Exception:  # noqa: BLE001 — toute rupture d'API bascule sur la copie
        return _sample_pick_xy_local(seed, place_x, place_y)


def sampler_source() -> str:
    """Nom de la voie de tirage effectivement disponible (traçabilité du run)."""
    try:
        from igus_vla.record_orchestrator import RecordOrchestrator  # noqa: F401
        return "record_orchestrator._sample_object_xy"
    except Exception:  # noqa: BLE001
        return "copie_locale (_sample_pick_xy_local)"


def episode_seed(master_seed: int, idx: int) -> int:
    """Seed de l'épisode `idx` (1-based) dérivé du seed maître.

    Dérivation INDÉPENDANTE de l'historique : la position de l'épisode 7 ne dépend
    pas du nombre de rejets du tirage de l'épisode 6. On peut donc rejouer un épisode
    isolément, ou allonger une campagne de 20 à 30 épisodes sans changer les 20
    premières positions — condition pour que deux campagnes restent comparables.
    """
    return (int(master_seed) * 100_003 + int(idx) * 7_919) % (2 ** 31 - 1)


# ══════════════════════════════════════════════════════════════════════════════
# Petites statistiques (stdlib seule : le nœud tourne avec le python système)
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Percentile par interpolation linéaire (méthode « linear », comme numpy)."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * float(q)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[int(pos)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def _median(values: Sequence[float]) -> Optional[float]:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return float(xs[mid]) if n % 2 else float((xs[mid - 1] + xs[mid]) / 2.0)


def _fmt(v: Optional[float], prec: int = 3, unit: str = "") -> str:
    """Formatage « — » si absent : une case vide dit « pas mesuré », pas « zéro »."""
    return "—" if v is None else f"{v:.{prec}f}{unit}"


# Colonnes de la sonde « episode » ET du CSV de rapport, dans cet ordre.
# Les 19 premières sont le contrat demandé ; les suivantes sont des compléments
# de diagnostic (verdict indépendant de dépose, durée mesurée côté orchestrateur…).
EPISODE_FIELDS = (
    "idx", "seed", "pick_x", "pick_y", "verdict", "reason", "attached",
    "dx", "dy", "dz", "dxy", "dist", "n_fermetures",
    "obj_final_x", "obj_final_y", "obj_final_z", "dist_bac", "pct_j5_neg", "duree_s",
    # ── compléments ──
    "attached_any", "place_ok", "duree_mesuree_s", "n_ech_j5",
)


class EvalOrchestrator(Node):
    """Déroule une campagne de N épisodes d'évaluation sur une sim déjà lancée."""

    def __init__(self) -> None:
        super().__init__("eval_orchestrator")

        # ── Campagne ──────────────────────────────────────────────────────────
        self.num_episodes = int(self.declare_parameter("num_episodes", 20).value)
        self.seed = int(self.declare_parameter("seed", 42).value)
        # Fichier de positions : rejoué s'il existe, ÉCRIT s'il n'existe pas. C'est
        # le filet de sécurité du protocole : même si la géométrie de la zone change
        # un jour, une campagne ancienne reste rejouable à l'identique.
        self.positions_file = str(self.declare_parameter("positions_file", "").value)
        self.report_root = str(self.declare_parameter("report_root", "outputs/eval").value)
        # Étiquette du modèle évalué (chemin de checkpoint) : sans elle, l'historique
        # cumulatif ne permet pas de dire QUEL modèle a produit quelle ligne.
        self.model_label = str(self.declare_parameter("model_label", "").value)

        # ── Géométrie de la scène ─────────────────────────────────────────────
        self.object_z = float(self.declare_parameter("object_z", 0.018).value)
        self.object_topic = str(self.declare_parameter(
            "object_topic", "/object_position_in_world").value)
        self.place_x = float(self.declare_parameter("place_x", 0.0).value)
        self.place_y = float(self.declare_parameter("place_y", 0.25).value)
        self.success_radius = float(self.declare_parameter("success_radius", 0.12).value)

        # ── Temporisations ────────────────────────────────────────────────────
        # Doit refléter le `episode_timeout_s` du nœud de politique : l'orchestrateur
        # attend TOUJOURS plus longtemps que lui (marge), pour que le verdict vienne
        # de la politique — un timeout orchestrateur signale une politique muette,
        # ce qui est un diagnostic différent d'un épisode simplement trop long.
        self.episode_timeout_s = float(self.declare_parameter("episode_timeout_s", 90.0).value)
        self.result_margin_s = float(self.declare_parameter("result_margin_s", 45.0).value)
        self.settle_time = float(self.declare_parameter("settle_time", 15.0).value)
        self.ready_timeout_s = float(self.declare_parameter("ready_timeout_s", 300.0).value)
        self.object_pose_timeout_s = float(
            self.declare_parameter("object_pose_timeout_s", 90.0).value)
        self.teleport_tol = float(self.declare_parameter("teleport_tol", 0.02).value)
        self.teleport_timeout_s = float(self.declare_parameter("teleport_timeout_s", 10.0).value)
        self.teleport_retries = int(self.declare_parameter("teleport_retries", 3).value)
        self.open_wait_s = float(self.declare_parameter("open_wait_s", 1.5).value)
        # Au-delà de ce délai MUR sans avancée de /clock, la sim est déclarée
        # figée : gz server gelé → shim muet, set_pose HS, timers sim morts.
        # Continuer une campagne là-dessus ne mesure rien ; on abandonne avec la
        # cause EXACTE (le mode de panne du 2026-08-14 au matin).
        self.sim_freeze_abort_s = float(
            self.declare_parameter("sim_freeze_abort_s", 15.0).value)
        # Fenêtre pendant laquelle un `episode_result` est considéré comme un RESTE de
        # l'épisode précédent (la politique repasse par HOME avant d'agir : elle ne peut
        # pas conclure en 2 s). À garder strictement inférieur à la durée du retour HOME.
        self.post_reset_grace_s = float(self.declare_parameter("post_reset_grace_s", 2.0).value)
        self.final_pose_wait_s = float(self.declare_parameter("final_pose_wait_s", 1.5).value)
        self.inter_episode_delay = float(self.declare_parameter("inter_episode_delay", 3.0).value)
        self.shutdown_when_done = bool(self.declare_parameter("shutdown_when_done", True).value)

        # ── Poignet retourné ──────────────────────────────────────────────────
        self.wrist_joint = str(self.declare_parameter("wrist_joint", "joint5").value)
        self.wrist_neg_threshold = float(
            self.declare_parameter("wrist_neg_threshold", 0.0).value)

        # ── Noms de services / topics (aucun chemin ni nom en dur) ────────────
        self.gripper_service = str(self.declare_parameter(
            "gripper_service", "/gripper/command").value)
        self.reset_service = str(self.declare_parameter(
            "reset_service", "/policy/reset").value)
        self.result_topic = str(self.declare_parameter(
            "result_topic", "/policy/episode_result").value)
        self.grasp_topic = str(self.declare_parameter(
            "grasp_topic", "/gripper/grasp_result").value)
        self.object_pose_topic = str(self.declare_parameter(
            "object_pose_topic", "/gripper/object_pose").value)
        self.joint_states_topic = str(self.declare_parameter(
            "joint_states_topic", "/joint_states").value)

        try:
            self._use_sim_time = bool(self.get_parameter("use_sim_time").value)
        except Exception:  # noqa: BLE001
            self._use_sim_time = False

        # ── E/S ROS ───────────────────────────────────────────────────────────
        self._reset_cli = self.create_client(Trigger, self.reset_service)
        self._gripper_cli = self.create_client(SetBool, self.gripper_service)
        self._object_pub = self.create_publisher(PointStamped, self.object_topic, 10)
        self.create_subscription(String, self.result_topic, self._on_episode_result, 10)
        self.create_subscription(String, self.grasp_topic, self._on_grasp_result, 10)
        self.create_subscription(PoseStamped, self.object_pose_topic,
                                 self._on_object_pose, 10)
        # QoS capteur : /joint_states vient de gz_ros2_control ; un abonné « best
        # effort » accepte aussi bien un éditeur fiable et ne bloque jamais la boucle.
        self.create_subscription(JointState, self.joint_states_topic,
                                 self._on_joint_states, qos_profile_sensor_data)
        # Surveillance de /clock EN TEMPS MUR (ce nœud n'est pas en use_sim_time,
        # exprès : ses garde-temps doivent survivre à une sim figée). On note
        # l'instant mur de la dernière AVANCÉE de l'horloge sim, pas la dernière
        # réception — un bridge qui répète la même valeur est aussi une sim morte.
        self._clock_val: Optional[float] = None
        self._clock_advance_wall: Optional[float] = None
        self.create_subscription(Clock, "/clock", self._on_clock,
                                 qos_profile_sensor_data)

        # ── État partagé entre callbacks (thread ROS) et séquence (thread run) ─
        self._result_event = threading.Event()
        self._last_result: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._first_grasp: Optional[dict[str, Any]] = None
        self._n_closures = 0
        self._attached_any = False
        self._last_object_pose: Optional[tuple[float, float, float]] = None
        self._j5_total = 0
        self._j5_neg = 0
        self._j5_active = False
        self._episode_armed = False       # True entre /policy/reset et le résultat

        # ── Télémétrie ────────────────────────────────────────────────────────
        # flush_every=1 : un épisode vaut ~1 min de simulation, on ne perd pas une
        # ligne sur un Ctrl-C.
        self.p_episode = Probe("episode", EPISODE_FIELDS, flush_every=1)

        # ── Rapport ───────────────────────────────────────────────────────────
        self._episodes: list[dict[str, Any]] = []
        self._run_start: Optional[float] = None
        self._run_stamp = ""
        self._abort_reason = ""
        self._history_appended = False
        self._positions: list[dict[str, Any]] = []
        self._positions_origin = ""
        # Chemins fixés par _init_report_paths() dès l'entrée dans la campagne.
        self._report_md = ""
        self._report_csv = ""
        self._history_md = ""
        self._history_csv = ""

        self.get_logger().info(
            f"eval_orchestrator : {self.num_episodes} épisodes, seed={self.seed}, "
            f"zone={ZONE_VERSION} via {sampler_source()}, "
            f"bac=({self.place_x}, {self.place_y}) r={self.success_radius} m, "
            f"timeout politique={self.episode_timeout_s} s "
            f"(+{self.result_margin_s} s côté orchestrateur), run_id={run_id()}")

        self._thread = threading.Thread(target=self._run_campaign, daemon=True)
        self._thread.start()

    # ══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════════
    def _t_sim(self) -> Optional[float]:
        """Horodatage simulé, ou None hors `use_sim_time` (cf. gripper_shim)."""
        if not self._use_sim_time:
            return None
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_episode_result(self, msg: String) -> None:
        """Verdict de fin d'épisode publié par le nœud de politique (JSON)."""
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            self.get_logger().warn(
                f"episode_result illisible (JSON attendu) : {msg.data!r}")
            data = {"verdict": "inconnu", "reason": "JSON illisible"}
        if not isinstance(data, dict):
            data = {"verdict": "inconnu", "reason": f"charge utile {type(data).__name__}"}
        with self._lock:
            if not self._episode_armed:
                # Résultat hors épisode (retardataire du précédent) : on le trace mais
                # on ne le laisse SURTOUT pas clore l'épisode courant.
                self.get_logger().warn(f"episode_result hors épisode ignoré : {data}")
                return
            self._last_result = data
            self._episode_armed = False
        self._result_event.set()

    def _on_grasp_result(self, msg: String) -> None:
        """Bilan d'une fermeture de pince (JSON du gripper_shim)."""
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            self.get_logger().warn(f"grasp_result illisible : {msg.data!r}")
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            self._n_closures += 1
            if bool(data.get("attached")):
                self._attached_any = True
            # PREMIÈRE fermeture seulement : c'est elle qui mesure la précision de
            # l'approche. Les suivantes sont des rattrapages, comptés à part.
            if self._first_grasp is None:
                self._first_grasp = data

    def _on_object_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._last_object_pose = (msg.pose.position.x, msg.pose.position.y,
                                      msg.pose.position.z)

    def _on_clock(self, msg: Clock) -> None:
        val = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self._clock_val is None or val > self._clock_val + 1e-9:
            self._clock_val = val
            self._clock_advance_wall = time.time()

    def _sim_health(self) -> tuple[bool, Optional[float], float]:
        """(figée ?, dernière valeur t_sim, secondes MUR depuis la dernière avancée).

        « Figée » exige d'avoir déjà vu /clock avancer : au démarrage de la pile
        l'absence d'horloge est normale (c'est `_wait_for_stack` qui la couvre).
        """
        if self._clock_advance_wall is None:
            return (False, self._clock_val, 0.0)
        age = time.time() - self._clock_advance_wall
        return (age >= self.sim_freeze_abort_s, self._clock_val, age)

    def _on_joint_states(self, msg: JointState) -> None:
        """Compte le temps passé « poignet retourné » (joint5 < seuil)."""
        with self._lock:
            if not self._j5_active:
                return
            try:
                idx = list(msg.name).index(self.wrist_joint)
            except ValueError:
                return
            if idx >= len(msg.position):
                return
            self._j5_total += 1
            if msg.position[idx] < self.wrist_neg_threshold:
                self._j5_neg += 1

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers ROS
    # ══════════════════════════════════════════════════════════════════════════
    def _call(self, client, request, label: str, timeout: float = 10.0):
        """Appel de service synchrone depuis le thread de séquence (cf. record_orch.).

        Retourne None (et log une ERREUR) si le service est absent ou muet : aucun
        appel ne doit pouvoir bloquer la campagne indéfiniment.
        """
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"{label} : service indisponible ({timeout:.0f} s).")
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            self.get_logger().error(f"{label} : timeout d'appel.")
            return None
        return future.result()

    def _publish_object(self, x: float, y: float) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), self.object_z
        self._object_pub.publish(msg)

    def _force_gripper_open(self) -> bool:
        """Ouvre la pince (SetBool data=False) et rapporte le succès du service."""
        resp = self._call(self._gripper_cli, SetBool.Request(),
                          f"{self.gripper_service}(open)", timeout=5.0)
        return resp is not None and bool(getattr(resp, "success", True))

    def _teleport_object(self, x: float, y: float) -> tuple[bool, str]:
        """Téléporte l'objet et ATTEND la confirmation du shim.

        Le shim ignore la demande si la pince est fermée / l'objet attaché : la
        pince est donc FORCÉE ouverte à CHAQUE tentative (pas seulement avant la
        première — un ordre d'ouverture perdu ne doit pas coûter la série).
        Sans cette confirmation, un épisode pourrait se jouer sur la position du
        PRÉCÉDENT sans que rien ne le signale — l'erreur silencieuse la plus
        coûteuse possible pour une campagne comparative.

        Si l'horloge sim est FIGÉE, aucune tentative n'aboutira : on sort tout de
        suite avec la vraie cause au lieu de brûler retries × timeout à accuser
        la pince (le faux diagnostic du 2026-08-14 au matin).
        """
        for attempt in range(1, max(1, self.teleport_retries) + 1):
            figee, t_sim, age = self._sim_health()
            if figee:
                return (False, f"horloge sim FIGÉE à t_sim={t_sim:.3f} s depuis "
                               f"{age:.0f} s mur — gz server gelé ou en pause, la "
                               "téléportation ne peut pas aboutir")
            if not self._force_gripper_open():
                self.get_logger().warn(
                    f"  ouverture pince non confirmée (essai {attempt}) — "
                    "téléportation tentée quand même.")
            time.sleep(self.open_wait_s)
            self._publish_object(x, y)
            deadline = time.time() + self.teleport_timeout_s
            while time.time() < deadline:
                with self._lock:
                    pose = self._last_object_pose
                if pose is not None and math.hypot(pose[0] - x, pose[1] - y) <= self.teleport_tol:
                    return (True, "")
                figee, t_sim, age = self._sim_health()
                if figee:
                    return (False, f"horloge sim FIGÉE à t_sim={t_sim:.3f} s depuis "
                                   f"{age:.0f} s mur — gz server gelé ou en pause, la "
                                   "téléportation ne peut pas aboutir")
                time.sleep(0.1)
            self.get_logger().warn(
                f"  téléportation non confirmée (essai {attempt}/{self.teleport_retries}) : "
                f"cible ({x:.3f}, {y:.3f}) m, croyance shim = "
                f"{'aucune' if self._last_object_pose is None else self._last_object_pose}")
        return (False, f"objet non téléporté à ({x:.3f}, {y:.3f}) m malgré pince "
                       f"ouverte et {self.teleport_retries} tentatives, sim VIVANTE "
                       "(shim mort ? service set_pose HS ?)")

    # ══════════════════════════════════════════════════════════════════════════
    # Positions de la campagne
    # ══════════════════════════════════════════════════════════════════════════
    def _build_positions(self) -> list[dict[str, Any]]:
        """Génère les N positions de pick à partir du seed maître."""
        out: list[dict[str, Any]] = []
        for i in range(1, self.num_episodes + 1):
            s = episode_seed(self.seed, i)
            x, y = sample_pick_xy(s, self.place_x, self.place_y)
            out.append({"idx": i, "seed": s, "x": round(float(x), 6),
                        "y": round(float(y), 6)})
        return out

    def _load_positions(self, path: str) -> list[dict[str, Any]]:
        """Charge un fichier de positions ; lève ValueError si inexploitable."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "positions" not in data:
            raise ValueError("format inattendu (clé 'positions' absente)")
        fmt = int(data.get("format", POSITIONS_FORMAT))
        if fmt > POSITIONS_FORMAT:
            raise ValueError(f"format {fmt} plus récent que ce harnais ({POSITIONS_FORMAT})")
        pos = list(data["positions"])
        if len(pos) < self.num_episodes:
            raise ValueError(f"{len(pos)} positions dans le fichier < num_episodes="
                             f"{self.num_episodes}")
        if int(data.get("seed", self.seed)) != self.seed:
            # Le FICHIER fait foi : c'est tout l'intérêt de le figer. On le dit fort.
            self.get_logger().warn(
                f"positions_file écrit avec seed={data.get('seed')} ≠ paramètre "
                f"seed={self.seed} → LE FICHIER FAIT FOI (comparabilité de campagne).")
        if str(data.get("zone_version", ZONE_VERSION)) != ZONE_VERSION:
            self.get_logger().warn(
                f"positions_file tiré avec zone_version={data.get('zone_version')} ≠ "
                f"{ZONE_VERSION} — positions rejouées telles quelles (voulu), mais la "
                f"zone accessible a changé depuis.")
        return pos[:self.num_episodes]

    def _save_positions(self, path: str) -> None:
        payload = {
            "format": POSITIONS_FORMAT,
            "seed": self.seed,
            "num_episodes": self.num_episodes,
            "zone_version": ZONE_VERSION,
            "sampler": sampler_source(),
            "place": [self.place_x, self.place_y],
            "object_z": self.object_z,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "positions": self._positions,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            self.get_logger().warn(f"Écriture de {path} impossible : {exc}")

    def _prepare_positions(self) -> None:
        """Charge (si fourni et existant) ou génère les positions, puis les archive."""
        if self.positions_file and os.path.exists(self.positions_file):
            self._positions = self._load_positions(self.positions_file)
            self._positions_origin = f"rechargées depuis {os.path.abspath(self.positions_file)}"
        else:
            self._positions = self._build_positions()
            if self.positions_file:
                self._save_positions(self.positions_file)
                self._positions_origin = (f"tirées (seed={self.seed}) puis écrites dans "
                                          f"{os.path.abspath(self.positions_file)}")
            else:
                self._positions_origin = f"tirées (seed={self.seed})"
        # Copie systématique dans le dossier du run : une campagne est toujours
        # rejouable à partir de ses propres artefacts, même sans positions_file.
        self._save_positions(str(run_dir() / "positions.json"))
        self.get_logger().info(
            f"Positions de la campagne ({len(self._positions)}) : {self._positions_origin}")
        for p in self._positions:
            self.get_logger().info(
                f"  #{p['idx']:02d} seed={p['seed']} → ({p['x']:+.3f}, {p['y']:+.3f}) m "
                f"| d={math.hypot(p['x'], p['y']):.3f} m")

    # ══════════════════════════════════════════════════════════════════════════
    # Déroulé d'un épisode
    # ══════════════════════════════════════════════════════════════════════════
    def _arm_episode(self) -> None:
        """Remet à zéro les compteurs d'épisode (avant l'appel à /policy/reset)."""
        with self._lock:
            self._first_grasp = None
            self._n_closures = 0
            self._attached_any = False
            self._j5_total = 0
            self._j5_neg = 0
            self._j5_active = True
            self._episode_armed = True
        self._result_event.clear()

    def _disarm_episode(self) -> None:
        with self._lock:
            self._j5_active = False
            self._episode_armed = False

    def _run_episode(self, idx: int, seed: int, x: float, y: float) -> dict[str, Any]:
        """Joue un épisode et renvoie sa ligne de résultat (dict des colonnes)."""
        log = self.get_logger()
        t_ep0 = time.time()
        row: dict[str, Any] = {k: None for k in EPISODE_FIELDS}
        row.update({"idx": idx, "seed": seed, "pick_x": float(x), "pick_y": float(y),
                    "n_fermetures": 0, "attached_any": False, "place_ok": False})

        # 1-2. Téléportation + confirmation. L'ouverture de pince (préalable
        #      indispensable : le shim refuse de déplacer un objet saisi) est
        #      forcée PAR TENTATIVE dans `_teleport_object`.
        ok, err = self._teleport_object(x, y)
        if not ok:
            figee, _, _ = self._sim_health()
            # Sim figée = panne d'INFRASTRUCTURE (verdict à part, la campagne
            # s'arrête) ; sim vivante = épisode `invalide_teleport`, la campagne
            # CONTINUE — un incident ponctuel ne doit plus coûter la nuit.
            verdict = "sim_figee" if figee else "invalide_teleport"
            log.error(f"  ✗ {err}")
            row.update({"verdict": verdict, "reason": err,
                        "duree_mesuree_s": time.time() - t_ep0, "duree_s": 0.0})
            return row

        # 3. Armement des compteurs PUIS ré-armement de la politique
        self._arm_episode()
        t_reset = time.time()
        resp = self._call(self._reset_cli, Trigger.Request(),
                          f"{self.reset_service}", timeout=15.0)
        if resp is None:
            self._disarm_episode()
            msg = (f"{self.reset_service} muet ou absent — le nœud de politique "
                   f"n'expose pas le service (mauvaise version ?) ou il est mort.")
            log.error(f"  ✗ {msg}")
            row.update({"verdict": "erreur_reset", "reason": msg,
                        "duree_mesuree_s": time.time() - t_reset, "duree_s": 0.0})
            return row
        if not bool(getattr(resp, "success", False)):
            self._disarm_episode()
            msg = f"reset refusé : {getattr(resp, 'message', '') or 'sans message'}"
            log.error(f"  ✗ {msg}")
            row.update({"verdict": "reset_refuse", "reason": msg,
                        "duree_mesuree_s": time.time() - t_reset, "duree_s": 0.0})
            return row
        log.info(f"  politique ré-armée : {getattr(resp, 'message', '') or 'OK'}")

        # Purge des résultats retardataires : la politique repasse par HOME avant
        # d'agir, elle ne peut PAS conclure en `post_reset_grace_s`. Tout verdict reçu
        # dans cette fenêtre appartient à l'épisode précédent.
        if self.post_reset_grace_s > 0:
            time.sleep(self.post_reset_grace_s)
            if self._result_event.is_set():
                log.warn("  verdict reçu pendant la fenêtre de grâce → ignoré "
                         "(reste de l'épisode précédent).")
                self._result_event.clear()
                with self._lock:
                    self._last_result = {}
                    self._episode_armed = True

        # 4. Attente du verdict (garde-temps orchestrateur > timeout de la politique)
        wait_s = self.episode_timeout_s + self.result_margin_s
        log.info(f"  épisode en cours (verdict attendu sous {wait_s:.0f} s)…")
        got = self._result_event.wait(timeout=wait_s)
        t_end = time.time()
        self._disarm_episode()

        # 5. Laisse le shim republier la pose finale (5 Hz) avant de la lire
        time.sleep(self.final_pose_wait_s)

        with self._lock:
            result = dict(self._last_result)
            grasp = dict(self._first_grasp) if self._first_grasp else None
            n_closures = self._n_closures
            attached_any = self._attached_any
            pose = self._last_object_pose
            j5_total, j5_neg = self._j5_total, self._j5_neg

        if got:
            verdict = str(result.get("verdict", "inconnu"))
            reason = str(result.get("reason", ""))
            duree = result.get("duration_s")
            duree = float(duree) if isinstance(duree, (int, float)) else (t_end - t_reset)
        else:
            verdict = "timeout_orchestrateur"
            reason = (f"aucun {self.result_topic} après {wait_s:.0f} s "
                      f"(politique bloquée ou n'implémentant pas le contrat)")
            duree = t_end - t_reset
            log.warn(f"  ✗ {reason}")

        row["verdict"] = verdict
        row["reason"] = reason
        row["duree_s"] = duree
        row["duree_mesuree_s"] = t_end - t_ep0
        row["n_fermetures"] = n_closures
        row["attached_any"] = attached_any
        row["n_ech_j5"] = j5_total
        row["pct_j5_neg"] = (100.0 * j5_neg / j5_total) if j5_total else None

        if grasp is not None:
            row["attached"] = bool(grasp.get("attached"))
            for k in ("dx", "dy", "dz", "dxy", "dist"):
                v = grasp.get(k)
                row[k] = float(v) if isinstance(v, (int, float)) else None
        else:
            # Aucune fermeture : la politique n'a jamais tenté de saisir. C'est un
            # échec d'un genre à part (≠ « fermé à côté »), on le laisse visible.
            row["attached"] = False

        if pose is not None:
            row["obj_final_x"], row["obj_final_y"], row["obj_final_z"] = pose
            d_bac = math.hypot(pose[0] - self.place_x, pose[1] - self.place_y)
            row["dist_bac"] = d_bac
            row["place_ok"] = bool(d_bac <= self.success_radius)

        log.info(
            f"  → verdict={verdict} · saisie={'OUI' if row['attached'] else 'non'} "
            f"· dxy={_fmt(row['dxy'])} m · dz={_fmt(row['dz'])} m "
            f"· fermetures={n_closures} · dist_bac={_fmt(row['dist_bac'])} m "
            f"· j5<0 {_fmt(row['pct_j5_neg'], 1)} % · {duree:.1f} s")
        return row

    # ══════════════════════════════════════════════════════════════════════════
    # Campagne
    # ══════════════════════════════════════════════════════════════════════════
    def _wait_for_stack(self) -> bool:
        """Vérifie bruyamment que la pile répond AVANT de brûler des épisodes."""
        log = self.get_logger()
        if self.settle_time > 0:
            log.info(f"Attente de mise en place de la pile ({self.settle_time:.0f} s)…")
            time.sleep(self.settle_time)

        # a) pose objet : le shim est vivant et son service set_pose fonctionne
        log.info(f"Attente d'un premier {self.object_pose_topic} "
                 f"(max {self.object_pose_timeout_s:.0f} s)…")
        deadline = time.time() + self.object_pose_timeout_s
        while time.time() < deadline and self._last_object_pose is None and rclpy.ok():
            time.sleep(0.2)
        if self._last_object_pose is None:
            self._abort_reason = (
                f"aucun {self.object_pose_topic} reçu : gripper_shim absent ou muet. "
                "Vérifie le bring-up (./kill_all.sh puis relance, un seul lancement).")
            log.error(f"❌ ABANDON : {self._abort_reason}")
            return False

        # b) service de reset de la politique : c'est le pivot du protocole. Le
        #    chargement d'un checkpoint SmolVLA prend souvent > 60 s en CPU, d'où
        #    un délai d'attente volontairement large.
        log.info(f"Attente du service {self.reset_service} "
                 f"(max {self.ready_timeout_s:.0f} s, chargement du checkpoint)…")
        if not self._reset_cli.wait_for_service(timeout_sec=self.ready_timeout_s):
            self._abort_reason = (
                f"service {self.reset_service} absent après {self.ready_timeout_s:.0f} s : "
                "le nœud de politique n'a pas démarré (checkpoint introuvable ?) ou "
                "n'expose pas le service de reset attendu par le protocole.")
            log.error(f"❌ ABANDON : {self._abort_reason}")
            return False
        log.info("Pile prête → démarrage de la campagne.")
        return True

    def _run_campaign(self) -> None:
        log = self.get_logger()
        self._run_start = time.time()
        self._init_report_paths()

        try:
            self._prepare_positions()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._abort_reason = f"positions inutilisables : {exc}"
            log.error(f"❌ ABANDON : {self._abort_reason}")
            self._finish(complete=True)
            return

        write_meta("eval_orchestrator", {
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "positions_file": os.path.abspath(self.positions_file) if self.positions_file else "",
            "positions_origin": self._positions_origin,
            "zone_version": ZONE_VERSION,
            "sampler": sampler_source(),
            "model_label": self.model_label,
            "place": [self.place_x, self.place_y],
            "success_radius": self.success_radius,
            "object_z": self.object_z,
            "episode_timeout_s": self.episode_timeout_s,
            "result_timeout_s": self.episode_timeout_s + self.result_margin_s,
            "sim_freeze_abort_s": self.sim_freeze_abort_s,
            "wrist_joint": self.wrist_joint,
            "report_root": os.path.abspath(self.report_root),
            "use_sim_time": self._use_sim_time,
        })

        if not self._wait_for_stack():
            self._finish(complete=True)
            return

        n_done = 0                 # épisodes ayant produit un verdict de politique
        consecutive_timeouts = 0
        consecutive_teleport = 0

        for p in self._positions:
            if not rclpy.ok():
                self._abort_reason = "arrêt demandé (rclpy)."
                break
            idx, seed = int(p["idx"]), int(p["seed"])
            x, y = float(p["x"]), float(p["y"])
            log.info(f"╔═══ Épisode {idx}/{len(self._positions)} — pick "
                     f"({x:+.3f}, {y:+.3f}) m · seed={seed} ═══╗")
            row = self._run_episode(idx, seed, x, y)
            self._episodes.append(row)
            self.p_episode.log(t_sim=self._t_sim(), **row)
            # Rapport réécrit après CHAQUE épisode : une campagne interrompue reste
            # exploitable (20-30 min de simulation ne doivent jamais partir en fumée).
            try:
                self._write_report(complete=False)
            except Exception as exc:  # noqa: BLE001
                log.warn(f"Écriture du rapport échouée : {exc}")

            verdict = row["verdict"]
            est_teleport = verdict in ("invalide_teleport", "erreur_teleport")
            est_muet = verdict in ("timeout_orchestrateur", "erreur_reset", "reset_refuse")
            consecutive_teleport = consecutive_teleport + 1 if est_teleport else 0
            if est_muet:
                consecutive_timeouts += 1
            elif not est_teleport:
                # Un échec de téléportation ne prouve ni ne réfute que la
                # politique répond : il ne touche pas au compteur de mutisme.
                consecutive_timeouts = 0
            if not est_teleport and not est_muet and verdict != "sim_figee":
                n_done += 1                      # verdict rendu PAR la politique

            log.info(f"╚═══ Épisode {idx} terminé ({verdict}) ═══╝")

            # Sim figée : panne d'infrastructure, aucune récupération possible
            # sans relancer la pile — on s'arrête avec la cause exacte (et pas
            # « pince fermée ? » comme le 2026-08-14 au matin).
            if verdict == "sim_figee":
                self._abort_reason = (
                    f"{row['reason']} — ./kill_all.sh puis relance complète requise "
                    "(un superviseur peut redémarrer la campagne sur les épisodes restants).")
                log.error(f"❌ ABANDON : {self._abort_reason}")
                break
            # Fail-fast : rien ne répond au démarrage ⇒ on arrête tout de suite au
            # lieu de dérouler 20 épisodes vides (même logique que record_orchestrator).
            if n_done == 0 and consecutive_timeouts >= ABORT_TIMEOUTS:
                self._abort_reason = (
                    f"{consecutive_timeouts} épisodes consécutifs sans verdict au démarrage : "
                    "la politique ne répond pas (bring-up cassé). ./kill_all.sh puis relance.")
                log.error(f"❌ ABANDON : {self._abort_reason}")
                break
            if consecutive_teleport >= ABORT_TELEPORTS:
                self._abort_reason = (
                    f"{consecutive_teleport} téléportations échouées d'affilée SIM VIVANTE : "
                    "l'objet ne bouge plus (shim mort ? service set_pose HS ?). Campagne "
                    "non comparable, arrêt.")
                log.error(f"❌ ABANDON : {self._abort_reason}")
                break

            if idx < len(self._positions):
                time.sleep(self.inter_episode_delay)

        self._finish(complete=True)

    def _finish(self, complete: bool) -> None:
        log = self.get_logger()
        try:
            self._write_report(complete=complete)
            log.info(f"📄 Rapport : {self._report_md}")
            log.info(f"📄 CSV : {self._report_csv}")
            log.info(f"📄 Télémétrie : {self.p_episode.path}")
        except Exception as exc:  # noqa: BLE001
            log.warn(f"Écriture du rapport final échouée : {exc}")
        self.p_episode.close()
        if self.shutdown_when_done and rclpy.ok():
            rclpy.shutdown()

    # ══════════════════════════════════════════════════════════════════════════
    # Rapport (même structure à deux niveaux que record_orchestrator)
    # ══════════════════════════════════════════════════════════════════════════
    def _init_report_paths(self) -> None:
        """Rapport LIVE réécrit à chaque épisode + HISTORIQUE cumulatif en append."""
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.abspath(self.report_root)
        try:
            os.makedirs(root, exist_ok=True)
        except OSError as exc:
            self.get_logger().warn(f"Création du dossier rapport échouée ({exc}).")
        self._report_md = os.path.join(root, "rapport_eval_courant.md")
        self._report_csv = os.path.join(root, "rapport_eval_courant.csv")
        self._history_md = os.path.join(root, "historique_evals.md")
        self._history_csv = os.path.join(root, "historique_evals.csv")

    def _stats(self) -> dict[str, Any]:
        """Agrégats de la campagne (dont le couple dxy / dz qui oriente le diagnostic)."""
        eps = self._episodes
        n = len(eps)
        graspes = [e for e in eps if e.get("dxy") is not None]
        dxy = [e["dxy"] for e in graspes]
        dz_signed = [e["dz"] for e in graspes if e.get("dz") is not None]
        dz_abs = [abs(v) for v in dz_signed]
        pj5 = [e["pct_j5_neg"] for e in eps if e.get("pct_j5_neg") is not None]
        durees = [e["duree_s"] for e in eps if e.get("duree_s") is not None]
        verdicts: dict[str, int] = {}
        for e in eps:
            v = str(e.get("verdict"))
            verdicts[v] = verdicts.get(v, 0) + 1
        return {
            "n": n,
            "n_pick": sum(1 for e in eps if e.get("attached")),
            "n_pick_any": sum(1 for e in eps if e.get("attached_any")),
            "n_place": sum(1 for e in eps if e.get("place_ok")),
            "n_succes_politique": sum(1 for e in eps if e.get("verdict") == "success"),
            "n_sans_fermeture": sum(1 for e in eps if not e.get("n_fermetures")),
            "n_multi_fermetures": sum(1 for e in eps if (e.get("n_fermetures") or 0) >= 2),
            "n_mesures": len(graspes),
            "dxy_med": _median(dxy), "dxy_p90": _percentile(dxy, 0.90),
            "dz_abs_med": _median(dz_abs), "dz_abs_p90": _percentile(dz_abs, 0.90),
            "dz_med_signe": _median(dz_signed),
            "pct_j5_neg_moy": (sum(pj5) / len(pj5)) if pj5 else None,
            "duree_moy": (sum(durees) / len(durees)) if durees else None,
            "verdicts": verdicts,
        }

    def _write_report(self, complete: bool) -> None:
        """(Ré)écrit le rapport Markdown + le CSV par épisode, puis l'historique."""
        eps = self._episodes
        st = self._stats()
        n = st["n"]
        now = time.time()
        total_s = now - (self._run_start or now)

        def pct(k: int) -> str:
            return f"{(100.0 * k / n):.1f} %" if n else "—"

        def hms(s: float) -> str:
            s = int(s)
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

        L: list[str] = []
        L.append(f"# Rapport d'évaluation — run {self._run_stamp}")
        L.append("")
        L.append(f"**Statut :** {'✅ TERMINÉ' if complete else '⏳ EN COURS'}  ")
        if self._abort_reason:
            L.append(f"**⛔ CAMPAGNE ABANDONNÉE :** {self._abort_reason}  ")
        L.append(f"**Modèle évalué :** `{self.model_label or '(non renseigné)'}`  ")
        L.append(f"**run_id télémétrie :** `{run_id()}` → `{run_dir()}`  ")
        L.append(f"**Seed maître :** `{self.seed}` · positions : {self._positions_origin}  ")
        L.append(f"**Zone de tirage :** `{ZONE_VERSION}` via `{sampler_source()}`  ")
        if self._run_start:
            L.append(f"**Début :** {datetime.fromtimestamp(self._run_start):%Y-%m-%d %H:%M:%S}  ")
        if complete:
            L.append(f"**Fin :** {datetime.fromtimestamp(now):%Y-%m-%d %H:%M:%S}  ")
        L.append(f"**Durée totale :** {hms(total_s)}")
        L.append("")
        L.append("## Résumé")
        L.append("")
        L.append(f"- Épisodes joués : **{n} / {self.num_episodes}**")
        L.append(f"- 🤏 **Taux de saisie** (1re fermeture, objet accroché) : "
                 f"**{st['n_pick']} / {n}** = **{pct(st['n_pick'])}**")
        L.append(f"  - dont saisi après ≥ 1 rattrapage : {st['n_pick_any'] - st['n_pick']}")
        L.append(f"- 📦 **Taux de dépose** (objet à ≤ {self.success_radius} m du bac, "
                 f"vérité-terrain) : **{st['n_place']} / {n}** = **{pct(st['n_place'])}**")
        L.append(f"- 🏷️ Verdicts « success » annoncés par la politique : "
                 f"{st['n_succes_politique']} / {n}")
        L.append(f"- 🚫 Épisodes SANS aucune fermeture de pince : {st['n_sans_fermeture']}")
        L.append(f"- 🔁 Épisodes avec ≥ 2 fermetures (tâtonnement) : {st['n_multi_fermetures']}")
        L.append(f"- 🔄 Poignet retourné ({self.wrist_joint} < "
                 f"{self.wrist_neg_threshold}) : {_fmt(st['pct_j5_neg_moy'], 1)} % du temps "
                 f"(moyenne des épisodes)")
        L.append(f"- ⏱️ Durée moyenne d'épisode : {_fmt(st['duree_moy'], 1)} s")
        if st["verdicts"]:
            detail = " · ".join(f"{k} = {v}" for k, v in
                                sorted(st["verdicts"].items(), key=lambda kv: -kv[1]))
            L.append(f"- 🔎 Répartition des verdicts : {detail}")
        L.append("")
        L.append("## Erreur de fermeture de pince (1re fermeture)")
        L.append("")
        L.append(f"Mesures disponibles : **{st['n_mesures']} / {n}** épisodes.")
        L.append("")
        L.append("| Grandeur | Médiane | P90 |")
        L.append("|:--|--:|--:|")
        L.append(f"| `dxy` — erreur LATÉRALE (m) | {_fmt(st['dxy_med'], 4)} "
                 f"| {_fmt(st['dxy_p90'], 4)} |")
        L.append(f"| `|dz|` — erreur VERTICALE (m) | {_fmt(st['dz_abs_med'], 4)} "
                 f"| {_fmt(st['dz_abs_p90'], 4)} |")
        L.append("")
        L.append(f"Biais vertical signé (médiane de `dz`, objet − pince) : "
                 f"**{_fmt(st['dz_med_signe'], 4)} m** — négatif = la pince ferme "
                 f"AU-DESSUS de l'objet.")
        L.append("")
        L.append("> Lecture : un `|dz|` élevé avec un `dxy` faible pointe le SUIVI de "
                 "trajectoire (la politique vise juste mais la commande n'y va pas — cf. "
                 "le bug `action_duration`). Un `dxy` élevé, surtout s'il grandit avec "
                 "l'excentricité de l'objet, pointe l'ANCRAGE VISUEL (la politique ne "
                 "localise pas l'objet).")
        L.append("")
        L.append("## Paramètres de la campagne")
        L.append("")
        L.append(f"- num_episodes : `{self.num_episodes}` · seed : `{self.seed}`")
        L.append(f"- positions_file : "
                 f"`{os.path.abspath(self.positions_file) if self.positions_file else '(aucun)'}`")
        L.append(f"- bac (place) : ({self.place_x}, {self.place_y}) m · "
                 f"success_radius : {self.success_radius} m")
        L.append(f"- timeout politique : {self.episode_timeout_s} s · "
                 f"garde-temps orchestrateur : "
                 f"{self.episode_timeout_s + self.result_margin_s} s")
        L.append(f"- object_z : {self.object_z} m · tolérance téléportation : "
                 f"{self.teleport_tol} m")
        L.append("")
        L.append("## Détail par épisode")
        L.append("")
        L.append("| #  | pick (x, y) m | verdict | saisie | dxy (m) | dz (m) | ferm. "
                 "| dist. bac (m) | j5<0 | durée (s) | cause |")
        L.append("|---:|:-------------:|:--------|:------:|--------:|-------:|-----:"
                 "|--------------:|-----:|----------:|:------|")
        for e in eps:
            saisie = "✅" if e.get("attached") else ("🔁" if e.get("attached_any") else "❌")
            L.append(
                f"| {e['idx']} | ({e['pick_x']:.3f}, {e['pick_y']:.3f}) | {e['verdict']} "
                f"| {saisie} | {_fmt(e.get('dxy'), 4)} | {_fmt(e.get('dz'), 4)} "
                f"| {e.get('n_fermetures') or 0} | {_fmt(e.get('dist_bac'))} "
                f"| {_fmt(e.get('pct_j5_neg'), 0)}% | {_fmt(e.get('duree_s'), 1)} "
                f"| {str(e.get('reason') or '').replace('|', '/')} |")
        L.append("")
        L.append(f"**TOTAL — 🤏 saisies : {st['n_pick']}/{n} · 📦 déposes : "
                 f"{st['n_place']}/{n} · joués : {n}/{self.num_episodes}**")
        L.append("")

        with open(self._report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(L))

        rows = [",".join(EPISODE_FIELDS)]
        for e in eps:
            rows.append(",".join(self._csv_cell(e.get(k)) for k in EPISODE_FIELDS))
        with open(self._report_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

        if complete and not self._history_appended:
            self._append_history(L, rows)
            self._history_appended = True

    @staticmethod
    def _csv_cell(v: Any) -> str:
        """Sérialise une valeur pour le CSV (vide = non mesuré, jamais 0 par défaut)."""
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v).replace(",", ";")

    def _append_history(self, md_lines: list[str], csv_rows: list[str]) -> None:
        """Ajoute la campagne terminée aux fichiers cumulatifs (comparaison de modèles)."""
        try:
            sep = "" if not os.path.exists(self._history_md) else "\n\n---\n\n"
            with open(self._history_md, "a", encoding="utf-8") as f:
                f.write(sep + "\n".join(md_lines) + "\n")
            new_csv = not os.path.exists(self._history_csv)
            model = (self.model_label or "").replace(",", ";")
            with open(self._history_csv, "a", encoding="utf-8") as f:
                if new_csv:
                    # `seed_maitre` (campagne) ≠ `seed` (épisode) : deux colonnes
                    # homonymes rendraient l'historique ambigu dans un tableur.
                    f.write("run,run_id,model,seed_maitre," + csv_rows[0] + "\n")
                for r in csv_rows[1:]:
                    f.write(f"{self._run_stamp},{run_id()},{model},{self.seed},{r}\n")
            self.get_logger().info(
                f"Historique mis à jour : {self._history_md} (+{len(csv_rows) - 1} ép.)")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Append historique échoué : {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvalOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.p_episode.close()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
