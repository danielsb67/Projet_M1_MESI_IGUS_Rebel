"""
gym_igus.py — Environnement gymnasium.Env au-dessus de la sim Gazebo (RL, HIL-SERL).

POURQUOI
========
La politique SmolVLA v2_2 plafonne à 65 % en imitation pure ; l'étage suivant est
le RL avec récompense succès/échec (audit PROGRESS.md §E : lerobot 0.4.4 embarque
HIL-SERL — SAC + learner RLPD dans lerobot/rl/). Il manquait le pont : un
`gymnasium.Env` (API 1.3.0 : `step → (obs, reward, terminated, truncated, info)`,
`reset(*, seed, options) → (obs, info)`) au-dessus de la pile ROS 2 existante.

ARCHITECTURE (décisions, cf. audit 2026-08-13)
==============================================
* SAC agit en PAS UNITAIRE (une action par tick), il ne peut PAS s'initialiser
  depuis SmolVLA (chunks de 50). L'action garde néanmoins la MÊME sémantique que
  le dataset : positions articulaires ABSOLUES + pince, cible à atteindre en
  1/15 s — ainsi les démos LeRobot existantes sont directement rejouables comme
  transitions offline (façon RLPD, learner.py:331-348).
* L'env réutilise l'existant éprouvé, jamais réinventé :
  - E/S robot : `GazeboBackend` (souscriptions capteurs, trajectoires datées
    temps sim, service pince) ;
  - téléport objet avec CONFIRMATION + détection d'horloge sim figée : pattern
    `eval_orchestrator` (le faux diagnostic du 2026-08-14 ne doit pas se rejouer) ;
  - retour HOME par trajectoire INTERPOLÉE depuis la mesure + resynchronisation
    du contrôleur (`switch_controller`) : pattern `vla_policy_node._go_home` /
    `_resync_controller` (contrôleur en open_loop_control : sans resync, un
    épisode divergent contamine tous les suivants — écart figé mesuré 1,4 rad) ;
  - tirage de la position de pick : `eval_orchestrator.sample_pick_xy`, qui
    appelle la fonction CANONIQUE `RecordOrchestrator._sample_object_xy` via une
    doublure (même zone anneau 0,15–0,54 m que les datasets, ~98-100 % d'IK).
* TOUTES les attentes sont bornées en temps MUR (leçon payée : l'horloge sim se
  fige parfois — gz server gelé → timers sim muets, processus sains). Une sim
  figée lève `HorlogeSimFigeeError` avec la cause exacte, jamais un blocage muet.

ESPACES
=======
observation : Dict{
    "front" : Box uint8 (480, 640, 3)   — caméra frontale RGB
    "wrist" : Box uint8 (480, 640, 3)   — caméra poignet RGB
    "state" : Box float32 (7,)          — j1..j6 (rad) + pince (0.0/1.0)
}
action : Box float32 (7,) — j1..j6 cibles ABSOLUES (rad, bornées URDF) +
         commande pince (≥ 0,5 = fermer, sémantique dataset VLA_PLAN §3).

UTILISATION
===========
    env = GymIgusPickPlace()          # crée son nœud + son exécuteur (thread)
    obs, info = env.reset(seed=42)    # HOME + téléport objet reproductible
    obs, r, term, trunc, info = env.step(action)
    env.close()

Un nœud rclpy peut aussi être INJECTÉ (`GymIgusPickPlace(node=mon_noeud)`) ; il
doit alors être spinné PAR L'APPELANT (exécuteur externe), l'env ne spinne que
les nœuds qu'il possède. Le nœud doit vivre en use_sim_time (trajectoires datées
temps sim, cf. GazeboBackend stamp_mode="now").
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
import rclpy.node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import SetBool

try:                                    # absent d'une pile sans ros2_control
    from controller_manager_msgs.srv import SwitchController
except ImportError:                     # pragma: no cover
    SwitchController = None

from tf2_ros import Buffer, TransformListener, TransformException

from igus_vla.backends.gazebo import GazeboBackend
# Pose HOME : LA MÊME que le déploiement (source unique, pas de copie qui diverge).
from igus_vla.vla_policy_node import HOME_JOINTS
# Tirage pick : passe par la fonction canonique de record_orchestrator (doublure),
# repli copie fidèle — exactement le mécanisme déjà validé par eval_orchestrator.
from igus_vla.eval_orchestrator import sample_pick_xy
from igus_vla.rl.recompense import CalculateurRecompense, ConfigRecompense

# ── Limites articulaires URDF (igus_rebel.description.xacro, lignes 38-192) ──
# j1/j4/j6 : ±179° ; j2/j3 : −80°..+140° ; j5 : ±90°. Vitesse max π/4 rad/s
# (le plafond de feedforward de GazeboBackend est calé dessus).
_LIM_BAS = [-math.pi * 179 / 180, -math.pi * 80 / 180, -math.pi * 80 / 180,
            -math.pi * 179 / 180, -math.pi / 2, -math.pi * 179 / 180]
_LIM_HAUT = [math.pi * 179 / 180, math.pi * 140 / 180, math.pi * 140 / 180,
             math.pi * 179 / 180, math.pi / 2, math.pi * 179 / 180]

# Seuil pince : sémantique dataset (VLA_PLAN §3, observation.apply_gripper).
_SEUIL_PINCE = 0.5


class HorlogeSimFigeeError(RuntimeError):
    """L'horloge /clock n'avance plus (gz server gelé/en pause).

    Aucune attente sim n'aboutira : trajectoires non consommées, shim muet,
    set_pose HS. Le SEUL remède est ./kill_all.sh + relance complète de la pile
    (cf. panne du 2026-08-14 : accuser la pince était un faux diagnostic).
    """


class GymIgusPickPlace(gym.Env):
    """Pick & place IGUS ReBeL 6 axes en sim Gazebo, API gymnasium 1.3.0.

    Parameters
    ----------
    node : rclpy.node.Node, optional
        Nœud ROS à réutiliser (spinné PAR L'APPELANT, use_sim_time requis).
        ``None`` (défaut) = l'env crée son nœud + un MultiThreadedExecutor
        dans un thread démon (et fait rclpy.init() si nécessaire).
    control_hz : float
        Cadence de contrôle (défaut 15 Hz = celle du dataset ; un pas d'env
        = 1/15 s de temps SIM).
    max_steps : int
        Plafond de pas avant `truncated` (défaut 675 = 45 s sim à 15 Hz :
        2,5× le cycle expert de 17,9 s — assez pour un rattrapage, pas assez
        pour errer indéfiniment).
    config_recompense : ConfigRecompense, optional
        Réglages de récompense (mode dense/sparse…). Défaut = dense.
    place_xy : tuple
        Centre du bac (m) — mêmes défauts que la pile ((0.0, 0.25)).
    rayon_succes : float
        Rayon de dépose réussie autour du bac (0,12 m = eval_orchestrator).
    object_z : float
        Hauteur de repos de l'objet (0,018 m = gripper_shim).
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        node: Optional[rclpy.node.Node] = None,
        *,
        control_hz: float = 15.0,
        max_steps: int = 675,
        config_recompense: Optional[ConfigRecompense] = None,
        place_xy: Tuple[float, float] = (0.0, 0.25),
        rayon_succes: float = 0.12,
        object_z: float = 0.018,
        image_topics: Optional[Dict[str, str]] = None,
        base_frame: str = "base_link",
        ee_frame: str = "gripper_tip_link",
        arm_controller: str = "rebel_arm_controller",
        # ── Garde-fous temps MUR (l'horloge sim n'est JAMAIS une garde) ──────
        sim_freeze_s: float = 10.0,        # /clock immobile au-delà → sim figée
        step_wall_timeout_s: float = 30.0,  # plafond mur d'UN tick sim (RTF lent toléré)
        ready_timeout_s: float = 120.0,     # 1re disponibilité capteurs + horloge
        home_timeout_wall_s: float = 90.0,  # convergence du retour HOME
        teleport_timeout_s: float = 10.0,   # confirmation d'UNE téléportation
        teleport_retries: int = 3,
        open_wait_s: float = 1.5,           # délai après ouverture pince (eval)
    ) -> None:
        super().__init__()

        self._dt = 1.0 / max(1e-3, float(control_hz))
        self._max_steps = int(max_steps)
        self._place_x, self._place_y = float(place_xy[0]), float(place_xy[1])
        self._rayon_succes = float(rayon_succes)
        self._object_z = float(object_z)
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._arm_controller = arm_controller
        self._sim_freeze_s = float(sim_freeze_s)
        self._step_wall_timeout_s = float(step_wall_timeout_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._home_timeout_wall_s = float(home_timeout_wall_s)
        self._teleport_timeout_s = float(teleport_timeout_s)
        self._teleport_retries = max(1, int(teleport_retries))
        self._open_wait_s = float(open_wait_s)

        # ── Nœud ROS : injecté (spinné dehors) ou possédé (spinné ici) ───────
        self._owns_rclpy = False
        self._owns_node = node is None
        if node is None:
            if not rclpy.ok():
                rclpy.init()
                self._owns_rclpy = True
            # use_sim_time OBLIGATOIRE : GazeboBackend date les trajectoires sur
            # l'horloge du nœud (stamp_mode="now") ; un stamp mur serait lu par
            # le contrôleur (temps sim) comme très ancien → saut de trajectoire.
            node = rclpy.node.Node(
                "gym_igus_env",
                parameter_overrides=[Parameter("use_sim_time", value=True)])
        self._node = node

        # ── Backend robot (souscriptions capteurs isolées, groupe réentrant) ─
        cb_group = ReentrantCallbackGroup()
        self._backend = GazeboBackend(
            node,
            image_topics=dict(image_topics) if image_topics
            else {"front": "/front_camera/image", "wrist": "/wrist_camera/image"},
            stamp_mode="now",
            callback_group=cb_group,
        )

        # ── E/S propres à l'env ───────────────────────────────────────────────
        self._lock = threading.Lock()
        # /clock EN TEMPS MUR : on note l'instant mur de la dernière AVANCÉE de
        # l'horloge sim (pattern eval_orchestrator — un bridge qui répète la même
        # valeur est aussi une sim morte).
        self._clock_val: Optional[float] = None
        self._clock_advance_mono: Optional[float] = None
        node.create_subscription(Clock, "/clock", self._on_clock,
                                 qos_profile_sensor_data, callback_group=cb_group)
        # Vérité terrain objet (croyance du shim, 5 Hz).
        self._pose_objet: Optional[Tuple[float, float, float]] = None
        node.create_subscription(PoseStamped, "/gripper/object_pose",
                                 self._on_object_pose, 10, callback_group=cb_group)
        # Bilan de chaque fermeture de pince (JSON, contrat gripper_shim :
        # clés toujours présentes — dx, dy, dz, dist, dxy, attached, radius).
        self._n_fermetures = 0
        self._evts_saisie = 0            # compteur cumulatif d'attached=True
        self._saisies_traitees = 0       # filigrane : évts déjà consommés par step()
        self._objet_saisi = False
        node.create_subscription(String, "/gripper/grasp_result",
                                 self._on_grasp_result, 10, callback_group=cb_group)
        # Téléport objet : MÊME topic que l'orchestrateur (le shim s'en charge).
        self._object_pub = node.create_publisher(
            PointStamped, "/object_position_in_world", 10)
        # Ouverture pince CONFIRMÉE au reset (le backend fait du fire-and-forget,
        # suffisant à 15 Hz en step, insuffisant avant un téléport : le shim
        # IGNORE toute téléportation pince fermée/objet attaché).
        self._gripper_cli = node.create_client(SetBool, "/gripper/command")
        # Resynchronisation du contrôleur (pattern vla_policy_node — best effort).
        self._switch_cli = None
        if SwitchController is not None:
            self._switch_cli = node.create_client(
                SwitchController, "/controller_manager/switch_controller",
                callback_group=cb_group)
        # TF : pose de la pointe de pince (même paire de frames que gripper_shim).
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        # ── Exécuteur (seulement si le nœud est à nous) ──────────────────────
        self._executor: Optional[MultiThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        if self._owns_node:
            self._executor = MultiThreadedExecutor(num_threads=2)
            self._executor.add_node(node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, daemon=True, name="gym_igus_spin")
            self._spin_thread.start()

        # ── Récompense + état d'épisode ──────────────────────────────────────
        self._calculateur = CalculateurRecompense(
            config=config_recompense or ConfigRecompense())
        self._n_pas = 0
        self._pince_cmd_fermee = False
        self._reset_fait = False
        self._pick_xy: Optional[Tuple[float, float]] = None

        # ── Espaces gymnasium (signatures 1.3.0 vérifiées) ───────────────────
        img_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "front": img_space,
            "wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
            "state": spaces.Box(
                low=np.array(_LIM_BAS + [0.0], dtype=np.float32),
                high=np.array(_LIM_HAUT + [1.0], dtype=np.float32),
                dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array(_LIM_BAS + [0.0], dtype=np.float32),
            high=np.array(_LIM_HAUT + [1.0], dtype=np.float32),
            dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Callbacks (thread exécuteur)
    # ══════════════════════════════════════════════════════════════════════════
    def _on_clock(self, msg: Clock) -> None:
        val = msg.clock.sec + msg.clock.nanosec * 1e-9
        with self._lock:
            if self._clock_val is None or val > self._clock_val + 1e-9:
                self._clock_val = val
                self._clock_advance_mono = time.monotonic()

    def _on_object_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._pose_objet = (msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z)

    def _on_grasp_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            self._node.get_logger().warn(f"grasp_result illisible : {msg.data!r}")
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            self._n_fermetures += 1
            if bool(data.get("attached")):
                self._evts_saisie += 1
                self._objet_saisi = True

    # ══════════════════════════════════════════════════════════════════════════
    # Gardes temps mur / horloge sim
    # ══════════════════════════════════════════════════════════════════════════
    def _t_sim(self) -> Optional[float]:
        with self._lock:
            return self._clock_val

    def _verifier_sim_vivante(self) -> None:
        """Lève HorlogeSimFigeeError si /clock n'avance plus (garde MUR).

        « Figée » exige d'avoir déjà vu l'horloge avancer : au démarrage,
        l'absence de /clock est normale (couverte par `_attendre_pile_prete`).
        """
        with self._lock:
            avance, val = self._clock_advance_mono, self._clock_val
        if avance is None:
            return
        age = time.monotonic() - avance
        if age >= self._sim_freeze_s:
            raise HorlogeSimFigeeError(
                f"horloge sim FIGÉE à t_sim={val:.3f} s depuis {age:.0f} s mur "
                "(gz server gelé/en pause) — ./kill_all.sh puis relance complète.")

    def _attendre(self, condition, timeout_mur: float, description: str,
                  periode: float = 0.02) -> None:
        """Attente active BORNÉE temps mur, avec surveillance de la sim.

        `condition` est un appelable sans argument → bool. Lève TimeoutError
        (cause explicite) ou HorlogeSimFigeeError — jamais d'attente muette.
        """
        deadline = time.monotonic() + float(timeout_mur)
        while time.monotonic() < deadline:
            if condition():
                return
            self._verifier_sim_vivante()
            time.sleep(periode)
        raise TimeoutError(
            f"{description} : rien après {timeout_mur:.0f} s mur.")

    def _attendre_tick(self, t_sim_depart: float) -> None:
        """Attend le tick sim suivant (t_sim_depart + dt), garde-fou mur.

        Un RTF lent est toléré jusqu'à `step_wall_timeout_s` ; une horloge
        IMMOBILE lève HorlogeSimFigeeError bien avant (sim_freeze_s < timeout).
        """
        cible = t_sim_depart + self._dt
        self._attendre(
            lambda: (self._t_sim() or 0.0) >= cible,
            self._step_wall_timeout_s,
            f"tick sim suivant (t_sim ≥ {cible:.3f} s)",
            periode=0.005)

    # ══════════════════════════════════════════════════════════════════════════
    # Mesures vérité terrain
    # ══════════════════════════════════════════════════════════════════════════
    def _pose_pince(self) -> Optional[Tuple[float, float, float]]:
        """Pose (x, y, z) de la pointe de pince — même TF que gripper_shim.

        None si la TF n'est pas encore peuplée : le shaping saute ce pas
        (recompense.py traite None comme « pas mesuré », jamais zéro).
        """
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame, self._ee_frame, Time())
        except (TransformException, Exception):  # noqa: BLE001
            return None
        tr = t.transform.translation
        return (tr.x, tr.y, tr.z)

    def _dist_pince_objet(self) -> Optional[float]:
        ee = self._pose_pince()
        with self._lock:
            obj = self._pose_objet
        if ee is None or obj is None:
            return None
        return math.dist(ee, obj)

    def _dist_bac(self) -> Optional[float]:
        with self._lock:
            obj = self._pose_objet
        if obj is None:
            return None
        return math.hypot(obj[0] - self._place_x, obj[1] - self._place_y)

    def _depose_reussie(self) -> bool:
        """Objet à ≤ rayon_succes du bac ET pince ouverte ET objet relâché.

        Même critère vérité-terrain que `place_ok` (eval_orchestrator) +
        `_object_in_bin` (vla_policy_node) : indépendant de toute
        auto-évaluation, impossible à « hacker » en survolant le bac
        pince fermée.
        """
        if self._pince_cmd_fermee:
            return False
        with self._lock:
            if self._objet_saisi:
                return False
        d = self._dist_bac()
        return d is not None and d <= self._rayon_succes

    # ══════════════════════════════════════════════════════════════════════════
    # Observation
    # ══════════════════════════════════════════════════════════════════════════
    def _construire_obs(self) -> Dict[str, np.ndarray]:
        """Observation courante {front, wrist, state} depuis le cache backend.

        NOTE fraîcheur : les images sont lues telles quelles (caméras 15 Hz,
        pas 15 Hz → une frame peut occasionnellement se répéter, mesuré 10-17 %
        de doublons côté recorder). Bloquer sur `get_image_seq` coûterait un
        tick entier ; à valider en sim si le RL y est sensible (TODO).
        """
        etat = np.concatenate([
            np.asarray(self._backend.get_joint_state(), dtype=np.float32),
            np.asarray([self._backend.get_gripper_state()], dtype=np.float32),
        ]).astype(np.float32)
        return {
            "front": self._backend.get_image("front"),
            "wrist": self._backend.get_image("wrist"),
            "state": etat,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # step
    # ══════════════════════════════════════════════════════════════════════════
    def step(self, action) -> tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """Applique UNE action (7,) et avance d'un tick sim (1/15 s).

        Sémantique action = celle du dataset : `action[:6]` = positions
        articulaires ABSOLUES à atteindre au tick suivant, `action[6]` ≥ 0,5 =
        fermer la pince. L'action est CLIPPÉE aux limites URDF (le contrôleur
        rejetterait une cible hors bornes ; le feedforward du backend plafonne
        déjà la vitesse à π/4 rad/s).
        """
        if not self._reset_fait:
            raise RuntimeError("step() avant reset() : appeler reset() d'abord.")
        t0 = self._t_sim()
        if t0 is None:
            raise RuntimeError("aucun /clock reçu — la sim n'est pas lancée ?")

        a = np.clip(np.asarray(action, dtype=np.float32).reshape(7),
                    self.action_space.low, self.action_space.high)

        # Bras : la cible est DUPLIQUÉE en 2 points (dt et 2·dt). Un mono-point
        # renouvelé à chaque tick fait battre le contrôleur entre préemption et
        # expiration (cf. backends/base.py) ; le 2e point identique laisse un
        # horizon au-delà du tick — le step suivant préempte avant l'expiration —
        # et donne au 1er point son feedforward de vitesse (différences centrées).
        q = a[:6].astype(np.float64)
        self._backend.send_joint_trajectory(np.vstack([q, q]), self._dt)

        # Pince : commande SUR CHANGEMENT uniquement (chaque fermeture déclenche
        # un cycle grasp complet dans le shim — la spammer fausserait le compteur
        # de fermetures ET le bonus de saisie). L'anti-rebond temporel du
        # déploiement (GripperDebouncer) est laissé à la politique : en RL, le
        # coût d'un battement doit rester VISIBLE dans la récompense, pas masqué.
        fermer = bool(a[6] >= _SEUIL_PINCE)
        if fermer != self._pince_cmd_fermee:
            self._backend.set_gripper(closed=fermer)
            self._pince_cmd_fermee = fermer
            if not fermer:
                with self._lock:
                    self._objet_saisi = False    # relâche (le shim fait pareil)

        # Attendre le tick sim suivant (garde mur, sim figée → exception).
        self._attendre_tick(t0)

        # Événement de saisie : FILIGRANE persistant (_saisies_traitees), PAS un
        # instantané local pris en début de pas. L'instantané absorbait tout
        # événement tombé ENTRE la lecture post-attente du pas N et celle du pas
        # N+1 (fenêtre réelle : obs + récompense + inférence de l'appelant, et
        # l'exécuteur mono-thread du shim peut y glisser sa réponse) → R_SAISIE
        # silencieusement jamais payé. Avec le filigrane, un événement inter-pas
        # est consommé au pas SUIVANT : jamais perdu, jamais compté deux fois
        # (le dédoublonnage par épisode reste dans
        # CalculateurRecompense._saisie_payee).
        with self._lock:
            evts_saisie = self._evts_saisie
            n_fermetures = self._n_fermetures
            objet_saisi = self._objet_saisi
        saisie_ce_pas = evts_saisie > self._saisies_traitees
        self._saisies_traitees = evts_saisie

        depose = self._depose_reussie()
        dist_po = self._dist_pince_objet()
        recompense, composantes = self._calculateur.calculer(
            dist_pince_objet=dist_po,
            saisie_cet_instant=saisie_ce_pas,
            depose_reussie=depose)

        self._n_pas += 1
        terminated = bool(depose)
        truncated = bool(self._n_pas >= self._max_steps) and not terminated

        info = {
            "t_sim": self._t_sim(),
            "composantes": composantes,
            "dist_pince_objet": dist_po,
            "dist_bac": self._dist_bac(),
            "objet_saisi": objet_saisi,
            "n_fermetures": n_fermetures,
            "n_pas": self._n_pas,
            "pick_xy": self._pick_xy,
        }
        return self._construire_obs(), recompense, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════════
    # reset
    # ══════════════════════════════════════════════════════════════════════════
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[Dict[str, np.ndarray], dict]:
        """Nouvel épisode : pince ouverte → resync contrôleur → HOME → téléport.

        `seed` rend l'épisode REPRODUCTIBLE (position de pick déterministe via
        `sample_pick_xy`, la même dérivation que les campagnes d'éval) ; sans
        seed, la position est tirée du générateur gymnasium (`self.np_random`).
        `options={"pick_xy": (x, y)}` force une position exacte (rejeu).
        """
        super().reset(seed=seed)          # seede self.np_random (API gymnasium)
        options = options or {}

        # 0. Pile prête : capteurs + horloge sim VIVANTE, borné temps mur.
        self._attendre_pile_prete()

        # 1. Pince ouverte CONFIRMÉE (lâche un éventuel objet tenu ; le shim
        #    refuse toute téléportation pince fermée/objet attaché).
        self._ouvrir_pince_confirmee()

        # 2. Resynchronisation du contrôleur : en open_loop_control, il interpole
        #    depuis sa DERNIÈRE CONSIGNE ; après un épisode divergent, seule la
        #    désactivation/réactivation le fait repartir de l'état MESURÉ.
        self._resync_controleur()

        # 3. Retour HOME par trajectoire interpolée + attente de CONVERGENCE.
        self._retour_home()

        # 4. Position de pick, puis téléport avec CONFIRMATION du shim.
        if "pick_xy" in options:
            x, y = float(options["pick_xy"][0]), float(options["pick_xy"][1])
        else:
            graine = int(seed) if seed is not None \
                else int(self.np_random.integers(2 ** 31 - 1))
            x, y = sample_pick_xy(graine, self._place_x, self._place_y)
        self._teleporter_objet(x, y)
        self._pick_xy = (x, y)

        # 5. Remise à zéro de l'état d'épisode.
        self._calculateur.reset()
        self._n_pas = 0
        with self._lock:
            self._n_fermetures = 0
            self._evts_saisie = 0
            self._objet_saisi = False
        self._saisies_traitees = 0       # filigrane realigné sur le compteur remis à 0
        self._reset_fait = True

        # 6. Une frame FRAÎCHE de chaque caméra avant la 1re obs (l'image du
        #    cache peut dater d'avant le téléport). Non fatal si une caméra
        #    traîne : mieux vaut une image d'un tick d'âge qu'un reset raté.
        seqs = {k: self._backend.get_image_seq(k) for k in ("front", "wrist")}
        try:
            self._attendre(
                lambda: all(self._backend.get_image_seq(k) > seqs[k] for k in seqs),
                timeout_mur=5.0, description="frame caméra fraîche post-reset")
        except TimeoutError as exc:
            self._node.get_logger().warn(f"reset : {exc} — obs sur cache.")

        info = {"pick_xy": self._pick_xy, "t_sim": self._t_sim()}
        return self._construire_obs(), info

    # ── Sous-étapes du reset ──────────────────────────────────────────────────
    def _attendre_pile_prete(self) -> None:
        """Capteurs backend + horloge sim qui AVANCE, borné `ready_timeout_s`."""
        debut_mono = time.monotonic()

        def _prete() -> bool:
            with self._lock:
                avance = self._clock_advance_mono
            return (self._backend.is_ready()
                    and avance is not None and avance >= debut_mono - 1.0)

        try:
            # Périglobal : pas de _verifier_sim_vivante ici (une horloge encore
            # muette au démarrage est normale) → attente manuelle équivalente.
            deadline = time.monotonic() + self._ready_timeout_s
            while time.monotonic() < deadline:
                if _prete():
                    return
                time.sleep(0.1)
            raise TimeoutError
        except TimeoutError:
            raise TimeoutError(
                f"pile pas prête après {self._ready_timeout_s:.0f} s mur : "
                f"backend.is_ready()={self._backend.is_ready()}, "
                f"t_sim={self._t_sim()} — sim lancée ? topics caméra corrects ? "
                "(./kill_all.sh entre deux runs, cf. hygiène DDS)") from None

    def _ouvrir_pince_confirmee(self) -> None:
        """Ouvre la pince et ATTEND la réponse du service (borné temps mur).

        `backend.set_gripper` est aussi appelé : c'est lui qui tient l'état
        interne servi dans `state[6]` — le laisser croire la pince fermée
        fausserait toutes les observations de l'épisode.
        """
        self._backend.set_gripper(closed=False)   # état interne + envoi async
        self._pince_cmd_fermee = False
        with self._lock:
            self._objet_saisi = False
        if not self._gripper_cli.wait_for_service(timeout_sec=10.0):
            self._node.get_logger().warn(
                "/gripper/command indisponible — ouverture non confirmée "
                "(le téléport échouera si la pince était fermée).")
            return
        req = SetBool.Request()
        req.data = False
        future = self._gripper_cli.call_async(req)
        try:
            self._attendre(future.done, timeout_mur=10.0,
                           description="réponse /gripper/command (ouverture)")
        except TimeoutError as exc:
            self._node.get_logger().warn(f"reset : {exc}")
        # Le shim a besoin d'un battement pour relâcher/épingler avant téléport.
        time.sleep(self._open_wait_s)

    def _resync_controleur(self) -> None:
        """Désactive puis réactive le contrôleur de bras (best effort, asynchrone).

        Copie du pattern `vla_policy_node._resync_controller` : best effort —
        un resync raté dégrade l'épisode, il ne doit pas le faire tomber.
        """
        if self._switch_cli is None or not self._switch_cli.service_is_ready():
            self._node.get_logger().warn(
                "switch_controller indisponible : pas de resynchronisation "
                "(un épisode divergent peut contaminer le suivant).",
                throttle_duration_sec=10.0)
            return

        def _req(activer: list, desactiver: list):
            r = SwitchController.Request()
            r.activate_controllers = activer
            r.deactivate_controllers = desactiver
            r.strictness = SwitchController.Request.BEST_EFFORT
            r.activate_asap = True
            return r

        def _apres_arret(_fut):
            # Réactivation APRÈS l'arrêt effectif, sinon le gestionnaire refuse.
            self._switch_cli.call_async(_req([self._arm_controller], []))

        try:
            fut = self._switch_cli.call_async(_req([], [self._arm_controller]))
            fut.add_done_callback(_apres_arret)
            # Court battement mur : laisser le cycle off/on aboutir avant
            # d'envoyer la trajectoire HOME (le switch est asynchrone).
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warn(f"resynchronisation impossible : {exc}")

    def _envoyer_home(self) -> float:
        """Envoie la trajectoire HOME interpolée depuis la MESURE ; renvoie sa durée.

        Copie du pattern `vla_policy_node._go_home` : durée proportionnelle à la
        distance restante (plancher 3 s, vitesse 0,5 rad/s), points + vitesses
        pour que le feedforward pilote quel que soit l'état interne du contrôleur.
        """
        cible = np.asarray(HOME_JOINTS, dtype=np.float64)
        q = np.asarray(self._backend.get_joint_state(), dtype=np.float64)
        err = float(np.max(np.abs(q - cible))) if q.shape == cible.shape else float("inf")
        dur = max(3.0, err / 0.5) if np.isfinite(err) else 3.0
        n = max(2, int(round(dur / self._dt)))
        alphas = np.linspace(0.0, 1.0, n + 1)[1:]      # exclut la pose courante
        chemin = q[None, :] + alphas[:, None] * (cible - q)[None, :]
        self._backend.send_joint_trajectory(chemin, self._dt)
        return dur

    def _retour_home(self) -> None:
        """HOME + attente de CONVERGENCE mesurée (pas une temporisation).

        Tolérance 0,02 rad stable 3 lectures, plateau accepté ≤ 0,10 rad après
        2 s sans progrès (résidu statique de la boucle ouverte, mesuré
        ≈ 0,065 rad), renvoi de la consigne toutes les 2 s mur — les mêmes
        constantes que `vla_policy_node` (validées en déploiement).
        """
        cible = np.asarray(HOME_JOINTS, dtype=np.float64)
        dur = self._envoyer_home()
        # Plafond mur : généreux (RTF < 1 possible) mais jamais infini.
        deadline = time.monotonic() + max(self._home_timeout_wall_s, 2.0 * dur + 10.0)
        ok_lectures = 0
        meilleure = float("inf")
        t_meilleure = time.monotonic()
        dernier_envoi = time.monotonic()
        while time.monotonic() < deadline:
            self._verifier_sim_vivante()
            q = np.asarray(self._backend.get_joint_state(), dtype=np.float64)
            err = (float(np.max(np.abs(q - cible)))
                   if q.shape == cible.shape else float("inf"))
            if err <= 0.02:
                ok_lectures += 1
                if ok_lectures >= 3:
                    return
            else:
                ok_lectures = 0
            if err < meilleure - 0.005:
                meilleure = err
                t_meilleure = time.monotonic()
            elif err <= 0.10 and time.monotonic() - t_meilleure >= 2.0:
                self._node.get_logger().info(
                    f"HOME : résidu stable à {err:.4f} rad — la boucle ouverte "
                    "ne fera pas mieux, on continue.")
                return
            if time.monotonic() - dernier_envoi >= 2.0:
                dur = self._envoyer_home()   # ré-interpole depuis la mesure
                dernier_envoi = time.monotonic()
            time.sleep(0.1)
        raise TimeoutError(
            f"retour HOME non convergé (résidu {meilleure:.3f} rad) après "
            f"{self._home_timeout_wall_s:.0f} s mur — contrôleur sourd ? "
            "(un kill -9 antérieur coince gz_ros2_control : restart sim complet)")

    def _teleporter_objet(self, x: float, y: float) -> None:
        """Téléporte l'objet et ATTEND la confirmation du shim (pattern eval).

        Sans confirmation, un épisode pourrait se jouer sur la position du
        PRÉCÉDENT sans que rien ne le signale. Sim figée → exception dédiée
        immédiate (pas retries × timeout à accuser la pince).
        """
        for essai in range(1, self._teleport_retries + 1):
            self._verifier_sim_vivante()
            msg = PointStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = self._base_frame
            msg.point.x, msg.point.y, msg.point.z = float(x), float(y), self._object_z
            self._object_pub.publish(msg)

            def _confirme() -> bool:
                with self._lock:
                    pose = self._pose_objet
                return (pose is not None
                        and math.hypot(pose[0] - x, pose[1] - y) <= 0.02)

            try:
                self._attendre(_confirme, self._teleport_timeout_s,
                               f"confirmation téléport objet → ({x:.3f}, {y:.3f}) m")
                return
            except TimeoutError:
                self._node.get_logger().warn(
                    f"téléport non confirmé (essai {essai}/{self._teleport_retries}) "
                    f"— croyance shim : {self._pose_objet}")
                # La cause la plus fréquente d'un refus : pince restée fermée.
                self._ouvrir_pince_confirmee()
        raise RuntimeError(
            f"objet non téléporté à ({x:.3f}, {y:.3f}) m après "
            f"{self._teleport_retries} tentatives, sim VIVANTE "
            "(gripper_shim mort ? service set_pose HS ?)")

    # ══════════════════════════════════════════════════════════════════════════
    # close
    # ══════════════════════════════════════════════════════════════════════════
    def close(self) -> None:
        """Arrête l'exécuteur/le nœud POSSÉDÉS (un nœud injecté reste à l'appelant)."""
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except Exception:  # noqa: BLE001
                pass
            self._executor = None
        if self._owns_node:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
