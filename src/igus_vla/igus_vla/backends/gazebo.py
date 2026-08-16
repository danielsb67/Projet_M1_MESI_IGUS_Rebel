"""
Backend Gazebo Ignition pour le pipeline VLA (SmolVLA).

Implémente RobotBackend pour la simulation :
    - Souscrit à /joint_states (QoS BEST_EFFORT = qos_profile_sensor_data).
    - Souscrit au topic image configurable (défaut /front_camera/image).
    - Publie les trajectoires sur /rebel_arm_controller/joint_trajectory.
    - Appelle le service /gripper/command (std_srvs/srv/SetBool) de manière
      asynchrone pour ne pas bloquer l'exécuteur ROS.

Ce backend n'appelle PAS rclpy.init() ni ne crée son propre nœud ROS.
Il utilise le nœud (rclpy.node.Node) passé au constructeur.

Référence : VLA_PLAN.md §7.1, CODEBASE_MAP.md §7.
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np

# Les imports ROS sont gardés au niveau module — py_compile ne les évalue pas
# à l'exécution ; si rclpy n'est pas installé, l'import du module échouera à
# l'exécution (normal en dehors d'un environnement ROS sourcé).
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_srvs.srv import SetBool

from igus_vla.backends.base import RobotBackend

# Noms canoniques des joints du bras igus ReBeL 6DOF (VLA_PLAN.md §2.2)
_JOINT_NAMES: list[str] = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
]


class GazeboBackend(RobotBackend):
    """Backend de simulation Gazebo Ignition.

    Utilise le nœud ROS passé en paramètre pour créer les souscriptions,
    le publisher de trajectoires et le client du service pince. Ne crée pas
    de nœud ROS propre et n'appelle pas rclpy.init().

    Parameters
    ----------
    node : rclpy.node.Node
        Nœud ROS actif fourni par le code appelant.
    image_topic : str
        Topic ROS de l'image frontale (défaut ``/front_camera/image``).
    joint_states_topic : str
        Topic /joint_states (défaut ``/joint_states``).
    arm_traj_topic : str
        Topic de publication des trajectoires articulaires
        (défaut ``/rebel_arm_controller/joint_trajectory``).
    gripper_service : str
        Nom du service pince SetBool
        (défaut ``/gripper/command``).
    stamp_mode : str
        ``"now"``  = ``header.stamp`` = horloge ROS du nœud au moment de la
        publication (le contrôleur date le début de la trajectoire dessus) ;
        ``"zero"`` = laisser le stamp à 0, ce qui signifie « démarrer à la
        réception ». Le mode ``"now"`` n'est correct QUE si le nœud tourne en
        temps sim (``use_sim_time:=true``) : un stamp calculé en temps mur alors
        que le contrôleur vit en temps sim serait interprété comme très ancien,
        et le contrôleur sauterait d'un coup au milieu de la trajectoire.
        ``"zero"`` est le repli immunisé contre tout décalage d'horloge.
    max_velocity : float
        Plafond (rad/s) du feedforward de vitesse écrit dans les points de
        trajectoire. Défaut = limite URDF du ReBeL (π/4 = 45 °/s), que les
        démonstrations de l'expert n'ont par construction jamais dépassée : sert
        de garde-fou si la politique prédit un saut de position, sans quoi la
        différence finie du 1er point (calculée depuis la mesure courante) peut
        demander une vitesse irréaliste. ``0`` ou négatif = pas de plafond.
    """

    def __init__(
        self,
        node: rclpy.node.Node,
        image_topic: str = "/front_camera/image",
        joint_states_topic: str = "/joint_states",
        arm_traj_topic: str = "/rebel_arm_controller/joint_trajectory",
        gripper_service: str = "/gripper/command",
        callback_group=None,
        image_topics: Optional[Dict[str, str]] = None,
        stamp_mode: str = "now",
        max_velocity: float = np.pi / 4.0,
    ) -> None:
        self._node = node
        self._stamp_mode = "zero" if str(stamp_mode).lower() == "zero" else "now"
        self._max_velocity = float(max_velocity)
        # Caméras : dict clé logique → topic ROS. Rétrocompatible : si image_topics
        # n'est pas fourni, on garde la seule caméra "front" (= image_topic). Le v2
        # ajoute "wrist" (eye-in-hand) → {"front": ..., "wrist": ...}. cf. VLA_V2_PLAN §4.
        self._image_topics: Dict[str, str] = (
            dict(image_topics) if image_topics else {"front": image_topic})
        # Groupe de callbacks dédié aux souscriptions capteurs (image + joints).
        # Indispensable avec un MultiThreadedExecutor : isole l'image d'un timer
        # lourd ou d'un flux /joint_states haute fréquence qui, dans le groupe
        # mutuellement exclusif par défaut, l'affament (image figée). cf. recorder.
        self._cb_group = callback_group

        # Cache des dernières valeurs reçues
        self._latest_joint_positions: Optional[np.ndarray] = None
        self._latest_images: Dict[str, Optional[np.ndarray]] = {
            k: None for k in self._image_topics}
        # Compteur de séquence PAR caméra, incrémenté par le callback image à
        # chaque frame reçue (0 = jamais reçue). Sert de marqueur « vu » pour le
        # dédoublonnage du recorder : le consommateur mémorise la séquence lue au
        # dernier tick consigné et saute le tick si elle n'a pas bougé (mesuré
        # v2.1 : 10,1 % de doublons front, 17,2 % wrist sans ce garde-fou).
        self._image_seq: Dict[str, int] = {k: 0 for k in self._image_topics}

        # État de la pince suivi en interne (pas de capteur disponible)
        self._gripper_closed: bool = False

        # Import cv_bridge protégé pour ne pas planter py_compile sans ROS
        try:
            from cv_bridge import CvBridge  # type: ignore
            self._bridge = CvBridge()
        except ImportError:
            self._bridge = None
            self._node.get_logger().warning(
                "cv_bridge non disponible — get_image() ne fonctionnera pas."
            )

        # Souscription /joint_states (BEST_EFFORT obligatoire — CODEBASE_MAP §7.2)
        self._js_sub = node.create_subscription(
            JointState,
            joint_states_topic,
            self._joint_states_callback,
            qos_profile_sensor_data,  # BEST_EFFORT
            callback_group=self._cb_group,
        )

        # Souscriptions image, une par caméra (BEST_EFFORT — cohérent caméra Gazebo)
        self._img_subs: Dict[str, object] = {}
        for key, topic in self._image_topics.items():
            self._img_subs[key] = node.create_subscription(
                Image,
                topic,
                self._make_image_callback(key),
                qos_profile_sensor_data,
                callback_group=self._cb_group,
            )

        # Publisher de trajectoires articulaires
        self._traj_pub = node.create_publisher(
            JointTrajectory,
            arm_traj_topic,
            10,
        )

        # Client de service pince
        self._gripper_client = node.create_client(SetBool, gripper_service)

        self._node.get_logger().info(
            f"GazeboBackend initialisé — caméras: {self._image_topics}, "
            f"joints: {joint_states_topic}, traj: {arm_traj_topic}, "
            f"gripper: {gripper_service}"
        )

    # ------------------------------------------------------------------
    # Callbacks de souscriptions
    # ------------------------------------------------------------------

    def _joint_states_callback(self, msg: JointState) -> None:
        """Reçoit /joint_states et réordonne les positions par nom de joint."""
        name_to_pos: Dict[str, float] = dict(zip(msg.name, msg.position))
        positions: list[float] = []
        all_found = True
        for jname in _JOINT_NAMES:
            if jname in name_to_pos:
                positions.append(name_to_pos[jname])
            else:
                # Joint manquant : utiliser 0.0 et logger un avertissement une fois
                positions.append(0.0)
                all_found = False
        if not all_found:
            self._node.get_logger().warning(
                f"Certains joints sont absents du message /joint_states. "
                f"Reçus : {list(msg.name)}. Attendus : {_JOINT_NAMES}. "
                f"Valeurs manquantes remplacées par 0.0.",
                throttle_duration_sec=5.0,
            )
        self._latest_joint_positions = np.array(positions, dtype=np.float64)

    def _make_image_callback(self, key: str):
        """Fabrique le callback image d'une caméra (capture la clé logique)."""
        def _cb(msg: Image) -> None:
            if self._bridge is None:
                return
            try:
                # cv_bridge retourne BGR → on stocke en RGB
                bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self._latest_images[key] = bgr[:, :, ::-1].copy()
                # Marqueur « vu » APRÈS l'image : un lecteur qui voit la séquence
                # bouger est certain que l'image correspondante est déjà en cache.
                self._image_seq[key] += 1
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().error(
                    f"Erreur conversion image '{key}' : {exc}",
                    throttle_duration_sec=2.0,
                )
        return _cb

    # ------------------------------------------------------------------
    # Implémentation des méthodes abstraites
    # ------------------------------------------------------------------

    def get_joint_state(self) -> np.ndarray:
        """Retourne les positions articulaires [joint1..joint6] en radians.

        Returns
        -------
        np.ndarray
            float64 shape (6,). Retourne des zéros si aucune mesure n'a encore
            été reçue (cas transitoire au démarrage).
        """
        if self._latest_joint_positions is None:
            return np.zeros(6, dtype=np.float64)
        return self._latest_joint_positions.copy()

    def get_gripper_state(self) -> float:
        """Retourne l'état de la pince suivi en interne.

        Returns
        -------
        float
            0.0 (ouverte) ou 1.0 (fermée).
        """
        return 1.0 if self._gripper_closed else 0.0

    def get_image(self, key: str = "front") -> np.ndarray:
        """Retourne la dernière image RGB (H×W×3, uint8) de la caméra ``key``.

        Parameters
        ----------
        key : str
            Clé logique de caméra (``"front"``, ``"wrist"``, …) parmi celles
            configurées au constructeur.

        Returns
        -------
        np.ndarray
            uint8 shape (480, 640, 3) RGB. Retourne un tableau noir si aucune
            image n'a encore été reçue.

        Raises
        ------
        KeyError
            Si ``key`` n'est pas une caméra configurée.
        """
        if key not in self._latest_images:
            raise KeyError(
                f"GazeboBackend : clé d'image inconnue '{key}'. "
                f"Caméras configurées : {list(self._image_topics)}."
            )
        img = self._latest_images[key]
        if img is None:
            # Retourner une image noire (évite un crash au démarrage)
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return img.copy()

    def get_image_seq(self, key: str) -> int:
        """Numéro de séquence de la dernière image reçue pour la caméra ``key``.

        Incrémenté par le callback image à chaque frame (0 = jamais reçue).
        Le recorder s'en sert pour DÉDOUBLONNER : il consigne la séquence
        consommée au dernier tick enregistré et saute le tick si elle n'a pas
        changé (image du cache identique à celle déjà écrite).

        Raises
        ------
        KeyError
            Si ``key`` n'est pas une caméra configurée.
        """
        if key not in self._image_seq:
            raise KeyError(
                f"GazeboBackend : clé d'image inconnue '{key}'. "
                f"Caméras configurées : {list(self._image_topics)}."
            )
        return self._image_seq[key]

    @staticmethod
    def _to_duration(seconds: float) -> Duration:
        """Convertit des secondes flottantes en builtin_interfaces/Duration."""
        seconds = max(0.0, float(seconds))
        secs = int(seconds)
        nsecs = int(round((seconds - secs) * 1e9))
        if nsecs >= 1_000_000_000:          # arrondi qui déborde (ex. 0.9999999996)
            secs += 1
            nsecs -= 1_000_000_000
        return Duration(sec=secs, nanosec=nsecs)

    def _new_trajectory_msg(self) -> JointTrajectory:
        """Fabrique un JointTrajectory horodaté selon ``stamp_mode``."""
        msg = JointTrajectory()
        msg.joint_names = _JOINT_NAMES
        if self._stamp_mode == "now":
            # Le contrôleur fait démarrer la trajectoire à cette date. La lire sur
            # l'horloge du nœud (donc l'horloge SIM si use_sim_time est posé) est
            # la seule façon d'être dans le même référentiel temporel que lui.
            msg.header.stamp = self._node.get_clock().now().to_msg()
        return msg

    def send_joint_targets(self, q: np.ndarray, duration: float) -> None:
        """Publie une consigne articulaire MONO-POINT.

        Réservée aux déplacements ponctuels (retour HOME, mode ``stream``). Pour
        rejouer un chunk de politique, utiliser ``send_joint_trajectory`` : un
        flux de messages mono-point fait battre le contrôleur entre préemption et
        expiration de trajectoire (oscillation), cf. ``base.py``.

        Aucune vitesse n'est renseignée : le contrôleur amène donc le bras au
        point puis l'y tient à vitesse nulle (comportement voulu pour un HOME).

        Parameters
        ----------
        q : np.ndarray
            Positions cibles float shape (6,), en radians, ordre [joint1..joint6].
        duration : float
            Durée totale de la trajectoire en secondes.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,):
            raise ValueError(
                f"send_joint_targets attend q.shape == (6,), reçu {q.shape}."
            )

        msg = self._new_trajectory_msg()

        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        point.time_from_start = self._to_duration(duration)

        msg.points = [point]
        self._traj_pub.publish(msg)

    def send_joint_trajectory(
        self,
        q_seq: np.ndarray,
        dt: float,
        with_velocities: bool = True,
    ) -> float:
        """Publie un chunk entier comme UNE trajectoire multi-points.

        Échéancier : ``q_seq[i]`` est datée à ``(i + 1) * dt``. C'est la
        sémantique exacte du dataset (``action[t] = state[t+1]`` à 1/dt Hz) : la
        i-ème action est la position que le bras doit avoir un pas plus tard.

        Vitesses (feedforward) : différences finies CENTRÉES sur la séquence
        augmentée de la position mesurée à l'instant 0, soit
        ``v[i] = (q[i+1] - q[i-1]) / (2·dt)`` avec ``q[-1]`` = mesure courante.
        Le contrôleur (``command_interfaces: [velocity]``, ``ff_velocity_scale``
        à 1.0) s'en sert directement ; sans elles il doit dériver l'interpolation
        lui-même et le suivi est plus mou.

        Le DERNIER point a une vitesse nulle, et ce n'est pas cosmétique : le
        contrôleur est configuré avec ``allow_nonzero_velocity_at_trajectory_end:
        false`` et REJETTE toute trajectoire dont le dernier point a une vitesse
        non nulle. Le bras marque donc un court arrêt en fin de chunk, pendant
        l'inférence suivante — c'est net et déterministe, contrairement au
        battement d'un flux mono-point.

        Parameters
        ----------
        q_seq : np.ndarray
            Positions cibles float shape (N, 6), radians, ordre [joint1..joint6].
        dt : float
            Pas de temps entre deux cibles (s), typiquement 1 / control_hz.
        with_velocities : bool
            ``False`` = ne publier que les positions (permet l'A/B sans rebuild).

        Returns
        -------
        float
            Durée totale de la trajectoire (s) = ``N * dt``.
        """
        q_seq = np.asarray(q_seq, dtype=np.float64)
        if q_seq.ndim != 2 or q_seq.shape[1] != 6:
            raise ValueError(
                f"send_joint_trajectory attend q_seq.shape == (N, 6), reçu {q_seq.shape}."
            )
        if q_seq.shape[0] == 0:
            return 0.0
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"send_joint_trajectory attend dt > 0, reçu {dt}.")

        n = q_seq.shape[0]
        vel: Optional[np.ndarray] = None
        if with_velocities and n >= 2:
            # Position à t=0 : la mesure courante (le bras part de là). Si aucune
            # mesure n'est encore arrivée, on prend la 1re cible → v[0] ≈ 0.
            q_start = (self._latest_joint_positions.copy()
                       if self._latest_joint_positions is not None
                       else q_seq[0].copy())
            ext = np.vstack([q_start[None, :], q_seq])      # (N+1, 6)
            vel = np.zeros_like(q_seq)
            vel[:-1] = (ext[2:] - ext[:-2]) / (2.0 * dt)    # différences centrées
            if self._max_velocity > 0.0:
                np.clip(vel, -self._max_velocity, self._max_velocity, out=vel)
            vel[-1] = 0.0                                   # exigé par le contrôleur

        msg = self._new_trajectory_msg()
        points = []
        for i in range(n):
            point = JointTrajectoryPoint()
            point.positions = q_seq[i].tolist()
            if vel is not None:
                point.velocities = vel[i].tolist()
            point.time_from_start = self._to_duration((i + 1) * dt)
            points.append(point)
        msg.points = points
        self._traj_pub.publish(msg)
        return float(n * dt)

    def set_gripper(self, closed: bool) -> None:
        """Ouvre ou ferme la pince via /gripper/command (SetBool).

        L'état interne est mis à jour immédiatement. L'appel au service est
        asynchrone (call_async) pour ne pas bloquer l'exécuteur ROS.

        Parameters
        ----------
        closed : bool
            ``True`` = fermer, ``False`` = ouvrir.
        """
        # Mise à jour immédiate de l'état interne
        self._gripper_closed = closed

        if not self._gripper_client.service_is_ready():
            self._node.get_logger().warning(
                "/gripper/command : service non disponible — commande ignorée.",
                throttle_duration_sec=2.0,
            )
            return

        req = SetBool.Request()
        req.data = closed
        future = self._gripper_client.call_async(req)

        # Callback de log non bloquant
        def _on_response(fut: rclpy.Future) -> None:  # type: ignore[name-defined]
            try:
                resp = fut.result()
                if not resp.success:
                    self._node.get_logger().warning(
                        f"/gripper/command répondu success=False : {resp.message}"
                    )
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().error(
                    f"/gripper/command erreur future : {exc}"
                )

        future.add_done_callback(_on_response)

    def is_ready(self) -> bool:
        """Vérifie que joints + TOUTES les caméras ont fourni au moins une mesure.

        Returns
        -------
        bool
            ``True`` si ``/joint_states`` ET chaque caméra configurée
            (front, et wrist en v2) ont publié au moins une fois.
        """
        return (
            self._latest_joint_positions is not None
            and all(v is not None for v in self._latest_images.values())
        )
