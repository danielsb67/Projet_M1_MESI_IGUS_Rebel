#!/usr/bin/env python3
"""
Pick & Place — Igus Rebel + Schunk EGP25
=========================================
Mode statique  (mode:=static, défaut) : un cycle pick→place avec positions fixes.
Mode réactif   (mode:=reactive)       : écoute /object_position_in_world et exécute
                                        un cycle à chaque détection du nœud perception.

Stratégie IK : TRAC-IK via /compute_ik (position-only, orientation libre).
Seeds enchaînés (solution N-1 = seed N) pour minimiser les sauts articulaires.
OMPL évité : segfault sur Humble — on utilise les joint goals MoveGroup.

Pré-requis :
  ros2 launch igus_rebel_moveit_config demo.launch.py \
      hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none camera:=none load_base:=false
  ros2 run mon_controleur securite
  ros2 run mon_controleur workspace_scene   (optionnel mais recommandé)
"""
import math
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK

# ══════════════════════════════════════════════════════════════════
#  VOS POINTS — modifier ici (en mètres, repère base du robot)
# ══════════════════════════════════════════════════════════════════

# Offsets de calibration caméra → base robot, mesurés expérimentalement.
# Appliqués aux coordonnées venant de /object_position_in_world en mode réactif.
# Convention : pos_robot = pos_camera + ERREUR_CALIBRATION_*
ERREUR_CALIBRATION_X = +0.04  # robot trop court en X : décaler la cible de +4 cm
ERREUR_CALIBRATION_Y = -0.05  # robot trop à +Y : décaler la cible de -5 cm
ERREUR_CALIBRATION_Z = -0.01  # offset URDF mesuré expérimentalement

PICK_X,  PICK_Y,  PICK_Z  = 0.4,  0.15,  (0.0 + ERREUR_CALIBRATION_Z)
# PLACE fixe : bac à 25 cm devant le robot (Y), centré (X=0), gripper relâche à 10 cm.
# Volontairement loin de la zone de pick pour ne jamais relâcher au même endroit
# que celui d'où on vient de saisir l'objet.
PLACE_X, PLACE_Y, PLACE_Z = 0.0,  0.25,  0.10

APPROACH_OFFSET = 0.1   # hauteur d'approche avant descente (m)
LIFT_OFFSET     = 0.1   # levée après saisie / dépôt (m)

# Orientation FIGÉE de la pince : verticale, pointant vers le sol.
# Le repère gripper_tip_link a son axe X aligné avec l'axe outil (voir
# schunk_egp25.urdf.xacro : offset (0.185,0,0) le long de X). Pour que cet
# axe X pointe en -Z monde, on applique une rotation de -π/2 autour de Y :
#   q = (0, sin(-π/4), 0, cos(-π/4)) ≈ (0, -0.7071, 0, 0.7071)
# Conséquence : joint5 est forcé dans la configuration "wrist vertical",
# son axe de rotation reste horizontal et le segment terminal pend
# parallèlement à Z monde. Pré-requis : position_only_ik=false dans
# kinematics.yaml (sinon TRAC-IK ignore cette orientation).
DOWNWARD_QUAT = (0.0, -0.70710678, 0.0, 0.70710678)  # (x, y, z, w)

# Hauteur de l'objet à saisir (m). La pince descend toujours à mi-hauteur
# de l'objet : la profondeur caméra est trop peu fiable pour la saisie.
HAUTEUR_OBJET = 0.036
PICK_Z_FIXE   = HAUTEUR_OBJET / 2.0   # 0.018 m — Z de saisie imposé

# ══════════════════════════════════════════════════════════════════
#  CONFIG ROBOT
# ══════════════════════════════════════════════════════════════════

BASE_FRAME  = "igus_rebel_base_link"
ARM_GROUP   = "rebel_arm"
EE_LINK     = "gripper_tip_link"
GRIPPER_SRV = "/gripper/command"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# HOME : pose de repos. L'effecteur DOIT rester hors de l'objet de collision
# 'profil_avant' publié par workspace_scene.py (X∈[0.51,0.66], Z∈[0,0.52] m).
# Valeurs en radians, converties depuis les angles en degrés
# [0, -28, 120, 0, 80, 0].
HOME_JOINTS = [0.0, -0.488692, 2.094395, 0.0, 1.396263, 0.0]

Z_MIN_M = 0.0 + ERREUR_CALIBRATION_Z  # plancher sécurité bout pince (m)

# ── Limites articulaires (validation post-IK) ─────────────────────
# Valeurs alignées sur le URDF igus_rebel.description.xacro
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "joint1": (-179.0, 179.0),
    "joint2": ( -80.0, 140.0),
    "joint3": ( -80.0, 140.0),  # asymétrique dans le URDF — 133.6° est valide
    "joint4": (-179.0, 179.0),
    "joint5": ( -90.0,  90.0),
    "joint6": (-179.0, 179.0),
}

# Seuil d'alerte pour les grands sauts articulaires entre deux étapes
JUMP_WARN_RAD = 2.5

VELOCITY_SCALE     = 0.25   # remplacé par paramètre ROS 'vel_scale'
ACCELERATION_SCALE = 0.15   # remplacé par paramètre ROS 'acc_scale'
GRIPPER_WAIT       = 1.5


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _normalize(angle: float) -> float:
    """Ramène un angle dans [-π, π]."""
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


# ══════════════════════════════════════════════════════════════════
class PickPlaceIA(Node):

    def __init__(self):
        super().__init__("pick_place_ia")

        # ── Paramètres ROS ────────────────────────────────────────
        self.declare_parameter("mode", "static")
        self._mode = self.get_parameter("mode").get_parameter_value().string_value

        self.declare_parameter("vel_scale", VELOCITY_SCALE)
        self.declare_parameter("acc_scale", ACCELERATION_SCALE)

        self.declare_parameter("pick_x",  PICK_X)
        self.declare_parameter("pick_y",  PICK_Y)
        self.declare_parameter("pick_z",  PICK_Z)
        self.declare_parameter("place_x", PLACE_X)
        self.declare_parameter("place_y", PLACE_Y)
        self.declare_parameter("place_z", PLACE_Z)

        # ── Callback group réentrant ──────────────────────────────
        self._cbg = ReentrantCallbackGroup()

        # ── Clients / Action — tous avec callback_group ───────────
        self._mg          = ActionClient(self, MoveGroup, "/move_action",
                                         callback_group=self._cbg)
        self._ik_cli      = self.create_client(GetPositionIK, "/compute_ik",
                                               callback_group=self._cbg)
        self._gripper_cli = self.create_client(SetBool, GRIPPER_SRV,
                                               callback_group=self._cbg)

        # ── État interne ──────────────────────────────────────────
        self._securite_ok    = True
        self._current_joints = list(HOME_JOINTS)
        self._busy           = False

        # ── Subscriptions ─────────────────────────────────────────
        self.create_subscription(Bool,       "/securite_mouvement",
                                 self._securite_cb,    10,
                                 callback_group=self._cbg)
        self.create_subscription(JointState, "/joint_states",
                                 self._joint_state_cb, 10,
                                 callback_group=self._cbg)

        # ── Mode réactif : subscription perception ────────────────
        if self._mode == "reactive":
            self.create_subscription(PointStamped, "/object_position_in_world",
                                     self._object_cb, 10,
                                     callback_group=self._cbg)
            self.get_logger().info(
                "Mode REACTIF activé — en attente de détections sur "
                "/object_position_in_world"
            )
        else:
            self.get_logger().info(
                "Mode STATIQUE activé — positions fixes "
                f"PICK=({PICK_X}, {PICK_Y}, {PICK_Z}), "
                f"PLACE=({PLACE_X}, {PLACE_Y}, {PLACE_Z})"
            )

    # ── Callbacks sécurité / état ─────────────────────────────────

    def _securite_cb(self, msg: Bool):
        self._securite_ok = msg.data

    def _joint_state_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        self._current_joints = [name_to_pos.get(n, 0.0) for n in JOINT_NAMES]

    # ── Callback mode réactif ─────────────────────────────────────

    def _object_cb(self, msg: PointStamped):
        if self._busy:
            return

        # Correction du décalage de calibration caméra → robot.
        # Sans ces offsets, le robot rate la cible de ~4 cm en X / ~5 cm en Y.
        px = msg.point.x + ERREUR_CALIBRATION_X
        py = msg.point.y + ERREUR_CALIBRATION_Y
        # Z imposé : la pince descend toujours à mi-hauteur de l'objet
        # (18 mm) au lieu de suivre la profondeur caméra, peu fiable.
        pz = PICK_Z_FIXE

        # ── Vérification portée robot avant de lancer le cycle ────────
        MAX_REACH_XY = 0.62   # portée max Igus Rebel en XY (m)
        MAX_Z        = 0.55   # hauteur max atteignable (m)
        xy_dist = math.sqrt(px ** 2 + py ** 2)

        if xy_dist > MAX_REACH_XY:
            self.get_logger().warn(
                f"Objet hors portée XY : {xy_dist:.3f} m (max {MAX_REACH_XY} m) — "
                "vérifier la calibration caméra→robot (yaw=π dans static_transform_publisher)"
            )
            return
        if pz > MAX_Z:
            self.get_logger().warn(
                f"Objet trop haut : Z={pz:.3f} m (max {MAX_Z} m) — cycle ignoré"
            )
            return
        if pz < Z_MIN_M:
            self.get_logger().warn(
                f"Objet sous le plancher de sécurité : Z={pz:.3f} m "
                f"(min {Z_MIN_M} m) — cycle ignoré"
            )
            return

        self._busy = True

        place_x = self.get_parameter("place_x").get_parameter_value().double_value
        place_y = self.get_parameter("place_y").get_parameter_value().double_value
        place_z = self.get_parameter("place_z").get_parameter_value().double_value

        self.get_logger().info(
            f"Objet détecté → caméra ({msg.point.x:.3f}, {msg.point.y:.3f}) m "
            f"→ PICK corrigé ({px:.3f}, {py:.3f}, {pz:.3f}) m "
            f"[offsets cal: X{ERREUR_CALIBRATION_X:+.3f} Y{ERREUR_CALIBRATION_Y:+.3f}]"
        )

        threading.Thread(
            target=self._run_cycle,
            args=(px, py, pz, place_x, place_y, place_z),
            daemon=True,
        ).start()

    # ── Spin helper ───────────────────────────────────────────────

    def _wait(self, future, timeout=50.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

    # ── IK : XYZ → angles articulaires via TRAC-IK ───────────────

    def _ik_to_joints(self, x, y, z, label='', seed=None, _no_retry=False):
        """Résout l'IK position-only via TRAC-IK (solve_type=Distance).

        seed : point de départ articulaire ; si None, utilise l'état courant.
               TRAC-IK retourne la solution la plus proche du seed.
        _no_retry : interne — évite la récursion infinie sur le fallback HOME.
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name       = ARM_GROUP
        req.ik_request.ik_link_name     = EE_LINK
        req.ik_request.avoid_collisions = False  # DIAG: désactivé temporairement
        req.ik_request.timeout.sec      = 5

        # Orientation figée verticale (pince vers le bas) — voir DOWNWARD_QUAT.
        # Nécessite position_only_ik=false dans kinematics.yaml, sinon TRAC-IK
        # ignore l'orientation et joint5 sort dans une pose inclinée.
        ps = PoseStamped()
        ps.header.frame_id    = BASE_FRAME
        ps.pose.position.x    = float(x)
        ps.pose.position.y    = float(y)
        ps.pose.position.z    = float(z)
        ps.pose.orientation.x = DOWNWARD_QUAT[0]
        ps.pose.orientation.y = DOWNWARD_QUAT[1]
        ps.pose.orientation.z = DOWNWARD_QUAT[2]
        ps.pose.orientation.w = DOWNWARD_QUAT[3]
        req.ik_request.pose_stamped = ps

        seed_positions = seed if seed is not None else self._current_joints
        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(seed_positions)
        req.ik_request.robot_state = rs

        res = self._wait(self._ik_cli.call_async(req), timeout=8.0)
        if res is None or res.error_code.val != 1:
            code = res.error_code.val if res else 'timeout'
            self.get_logger().error(
                f'IK échoué (code {code}) pour {label} ({x:.3f}, {y:.3f}, {z:.3f})'
            )
            return None

        js = res.solution.joint_state
        positions = []
        for name in JOINT_NAMES:
            if name in js.name:
                positions.append(js.position[js.name.index(name)])
            else:
                self.get_logger().error(f'Joint "{name}" absent de la solution IK')
                return None

        positions = [_normalize(p) for p in positions]

        # Alerte saut articulaire entre seed et solution
        seed_ref = seed if seed is not None else self._current_joints
        jump = sum(abs(_normalize(a - b)) for a, b in zip(positions, seed_ref))
        if jump > JUMP_WARN_RAD:
            self.get_logger().warn(
                f'  ⚠ {label} : saut articulaire {jump:.2f} rad depuis le seed'
            )

        # Validation des limites
        violations = []
        for name, val in zip(JOINT_NAMES, positions):
            lo, hi = JOINT_LIMITS_DEG[name]
            if not (_deg2rad(lo) <= val <= _deg2rad(hi)):
                violations.append(f'{name}={math.degrees(val):.1f}° hors [{lo:.0f}°,{hi:.0f}°]')
        if violations:
            self.get_logger().warn(f'  ⚠ {label} hors limites : {", ".join(violations)}')
            if not _no_retry:
                self.get_logger().info(f'  → retry IK depuis HOME pour {label}')
                alt = self._ik_to_joints(x, y, z, label, seed=list(HOME_JOINTS), _no_retry=True)
                if alt is not None:
                    return alt
            self.get_logger().error(
                f'  ✗ {label} : aucune solution dans les limites articulaires.\n'
                f'    Piste : ajuster les coordonnées ou élargir JOINT_LIMITS_DEG.'
            )
            return None

        self.get_logger().info(
            f'  IK {label} → {[f"{math.degrees(p):.1f}°" for p in positions]}'
        )
        return positions

    # ── Vérification "déjà en position" ──────────────────────────

    def _is_at_goal(self, positions, tol: float = 0.05) -> bool:
        return all(
            abs(_normalize(c) - _normalize(t)) < tol
            for c, t in zip(self._current_joints, positions)
        )

    # ── Goal articulaire ──────────────────────────────────────────

    def _joint_goal(self, positions) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 3
        req.allowed_planning_time           = 10.0
        req.max_velocity_scaling_factor     = float(
            self.get_parameter("vel_scale").get_parameter_value().double_value)
        req.max_acceleration_scaling_factor = float(
            self.get_parameter("acc_scale").get_parameter_value().double_value)
        req.pipeline_id = "pilz_industrial_motion_planner"
        req.planner_id  = "PTP"

        c = Constraints()
        for name, pos in zip(JOINT_NAMES, positions):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.05  # 0.01 → 0.05 rad : évite "goal already reached"
            jc.tolerance_below = 0.05  # sur Pilz PTP quand le robot est très proche
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]

        goal.planning_options.plan_only = False
        return goal

    def _joint_goal_stomp(self, constraints: Constraints) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 3
        req.allowed_planning_time           = 30.0
        req.max_velocity_scaling_factor     = float(
            self.get_parameter("vel_scale").get_parameter_value().double_value)
        req.max_acceleration_scaling_factor = float(
            self.get_parameter("acc_scale").get_parameter_value().double_value)
        req.pipeline_id = "stomp"
        req.planner_id  = "STOMP"
        req.goal_constraints                = [constraints]
        goal.planning_options.plan_only     = False
        return goal

    # ── Validation Z ──────────────────────────────────────────────

    def _z_safe(self, z: float, label: str) -> bool:
        if z < Z_MIN_M:
            self.get_logger().error(
                f'  SÉCURITÉ : cible Z={z:.3f} m < {Z_MIN_M} m — "{label}" refusé !'
            )
            return False
        return True

    # ── Envoi goal MoveGroup ──────────────────────────────────────

    def _send(self, goal, label: str,
              _ompl_fallback: bool = True,
              _retry: bool = True) -> bool:
        if not self._securite_ok:
            self.get_logger().error(f'  SÉCURITÉ active — "{label}" bloqué.')
            return False

        # Skip si déjà en position (évite CONTROL_FAILED -4 sur trajectoire 1-point)
        targets = [jc.position for jc in goal.request.goal_constraints[0].joint_constraints]
        if self._is_at_goal(targets):
            self.get_logger().info(f"  ✓ {label} (déjà en position)")
            return True

        self.get_logger().info(f"  → {label}")
        gh = self._wait(self._mg.send_goal_async(goal), timeout=12.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ Goal rejeté : {label}")
            if _retry:
                time.sleep(1.0)
                return self._send(goal, label, _ompl_fallback, _retry=False)
            return False

        result_future = gh.get_result_async()
        deadline = time.time() + 60.0
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.05)
            if not self._securite_ok:
                self.get_logger().error(
                    f'  ARRÊT D\'URGENCE pendant "{label}" — annulation goal.'
                )
                gh.cancel_goal_async()
                return False

        if not result_future.done():
            self.get_logger().warn(f"  ✗ Timeout : {label} — annulation")
            gh.cancel_goal_async()
            time.sleep(0.5)
            return False

        code = result_future.result().result.error_code.val
        if code == 1:
            self.get_logger().info(f"  ✓ {label}")
            return True

        # CONTROL_FAILED (-4) : contrôleur encore occupé → retry
        if code == MoveItErrorCodes.CONTROL_FAILED and _retry:
            self.get_logger().warn(f"  CONTROL_FAILED pour {label} — retry dans 1 s")
            time.sleep(1.0)
            return self._send(goal, label, _ompl_fallback, _retry=False)

        # PLANNING_FAILED (-1) ou INVALID_MOTION_PLAN (-2) → fallback STOMP
        if code in (MoveItErrorCodes.PLANNING_FAILED,
                    MoveItErrorCodes.INVALID_MOTION_PLAN) and _ompl_fallback:
            self.get_logger().warn(
                f"  Pilz PTP échec ({code}) pour {label} — fallback STOMP"
            )
            stomp_goal = self._joint_goal_stomp(goal.request.goal_constraints[0])
            return self._send(stomp_goal, label + "_STOMP", _ompl_fallback=False)

        self.get_logger().warn(f"  ✗ {label} — code MoveIt {code}")
        return False

    # ── Pince ─────────────────────────────────────────────────────

    def _gripper(self, close: bool):
        self.get_logger().info(f"  Pince : {'FERMER' if close else 'OUVRIR'}")
        if not self._gripper_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("  Service pince indisponible.")
            return
        req      = SetBool.Request()
        req.data = close
        self._wait(self._gripper_cli.call_async(req), timeout=4.0)
        time.sleep(GRIPPER_WAIT)

    # ── Attente serveurs (séparée de run) ─────────────────────────

    def wait_for_servers(self) -> bool:
        log = self.get_logger()

        # Block reactive callbacks during the entire init phase so no detection
        # fires while HOME_INIT is executing (race condition → CONTROL_FAILED -4).
        self._busy = True

        log.info("Attente du serveur /move_group...")
        if not self._mg.wait_for_server(timeout_sec=15.0):
            log.error("/move_group non disponible — lancez demo.launch.py d'abord.")
            self._busy = False
            return False

        log.info("Attente du service /compute_ik...")
        if not self._ik_cli.wait_for_service(timeout_sec=10.0):
            log.error("/compute_ik non disponible.")
            self._busy = False
            return False

        log.info("Synchronisation état sécurité...")
        for _ in range(30):
            time.sleep(0.05)
        if not self._securite_ok:
            log.error(
                "Le nœud sécurité bloque le mouvement au démarrage. "
                "Vérifiez que le robot est en position sûre (Z ≥ 0.007 m) "
                "et relancez — ou tapez 'go' dans le terminal du nœud sécurité."
            )
            self._busy = False
            return False

        log.info("Envoi en position HOME au démarrage...")
        if not self._send(self._joint_goal(HOME_JOINTS), "HOME_INIT"):
            log.warn(
                "HOME initial échoué — la cible HOME_JOINTS est probablement en "
                "collision avec un objet de la planning scene (ex. 'profil_avant' "
                "publié par workspace_scene). Diagnostic : "
                "ros2 service call /check_state_validity moveit_msgs/srv/GetStateValidity "
                "avec HOME_JOINTS — le champ 'contacts' indique le lien fautif."
            )

        # Laisser le contrôleur de trajectoire se stabiliser avant d'accepter
        # un nouveau goal (évite CONTROL_FAILED -4 si une détection arrive
        # immédiatement après HOME_INIT).
        time.sleep(1.0)
        self._busy = False
        return True

    # ── Pré-calcul IK pour toutes les étapes ──────────────────────

    def _compute_all_ik(self, pick_x, pick_y, pick_z,
                        place_x, place_y, place_z) -> 'dict | None':
        """Calcule l'IK pour les 6 étapes du cycle.

        Seeds enchaînés (solution N-1 = seed N) pour minimiser les sauts.
        Retourne un dict {label: joints} ou None si un IK échoue.
        """
        log = self.get_logger()
        log.info("Calcul IK de toutes les étapes (TRAC-IK, orientation libre)...")

        sequence = [
            ("APPROCHE_PICK",  pick_x,  pick_y,  pick_z  + APPROACH_OFFSET),
            ("PICK",           pick_x,  pick_y,  pick_z),
            ("LIFT_PICK",      pick_x,  pick_y,  pick_z  + LIFT_OFFSET),
            ("APPROCHE_PLACE", place_x, place_y, place_z + APPROACH_OFFSET),
            ("PLACE",          place_x, place_y, place_z),
            ("LIFT_PLACE",     place_x, place_y, place_z + LIFT_OFFSET),
        ]

        joints = {}
        seed = list(HOME_JOINTS)
        for label, x, y, z in sequence:
            j = self._ik_to_joints(x, y, z, label, seed=seed)
            if j is None:
                log.error(f"IK impossible pour {label} — vérifiez les coordonnées.")
                return None
            joints[label] = j
            # Pour LIFT_*, utiliser le seed de l'APPROCHE correspondante
            # (même hauteur ≈ que l'APPROACH, évite le saut depuis la position basse)
            if label == "PICK":
                seed = joints["APPROCHE_PICK"]
            elif label == "PLACE":
                seed = joints["APPROCHE_PLACE"]
            else:
                seed = j

        log.info("IK calculée — trajectoires optimisées.")
        return joints

    # ── Cycle pick → place ────────────────────────────────────────

    def _run_cycle(self, pick_x, pick_y, pick_z, place_x, place_y, place_z):
        log = self.get_logger()
        _cycle_ok = False
        try:
            # Validation Z de toutes les étapes avant de commencer
            for label, z in [
                ("PICK",           pick_z),
                ("APPROCHE_PICK",  pick_z  + APPROACH_OFFSET),
                ("LIFT_PICK",      pick_z  + LIFT_OFFSET),
                ("APPROCHE_PLACE", place_z + APPROACH_OFFSET),
                ("PLACE",          place_z),
                ("LIFT_PLACE",     place_z + LIFT_OFFSET),
            ]:
                if not self._z_safe(z, label):
                    log.error("Corrigez les coordonnées avant de relancer.")
                    return

            # Pré-calcul IK
            joints = self._compute_all_ik(pick_x, pick_y, pick_z,
                                           place_x, place_y, place_z)
            if joints is None:
                return

            log.info("=" * 60)
            log.info(f"  PICK  A = ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f}) m")
            log.info(f"  PLACE B = ({place_x:.3f}, {place_y:.3f}, {place_z:.3f}) m")
            log.info(f"  Plancher sécurité Z >= {Z_MIN_M:.3f} m")
            log.info("=" * 60)

            log.info("[1] Aller à HOME")
            if not self._send(self._joint_goal(HOME_JOINTS), "HOME"):
                log.error("HOME échoué — arrêt."); return

            log.info("[2] Ouverture pince")
            self._gripper(close=False)

            log.info(f"[3] Approche Pick (A + {APPROACH_OFFSET*100:.0f} cm)")
            if not self._send(self._joint_goal(joints["APPROCHE_PICK"]), "APPROCHE_PICK"):
                log.error("Approche Pick échouée — arrêt."); return

            log.info("[4] Descente sur A (PICK)")
            if not self._send(self._joint_goal(joints["PICK"]), "PICK"):
                log.error("Pick échoué — arrêt."); return

            log.info("[5] Saisie de l'objet")
            self._gripper(close=True)

            log.info(f"[6] Levée ({LIFT_OFFSET*100:.0f} cm)")
            if not self._send(self._joint_goal(joints["LIFT_PICK"]), "LIFT_PICK"):
                log.error("Levée Pick échouée — arrêt."); return

            log.info(f"[7] Approche Place (B + {APPROACH_OFFSET*100:.0f} cm)")
            if not self._send(self._joint_goal(joints["APPROCHE_PLACE"]), "APPROCHE_PLACE"):
                log.error("Approche Place échouée — arrêt."); return

            log.info("[8] Descente sur B (PLACE)")
            if not self._send(self._joint_goal(joints["PLACE"]), "PLACE"):
                log.error("Place échouée — arrêt."); return

            log.info("[9] Lâcher de l'objet")
            self._gripper(close=False)

            log.info("[10] Remontée et retour HOME")
            self._send(self._joint_goal(joints["LIFT_PLACE"]), "LIFT_PLACE")
            self._send(self._joint_goal(HOME_JOINTS), "HOME")

            log.info("=" * 60)
            log.info("  CYCLE TERMINÉ AVEC SUCCÈS")
            log.info("=" * 60)
            _cycle_ok = True

        finally:
            # Retour HOME si le cycle a été interrompu (erreur, timeout, sécurité)
            # et que la sécurité permet le mouvement
            if not _cycle_ok and self._securite_ok:
                log.info("⟲  Cycle interrompu — retour en position HOME...")
                self._send(self._joint_goal(HOME_JOINTS), "HOME_RECOVERY")
            self._busy = False

    # ── Point d'entrée mode statique (et test_repetabilite.py) ───

    def run(self):
        """Lance un cycle unique en mode statique.

        Appelée directement par main() en mode statique, et par
        PickPlaceTest.run_cycle() dans test_repetabilite.py.
        """
        pick_x  = self.get_parameter("pick_x").get_parameter_value().double_value
        pick_y  = self.get_parameter("pick_y").get_parameter_value().double_value
        pick_z  = self.get_parameter("pick_z").get_parameter_value().double_value
        place_x = self.get_parameter("place_x").get_parameter_value().double_value
        place_y = self.get_parameter("place_y").get_parameter_value().double_value
        place_z = self.get_parameter("place_z").get_parameter_value().double_value

        self._busy = True
        self._run_cycle(pick_x, pick_y, pick_z, place_x, place_y, place_z)


# ══════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceIA()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # L'executor tourne dans un thread séparé pour que _wait() puisse utiliser time.sleep()
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not node.wait_for_servers():
            return

        if node._mode == "static":
            node.run()
        else:
            # Mode réactif : _object_cb gère tout. On attend indéfiniment.
            node.get_logger().info("Mode réactif actif — en attente de détections...")
            spin_thread.join()

    except KeyboardInterrupt:
        node.get_logger().info("Arrêt (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
