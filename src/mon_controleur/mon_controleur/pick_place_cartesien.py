#!/usr/bin/env python3
"""
Pick & Place Cartésien — Igus Rebel + Schunk EGP25
===================================================
Segments critiques en ligne droite (espace cartésien via compute_cartesian_path) :
  approche → pick    (descente verticale)
  pick     → lift    (montée verticale)
  approche → place   (descente verticale)
  place    → lift    (montée verticale)

Repositionnements (HOME→approche, lift→approche_place, lift→HOME) : joint space.

Usage :
  ros2 run mon_controleur pick_place_cartesien --ros-args \
      -p use_sim_time:=true -p vel_scale:=0.8 -p acc_scale:=0.4
"""
import math, time, threading, copy

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints, JointConstraint, RobotState,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetCartesianPath

# ══════════════════════════════════════════════════════════════════
#  COORDONNÉES — modifier ici (mètres, repère base robot)
# ══════════════════════════════════════════════════════════════════

PICK_X,  PICK_Y,  PICK_Z  =  0.38,  0.15, -0.01
PLACE_X, PLACE_Y, PLACE_Z =  -0.38, -0.15, -0.01

APPROACH_OFFSET = 0.10   # hauteur approche au-dessus du point (m)
LIFT_OFFSET     = 0.10   # hauteur de levée après saisie/dépôt (m)

# Résolution du chemin cartésien : distance max entre deux points consécutifs
CARTESIAN_MAX_STEP    = 0.005   # 5 mm — trajectoire lisse
CARTESIAN_JUMP_THRESH = 0.0     # désactivé (0 = pas de limite de saut articulaire)
CARTESIAN_MIN_FRAC    = 0.95    # fraction minimale du chemin calculée (95 %)

# ══════════════════════════════════════════════════════════════════
#  CONFIG ROBOT
# ══════════════════════════════════════════════════════════════════

BASE_FRAME  = "igus_rebel_base_link"
ARM_GROUP   = "rebel_arm"
EE_LINK     = "gripper_tip_link"
GRIPPER_SRV = "/gripper/command"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_JOINTS = [0.15, 0.4, 0.7, 0.05, 1.0, 1.1]

Z_MIN_M            = -0.01
VELOCITY_SCALE     = 0.25
ACCELERATION_SCALE = 0.15
GRIPPER_WAIT       = 1.5

JOINT_LIMITS_DEG = {
    "joint1": (-179.0, 179.0),
    "joint2": ( -80.0, 140.0),
    "joint3": ( -80.0, 140.0),
    "joint4": (-179.0, 179.0),
    "joint5": ( -90.0,  90.0),
    "joint6": (-179.0, 179.0),
}


def _deg2rad(d): return d * math.pi / 180.0
def _normalize(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ══════════════════════════════════════════════════════════════════
class PickPlaceCartesien(Node):

    def __init__(self):
        super().__init__("pick_place_cartesien")

        self.declare_parameter("mode",      "static")
        self.declare_parameter("pick_x",    PICK_X)
        self.declare_parameter("pick_y",    PICK_Y)
        self.declare_parameter("pick_z",    PICK_Z)
        self.declare_parameter("place_x",   PLACE_X)
        self.declare_parameter("place_y",   PLACE_Y)
        self.declare_parameter("place_z",   PLACE_Z)
        self.declare_parameter("vel_scale", VELOCITY_SCALE)
        self.declare_parameter("acc_scale", ACCELERATION_SCALE)

        self._cbg = ReentrantCallbackGroup()

        # ── Actions ──────────────────────────────────────────────
        self._mg   = ActionClient(self, MoveGroup,         "/move_action",         callback_group=self._cbg)
        self._exec = ActionClient(self, ExecuteTrajectory, "/execute_trajectory",  callback_group=self._cbg)

        # ── Services ─────────────────────────────────────────────
        self._ik_cli      = self.create_client(GetPositionIK,   "/compute_ik",            callback_group=self._cbg)
        self._fk_cli      = self.create_client(GetPositionFK,   "/compute_fk",            callback_group=self._cbg)
        self._cart_cli    = self.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=self._cbg)
        self._gripper_cli = self.create_client(SetBool,         GRIPPER_SRV,              callback_group=self._cbg)

        # ── État interne ──────────────────────────────────────────
        self._securite_ok      = True
        self._current_joints   = list(HOME_JOINTS)
        self._busy             = False
        self._cycle_orientation = None  # orientation figée en début de cycle

        self.create_subscription(Bool,       "/securite_mouvement",  self._securite_cb,    10, callback_group=self._cbg)
        self.create_subscription(JointState, "/joint_states",         self._joint_state_cb, 10, callback_group=self._cbg)

        self.get_logger().info("PickPlaceCartesien démarré — mouvements pick/place en ligne droite")

    # ── Callbacks ─────────────────────────────────────────────────

    def _securite_cb(self, msg: Bool):
        self._securite_ok = msg.data

    def _joint_state_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        self._current_joints = [name_to_pos.get(n, 0.0) for n in JOINT_NAMES]

    # ── Utilitaires ───────────────────────────────────────────────

    def _wait(self, future, timeout=50.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

    def _vel(self):
        return float(self.get_parameter("vel_scale").get_parameter_value().double_value)

    def _acc(self):
        return float(self.get_parameter("acc_scale").get_parameter_value().double_value)

    def _z_safe(self, z, label):
        if z < Z_MIN_M:
            self.get_logger().error(f"SÉCURITÉ : {label} Z={z:.3f} < {Z_MIN_M} m — refusé")
            return False
        return True

    # ── IK : XYZ → angles articulaires ───────────────────────────

    def _ik(self, x, y, z, label="", seed=None):
        req = GetPositionIK.Request()
        req.ik_request.group_name       = ARM_GROUP
        req.ik_request.ik_link_name     = EE_LINK
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec      = 5

        ps = PoseStamped()
        ps.header.frame_id    = BASE_FRAME
        ps.pose.position.x    = float(x)
        ps.pose.position.y    = float(y)
        ps.pose.position.z    = float(z)
        ps.pose.orientation.w = 1.0
        req.ik_request.pose_stamped = ps

        seed_pos = seed if seed is not None else self._current_joints
        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(seed_pos)
        req.ik_request.robot_state = rs

        res = self._wait(self._ik_cli.call_async(req), timeout=8.0)
        if res is None or res.error_code.val != 1:
            self.get_logger().error(f"IK échoué pour {label} ({x:.3f},{y:.3f},{z:.3f})")
            return None

        js  = res.solution.joint_state
        pos = [js.position[js.name.index(n)] for n in JOINT_NAMES if n in js.name]
        if len(pos) != len(JOINT_NAMES):
            return None

        pos = [_normalize(p) for p in pos]

        violations = [
            f"{n}={math.degrees(v):.1f}°"
            for n, v in zip(JOINT_NAMES, pos)
            if not (_deg2rad(JOINT_LIMITS_DEG[n][0]) <= v <= _deg2rad(JOINT_LIMITS_DEG[n][1]))
        ]
        if violations:
            self.get_logger().warn(f"IK {label} hors limites : {violations}")
            alt = self._ik(x, y, z, label + "_retry", seed=list(HOME_JOINTS))
            if alt:
                return alt
            return None

        self.get_logger().info(
            f"IK {label} → {[f'{math.degrees(p):.1f}°' for p in pos]}"
        )
        return pos

    # ── FK : angles → pose 6D du gripper ─────────────────────────

    def _fk(self, joints=None) -> Pose | None:
        """Retourne la pose 6D courante de l'EE (FK)."""
        req = GetPositionFK.Request()
        req.header.frame_id   = BASE_FRAME
        req.fk_link_names     = [EE_LINK]
        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(joints if joints is not None else self._current_joints)
        req.robot_state = rs

        res = self._wait(self._fk_cli.call_async(req), timeout=5.0)
        if res is None or res.error_code.val != 1 or not res.pose_stamped:
            self.get_logger().warn("FK indisponible")
            return None
        return res.pose_stamped[0].pose

    # ── Goal articulaire (joint space) ────────────────────────────

    def _joint_goal(self, positions) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 3
        req.allowed_planning_time           = 10.0
        req.max_velocity_scaling_factor     = self._vel()
        req.max_acceleration_scaling_factor = self._acc()
        # Pilz PTP : planification déterministe en <1s (vs STOMP qui peut timeout)
        req.pipeline_id = "pilz_industrial_motion_planner"
        req.planner_id  = "PTP"

        c = Constraints()
        for name, pos in zip(JOINT_NAMES, positions):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints           = [c]
        goal.planning_options.plan_only = False
        return goal

    # ── Vérification "déjà en position" ──────────────────────────

    def _is_at_goal(self, positions, tol: float = 0.05) -> bool:
        """Vrai si toutes les articulations sont à moins de tol rad de la cible."""
        return all(
            abs(_normalize(c) - _normalize(t)) < tol
            for c, t in zip(self._current_joints, positions)
        )

    # ── Envoi goal MoveGroup (joint space) ───────────────────────

    def _send_joint(self, goal, label: str,
                    _ompl_fallback: bool = True,
                    _retry: bool = True) -> bool:
        if not self._securite_ok:
            self.get_logger().error(f'SÉCURITÉ active — "{label}" bloqué.')
            return False

        # Si déjà en position → skip (évite CONTROL_FAILED sur trajectoire à 1 point)
        targets = [jc.position for jc in goal.request.goal_constraints[0].joint_constraints]
        if self._is_at_goal(targets):
            self.get_logger().info(f"  ✓ {label} (déjà en position)")
            return True

        self.get_logger().info(f"  → [joint] {label}")
        gh = self._wait(self._mg.send_goal_async(goal), timeout=5.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ Goal rejeté : {label}")
            if _retry:
                self.get_logger().info(f"  ↺ Retry dans 1 s : {label}")
                time.sleep(1.0)
                return self._send_joint(goal, label, _ompl_fallback, _retry=False)
            return False

        result_future = gh.get_result_async()
        deadline = time.time() + 60.0
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.05)
            if not self._securite_ok:
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

        # CONTROL_FAILED (-4) : contrôleur encore occupé → attendre et retenter
        if code == MoveItErrorCodes.CONTROL_FAILED and _retry:
            self.get_logger().warn(
                f"  CONTROL_FAILED pour {label} — attente 1 s puis retry"
            )
            time.sleep(1.0)
            return self._send_joint(goal, label, _ompl_fallback, _retry=False)

        # PLANNING_FAILED (-1) ou INVALID_MOTION_PLAN (-2) : Pilz PTP en collision
        # (-2 = plan calculé mais invalide, ex. contact avec un objet de la scène)
        # → fallback STOMP qui optimise la trajectoire autour des obstacles
        if code in (MoveItErrorCodes.PLANNING_FAILED,
                    MoveItErrorCodes.INVALID_MOTION_PLAN) and _ompl_fallback:
            self.get_logger().warn(
                f"  Pilz PTP échec ({code}) pour {label} — fallback STOMP"
            )
            stomp_goal = self._joint_goal_stomp(goal.request.goal_constraints[0])
            return self._send_joint(stomp_goal, label + "_STOMP", _ompl_fallback=False)

        if code == MoveItErrorCodes.CONTROL_FAILED:
            # Pilz a généré une trajectoire dégénérée (1 point = "déjà en position")
            # Le contrôleur répond UNKNOWN → CONTROL_FAILED, mais le robot est bien en place
            if self._is_at_goal(targets, tol=0.10):
                self.get_logger().info(f"  ✓ {label} (position validée malgré CONTROL_FAILED)")
                return True

        self.get_logger().warn(f"  ✗ {label} — code MoveIt {code}")
        return False

    def _joint_goal_stomp(self, constraints: Constraints) -> MoveGroup.Goal:
        """Goal identique planifié avec STOMP (évitement de collisions par optimisation)."""
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 3
        req.allowed_planning_time           = 30.0
        req.max_velocity_scaling_factor     = self._vel()
        req.max_acceleration_scaling_factor = self._acc()
        req.pipeline_id = "stomp"
        req.planner_id  = "STOMP"
        req.goal_constraints                = [constraints]
        goal.planning_options.plan_only     = False
        return goal

    # ── Mouvement cartésien en ligne droite ───────────────────────

    def _send_cartesian(self, target_x, target_y, target_z, label: str) -> bool:
        """Déplace le gripper en ligne droite vers (target_x, target_y, target_z).

        Utilise compute_cartesian_path puis execute_trajectory.
        L'orientation est maintenue (on conserve celle de la pose courante).
        """
        if not self._securite_ok:
            self.get_logger().error(f'SÉCURITÉ active — "{label}" cartésien bloqué.')
            return False

        # Orientation figée au début du cycle (joint5 colinéaire à Z)
        # Si pas encore capturée, on la lit depuis la FK courante
        if self._cycle_orientation is None:
            current_pose = self._fk()
            if current_pose is None:
                self.get_logger().error(f"FK indisponible — impossible de lancer {label} en cartésien")
                return False
            self._cycle_orientation = copy.deepcopy(current_pose.orientation)
            self.get_logger().info(
                f"  Orientation capturée : "
                f"x={self._cycle_orientation.x:.3f} "
                f"y={self._cycle_orientation.y:.3f} "
                f"z={self._cycle_orientation.z:.3f} "
                f"w={self._cycle_orientation.w:.3f}"
            )

        # Cible : orientation figée du cycle + nouvelle position
        target_pose = Pose()
        target_pose.position.x    = float(target_x)
        target_pose.position.y    = float(target_y)
        target_pose.position.z    = float(target_z)
        target_pose.orientation   = copy.deepcopy(self._cycle_orientation)

        # Prépare la requête compute_cartesian_path
        req = GetCartesianPath.Request()
        req.header.frame_id    = BASE_FRAME
        req.group_name         = ARM_GROUP
        req.link_name          = EE_LINK
        req.waypoints          = [target_pose]
        req.max_step           = CARTESIAN_MAX_STEP
        req.jump_threshold     = CARTESIAN_JUMP_THRESH
        req.avoid_collisions   = True
        req.max_velocity_scaling_factor     = self._vel()
        req.max_acceleration_scaling_factor = self._acc()

        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(self._current_joints)
        req.start_state = rs

        self.get_logger().info(
            f"  → [cartésien] {label}  "
            f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f}) m"
        )

        res = self._wait(self._cart_cli.call_async(req), timeout=15.0)
        if res is None:
            self.get_logger().error(f"  ✗ compute_cartesian_path timeout : {label}")
            return False

        if res.fraction < CARTESIAN_MIN_FRAC:
            self.get_logger().error(
                f"  ✗ {label} : chemin cartésien incomplet "
                f"({res.fraction*100:.1f}% < {CARTESIAN_MIN_FRAC*100:.0f}%) "
                "— obstacle ou limite articulaire en chemin ?"
            )
            return False

        self.get_logger().info(
            f"  Chemin cartésien calculé à {res.fraction*100:.1f}% "
            f"({len(res.solution.joint_trajectory.points)} points)"
        )

        # Exécution de la trajectoire
        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = res.solution

        gh = self._wait(self._exec.send_goal_async(exec_goal), timeout=12.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ execute_trajectory rejeté : {label}")
            return False

        result_future = gh.get_result_async()
        deadline = time.time() + 60.0
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.05)
            if not self._securite_ok:
                gh.cancel_goal_async()
                return False

        if not result_future.done():
            self.get_logger().warn(f"  ✗ Timeout exécution cartésienne : {label}")
            gh.cancel_goal_async()
            time.sleep(0.5)
            return False

        code = result_future.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"  ✓ {label}")
            return True

        self.get_logger().warn(f"  ✗ {label} — code ExecuteTrajectory {code}")
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

    # ── Attente serveurs ──────────────────────────────────────────

    def wait_for_servers(self) -> bool:
        log = self.get_logger()
        self._busy = True

        log.info("Attente /move_group…")
        if not self._mg.wait_for_server(timeout_sec=15.0):
            log.error("/move_group indisponible."); self._busy = False; return False

        log.info("Attente /execute_trajectory…")
        if not self._exec.wait_for_server(timeout_sec=10.0):
            log.error("/execute_trajectory indisponible."); self._busy = False; return False

        log.info("Attente /compute_ik…")
        if not self._ik_cli.wait_for_service(timeout_sec=10.0):
            log.error("/compute_ik indisponible."); self._busy = False; return False

        log.info("Attente /compute_fk…")
        if not self._fk_cli.wait_for_service(timeout_sec=10.0):
            log.error("/compute_fk indisponible."); self._busy = False; return False

        log.info("Attente /compute_cartesian_path…")
        if not self._cart_cli.wait_for_service(timeout_sec=10.0):
            log.error("/compute_cartesian_path indisponible."); self._busy = False; return False

        log.info("Synchronisation sécurité…")
        for _ in range(30):
            time.sleep(0.05)

        log.info("Envoi HOME initial…")
        if not self._send_joint(self._joint_goal(HOME_JOINTS), "HOME_INIT"):
            log.warn("HOME initial échoué — continuez depuis la position actuelle.")

        time.sleep(1.0)
        self._busy = False
        return True

    # ── Cycle pick → place (hybride joint + cartésien) ────────────

    def _run_cycle(self, pick_x, pick_y, pick_z, place_x, place_y, place_z):
        log = self.get_logger()
        _ok = False
        self._cycle_orientation = None  # réinitialise à chaque cycle

        # Vérification Z
        for label, z in [
            ("PICK",          pick_z),
            ("APPROCHE_PICK", pick_z  + APPROACH_OFFSET),
            ("PLACE",         place_z),
            ("APPROCHE_PLACE",place_z + APPROACH_OFFSET),
        ]:
            if not self._z_safe(z, label):
                return

        try:
            # Pré-calcul IK pour les points d'approche (joint space)
            log.info("Calcul IK des points d'approche…")
            ik_app_pick = self._ik(
                pick_x, pick_y, pick_z + APPROACH_OFFSET, "APPROCHE_PICK"
            )
            if ik_app_pick is None: return

            ik_app_place = self._ik(
                place_x, place_y, place_z + APPROACH_OFFSET, "APPROCHE_PLACE",
                seed=ik_app_pick
            )
            if ik_app_place is None: return

            log.info("=" * 58)
            log.info(f"  PICK  = ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f}) m")
            log.info(f"  PLACE = ({place_x:.3f}, {place_y:.3f}, {place_z:.3f}) m")
            log.info("  Segments cartésiens : descente/montée pick & place")
            log.info("=" * 58)

            # [1] HOME
            log.info("[1] HOME")
            if not self._send_joint(self._joint_goal(HOME_JOINTS), "HOME"):
                return

            # [2] Ouvrir pince
            log.info("[2] Ouverture pince")
            self._gripper(close=False)

            # [3] Approche pick (joint space — repositionnement)
            log.info("[3] Approche Pick (joint space)")
            if not self._send_joint(self._joint_goal(ik_app_pick), "APPROCHE_PICK"):
                return

            # Capture l'orientation réelle à l'approche pick (joint5 le long de Z)
            # Tous les segments cartésiens du cycle utiliseront cette orientation figée
            pose_approche = self._fk()
            if pose_approche is not None:
                self._cycle_orientation = copy.deepcopy(pose_approche.orientation)
                log.info(
                    f"  Orientation axe outil figée : "
                    f"x={self._cycle_orientation.x:.3f} "
                    f"y={self._cycle_orientation.y:.3f} "
                    f"z={self._cycle_orientation.z:.3f} "
                    f"w={self._cycle_orientation.w:.3f}"
                )

            # [4] Descente vers pick (CARTÉSIEN — ligne droite verticale)
            log.info("[4] Descente Pick (cartésien ↓)")
            if not self._send_cartesian(pick_x, pick_y, pick_z, "PICK"):
                return

            # [5] Saisie
            log.info("[5] Saisie")
            self._gripper(close=True)

            # [6] Levée pick (CARTÉSIEN — ligne droite verticale)
            log.info("[6] Levée Pick (cartésien ↑)")
            if not self._send_cartesian(
                pick_x, pick_y, pick_z + LIFT_OFFSET, "LIFT_PICK"
            ):
                return

            # [7] Approche place (joint space — repositionnement)
            log.info("[7] Approche Place (joint space)")
            if not self._send_joint(self._joint_goal(ik_app_place), "APPROCHE_PLACE"):
                return

            # [8] Descente vers place (CARTÉSIEN)
            log.info("[8] Descente Place (cartésien ↓)")
            if not self._send_cartesian(place_x, place_y, place_z, "PLACE"):
                return

            # [9] Lâcher
            log.info("[9] Lâcher")
            self._gripper(close=False)

            # [10] Levée place (CARTÉSIEN)
            log.info("[10] Levée Place (cartésien ↑)")
            if not self._send_cartesian(
                place_x, place_y, place_z + LIFT_OFFSET, "LIFT_PLACE"
            ):
                return

            # [11] Retour HOME
            log.info("[11] Retour HOME")
            self._send_joint(self._joint_goal(HOME_JOINTS), "HOME_FINAL")

            log.info("=" * 58)
            log.info("  CYCLE TERMINÉ AVEC SUCCÈS")
            log.info("=" * 58)
            _ok = True

        finally:
            if not _ok and self._securite_ok:
                log.info("⟲  Interruption — retour HOME…")
                self._send_joint(self._joint_goal(HOME_JOINTS), "HOME_RECOVERY")
            self._busy = False

    def run(self):
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
    node = PickPlaceCartesien()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not node.wait_for_servers():
            return
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
