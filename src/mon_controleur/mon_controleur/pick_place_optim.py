#!/usr/bin/env python3
"""
Pick & Place OPTIMISÉ — Igus Rebel + Schunk EGP25
==================================================
Trajectoire continue sans arrêts brusques aux coins (LIFT, APPROCHE).
Inspiré du "trajectory blending" industriel (FANUC, KUKA, ABB).

Principe :
  - Les coins (LIFT_PICK, APPROCHE_PLACE, LIFT_PLACE) sont ARRONDIS
    avec une courbe de Bézier quadratique → mouvement fluide en "U".
  - Le robot ne s'arrête QU'À PICK et PLACE (obligatoire : gripper).
  - Tous les segments fluides sont calculés en CARTÉSIEN
    (compute_cartesian_path) avec un grand nombre de points.
  - L'orientation est figée pour tout le cycle.

Architecture des segments :
  [1] HOME → APPROCHE_PICK              : joint space (grand déplacement)
  [2] APPROCHE_PICK → PICK              : cartésien droit (descente)
  [3] PICK : pince ferme (arrêt obligatoire)
  [4] PICK → LIFT_PICK → APPROCHE_PLACE → PLACE
                                        : cartésien FLUIDE (2 coins arrondis)
  [5] PLACE : pince ouvre (arrêt obligatoire)
  [6] PLACE → LIFT_PLACE → APPROCHE_PICK_HOMING
                                        : cartésien FLUIDE (1 coin arrondi)
  [7] retour HOME                       : joint space

Usage :
  ros2 run mon_controleur pick_place_optim --ros-args -p use_sim_time:=true
  ros2 launch mon_controleur sim_pick_place_optim.launch.py
"""
import math
import time
import copy
import threading

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
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetCartesianPath


# ══════════════════════════════════════════════════════════════════
#  COORDONNÉES (modifiables via paramètres ROS)
# ══════════════════════════════════════════════════════════════════

ERREUR_CALIBRATION_Z = -0.01

PICK_X,  PICK_Y,  PICK_Z  = 0.4,  0.15,  0.0 + ERREUR_CALIBRATION_Z
PLACE_X, PLACE_Y, PLACE_Z = 0.4, -0.15,  0.0 + ERREUR_CALIBRATION_Z

APPROACH_OFFSET = 0.10
LIFT_OFFSET     = 0.10

# ── Paramètres de blending (filleting Bézier) ──────────────────
BLEND_RADIUS    = 0.04   # 4 cm — rayon d'arrondi aux coins
BLEND_SAMPLES   = 20     # pts par courbe Bézier
CART_MAX_STEP   = 0.003  # 3 mm — trajectoire très lisse
CART_MIN_FRAC   = 0.85   # accepte 85 % du chemin
CART_JUMP_THR   = 0.0

# ══════════════════════════════════════════════════════════════════
BASE_FRAME  = "igus_rebel_base_link"
ARM_GROUP   = "rebel_arm"
EE_LINK     = "gripper_tip_link"
GRIPPER_SRV = "/gripper/command"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_JOINTS = [0.15, 0.4, 0.7, 0.05, 1.0, 1.1]
Z_MIN_M     = 0.0 + ERREUR_CALIBRATION_Z

VELOCITY_SCALE     = 0.30
ACCELERATION_SCALE = 0.20
GRIPPER_WAIT       = 1.5


# ══════════════════════════════════════════════════════════════════
#  HELPERS GÉOMÉTRIQUES — fillet Bézier
# ══════════════════════════════════════════════════════════════════

def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _scale(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)

def _norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def _unit(v):
    n = _norm(v)
    return (v[0]/n, v[1]/n, v[2]/n) if n > 1e-9 else (0., 0., 0.)


def fillet_corner(prev_pt, corner, next_pt, radius, n_samples):
    """Arrondit le coin `corner` entre `prev_pt` et `next_pt`.

    Retourne (entry_pt, [points_bezier_intermediaires], exit_pt).
    La courbe Bézier quadratique a pour control points (entry, corner, exit).
    Si une arête est trop courte, le rayon est clampé à 49 % de sa longueur.
    """
    v_back = _sub(prev_pt, corner)   # corner → prev
    v_next = _sub(next_pt, corner)   # corner → next
    len_back = _norm(v_back)
    len_next = _norm(v_next)

    # rayon adaptatif : ≤ 49 % de la plus courte arête
    r = min(radius, 0.49 * len_back, 0.49 * len_next)
    if r < 1e-4:
        # Trop court pour fillet : on garde le coin pointu
        return corner, [], corner

    entry = _add(corner, _scale(_unit(v_back), r))
    exit_ = _add(corner, _scale(_unit(v_next), r))

    # Bézier quadratique B(t) = (1-t)²·entry + 2(1-t)t·corner + t²·exit
    points = []
    for i in range(1, n_samples):
        t = i / n_samples
        u = 1.0 - t
        b0 = u * u
        b1 = 2 * u * t
        b2 = t * t
        x = b0*entry[0] + b1*corner[0] + b2*exit_[0]
        y = b0*entry[1] + b1*corner[1] + b2*exit_[1]
        z = b0*entry[2] + b1*corner[2] + b2*exit_[2]
        points.append((x, y, z))

    return entry, points, exit_


def smooth_path(keypoints, radius, n_samples):
    """Transforme une polyligne pointue en chemin fluide avec fillets aux coins.

    keypoints : liste de tuples (x, y, z)
    Le premier et le dernier point sont conservés tels quels.
    Chaque coin intérieur est remplacé par : entry → Bézier → exit.
    """
    if len(keypoints) <= 2:
        return list(keypoints)

    result = [keypoints[0]]
    for i in range(1, len(keypoints) - 1):
        entry, bezier_pts, exit_ = fillet_corner(
            keypoints[i-1], keypoints[i], keypoints[i+1], radius, n_samples
        )
        # Ligne droite jusqu'à entry, puis courbe Bézier
        if entry != result[-1]:
            result.append(entry)
        result.extend(bezier_pts)
        result.append(exit_)
    result.append(keypoints[-1])
    return result


# ══════════════════════════════════════════════════════════════════
class PickPlaceOptim(Node):

    def __init__(self):
        super().__init__("pick_place_optim")

        self.declare_parameter("pick_x",    PICK_X)
        self.declare_parameter("pick_y",    PICK_Y)
        self.declare_parameter("pick_z",    PICK_Z)
        self.declare_parameter("place_x",   PLACE_X)
        self.declare_parameter("place_y",   PLACE_Y)
        self.declare_parameter("place_z",   PLACE_Z)
        self.declare_parameter("vel_scale", VELOCITY_SCALE)
        self.declare_parameter("acc_scale", ACCELERATION_SCALE)
        self.declare_parameter("blend_radius", BLEND_RADIUS)

        self._cbg = ReentrantCallbackGroup()

        self._mg    = ActionClient(self, MoveGroup,         "/move_action",         callback_group=self._cbg)
        self._exec  = ActionClient(self, ExecuteTrajectory, "/execute_trajectory",  callback_group=self._cbg)

        self._ik_cli      = self.create_client(GetPositionIK,    "/compute_ik",             callback_group=self._cbg)
        self._fk_cli      = self.create_client(GetPositionFK,    "/compute_fk",             callback_group=self._cbg)
        self._cart_cli    = self.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=self._cbg)
        self._gripper_cli = self.create_client(SetBool,          GRIPPER_SRV,               callback_group=self._cbg)

        self._securite_ok        = True
        self._current_joints     = list(HOME_JOINTS)
        self._cycle_orientation  = None

        self.create_subscription(Bool,       "/securite_mouvement", self._securite_cb,    10, callback_group=self._cbg)
        self.create_subscription(JointState, "/joint_states",        self._joint_state_cb, 10, callback_group=self._cbg)

        self.get_logger().info("PickPlaceOptim démarré — trajectoires fluides avec blending")

    # ── Callbacks ────────────────────────────────────────────────
    def _securite_cb(self, msg: Bool):
        self._securite_ok = msg.data

    def _joint_state_cb(self, msg: JointState):
        nm = dict(zip(msg.name, msg.position))
        self._current_joints = [nm.get(n, 0.0) for n in JOINT_NAMES]

    # ── Utils ────────────────────────────────────────────────────
    def _wait(self, future, timeout=20.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

    def _vel(self):
        return float(self.get_parameter("vel_scale").get_parameter_value().double_value)

    def _acc(self):
        return float(self.get_parameter("acc_scale").get_parameter_value().double_value)

    def _radius(self):
        return float(self.get_parameter("blend_radius").get_parameter_value().double_value)

    def _z_safe(self, z, label):
        if z < Z_MIN_M:
            self.get_logger().error(f"SÉCURITÉ — {label} Z={z:.3f} < {Z_MIN_M}")
            return False
        return True

    # ── IK ───────────────────────────────────────────────────────
    def _ik(self, x, y, z, label, seed=None):
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

        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(seed if seed is not None else self._current_joints)
        req.ik_request.robot_state = rs

        res = self._wait(self._ik_cli.call_async(req), timeout=8.0)
        if res is None or res.error_code.val != 1:
            self.get_logger().error(f"IK échec : {label} ({x:.3f}, {y:.3f}, {z:.3f})")
            return None
        js = res.solution.joint_state
        return [js.position[js.name.index(n)] for n in JOINT_NAMES if n in js.name]

    def _fk(self):
        req = GetPositionFK.Request()
        req.header.frame_id = BASE_FRAME
        req.fk_link_names   = [EE_LINK]
        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(self._current_joints)
        req.robot_state = rs
        res = self._wait(self._fk_cli.call_async(req), timeout=5.0)
        if res is None or not res.pose_stamped:
            return None
        return res.pose_stamped[0].pose

    # ── Goal joint space (Pilz PTP) — pour grands déplacements ──
    def _joint_goal(self, positions) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name = ARM_GROUP
        req.num_planning_attempts = 3
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor     = self._vel()
        req.max_acceleration_scaling_factor = self._acc()
        req.pipeline_id = "pilz_industrial_motion_planner"
        req.planner_id  = "PTP"

        c = Constraints()
        for n, p in zip(JOINT_NAMES, positions):
            jc = JointConstraint()
            jc.joint_name      = n
            jc.position        = float(p)
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        goal.planning_options.plan_only = False
        return goal

    def _send_joint(self, positions, label) -> bool:
        if not self._securite_ok:
            self.get_logger().error(f"SÉCURITÉ — {label} bloqué.")
            return False
        if all(abs(c - t) < 0.05 for c, t in zip(self._current_joints, positions)):
            self.get_logger().info(f"  ✓ {label} (déjà en position)")
            return True
        self.get_logger().info(f"  → [joint] {label}")
        gh = self._wait(self._mg.send_goal_async(self._joint_goal(positions)), timeout=8.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ Goal rejeté : {label}")
            return False
        rf = gh.get_result_async()
        deadline = time.time() + 30.0
        while not rf.done() and time.time() < deadline:
            time.sleep(0.05)
            if not self._securite_ok:
                gh.cancel_goal_async()
                return False
        if not rf.done():
            self.get_logger().warn(f"  ✗ Timeout {label}")
            gh.cancel_goal_async()
            return False
        code = rf.result().result.error_code.val
        if code == 1:
            self.get_logger().info(f"  ✓ {label}")
            return True
        self.get_logger().warn(f"  ✗ {label} — code MoveIt {code}")
        return False

    # ── Tracé cartésien fluide multi-waypoints ──────────────────
    def _execute_smooth(self, points_xyz, label: str) -> bool:
        """Exécute une trajectoire cartésienne fluide passant par tous les points.

        points_xyz : liste de tuples (x, y, z) en repère base.
        L'orientation est figée (_cycle_orientation).
        """
        if not self._securite_ok:
            self.get_logger().error(f"SÉCURITÉ — {label} bloqué.")
            return False

        if self._cycle_orientation is None:
            pose = self._fk()
            if pose is None:
                self.get_logger().error("FK indisponible — orientation non capturée.")
                return False
            self._cycle_orientation = copy.deepcopy(pose.orientation)

        waypoints = []
        for x, y, z in points_xyz:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(z)
            p.orientation = copy.deepcopy(self._cycle_orientation)
            waypoints.append(p)

        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.group_name       = ARM_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = waypoints
        req.max_step         = CART_MAX_STEP
        req.jump_threshold   = CART_JUMP_THR
        req.avoid_collisions = True
        req.max_velocity_scaling_factor     = self._vel()
        req.max_acceleration_scaling_factor = self._acc()

        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(self._current_joints)
        req.start_state = rs

        self.get_logger().info(f"  → [cartésien fluide] {label} ({len(waypoints)} pts)")
        res = self._wait(self._cart_cli.call_async(req), timeout=20.0)
        if res is None:
            self.get_logger().error(f"  ✗ compute_cartesian_path timeout : {label}")
            return False
        if res.fraction < CART_MIN_FRAC:
            self.get_logger().error(
                f"  ✗ {label} : chemin incomplet ({res.fraction*100:.0f}%)")
            return False
        self.get_logger().info(
            f"  Chemin {res.fraction*100:.0f}% — {len(res.solution.joint_trajectory.points)} points")

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = res.solution

        gh = self._wait(self._exec.send_goal_async(exec_goal), timeout=12.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ execute_trajectory rejeté : {label}")
            return False

        rf = gh.get_result_async()
        deadline = time.time() + 60.0
        while not rf.done() and time.time() < deadline:
            time.sleep(0.05)
            if not self._securite_ok:
                gh.cancel_goal_async()
                return False

        if not rf.done():
            self.get_logger().warn(f"  ✗ Timeout exécution {label}")
            gh.cancel_goal_async()
            return False

        code = rf.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"  ✓ {label}")
            return True
        self.get_logger().warn(f"  ✗ {label} — code {code}")
        return False

    # ── Pince ────────────────────────────────────────────────────
    def _gripper(self, close: bool):
        self.get_logger().info(f"  Pince : {'FERMER' if close else 'OUVRIR'}")
        if not self._gripper_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("  Service pince indisponible.")
            return
        req = SetBool.Request()
        req.data = close
        self._wait(self._gripper_cli.call_async(req), timeout=4.0)
        time.sleep(GRIPPER_WAIT)

    # ── Attente serveurs ────────────────────────────────────────
    def wait_for_servers(self) -> bool:
        log = self.get_logger()
        log.info("Attente /move_group…")
        if not self._mg.wait_for_server(timeout_sec=15.0): return False
        log.info("Attente /execute_trajectory…")
        if not self._exec.wait_for_server(timeout_sec=10.0): return False
        for cli, name in [(self._ik_cli, "/compute_ik"),
                          (self._fk_cli, "/compute_fk"),
                          (self._cart_cli, "/compute_cartesian_path")]:
            log.info(f"Attente {name}…")
            if not cli.wait_for_service(timeout_sec=10.0): return False
        log.info("Synchronisation sécurité…")
        for _ in range(30):
            time.sleep(0.05)
        log.info("HOME initial…")
        self._send_joint(HOME_JOINTS, "HOME_INIT")
        time.sleep(1.0)
        return True

    # ── Cycle optimisé avec blending ────────────────────────────
    def _run_cycle(self, pick_x, pick_y, pick_z, place_x, place_y, place_z):
        log = self.get_logger()
        _ok = False
        self._cycle_orientation = None

        # Points clés du cycle (cartésiens)
        P_PICK     = (pick_x,  pick_y,  pick_z)
        P_APP_PICK = (pick_x,  pick_y,  pick_z  + APPROACH_OFFSET)
        P_LIFT_PK  = (pick_x,  pick_y,  pick_z  + LIFT_OFFSET)
        P_APP_PLC  = (place_x, place_y, place_z + APPROACH_OFFSET)
        P_PLACE    = (place_x, place_y, place_z)
        P_LIFT_PL  = (place_x, place_y, place_z + LIFT_OFFSET)

        # Validation Z
        for label, pt in [("PICK", P_PICK), ("APPROCHE_PICK", P_APP_PICK),
                          ("LIFT_PICK", P_LIFT_PK), ("APPROCHE_PLACE", P_APP_PLC),
                          ("PLACE", P_PLACE), ("LIFT_PLACE", P_LIFT_PL)]:
            if not self._z_safe(pt[2], label):
                return

        try:
            r = self._radius()

            log.info("=" * 60)
            log.info(f"  PICK  = ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f}) m")
            log.info(f"  PLACE = ({place_x:.3f}, {place_y:.3f}, {place_z:.3f}) m")
            log.info(f"  Blend radius = {r*1000:.0f} mm  ({BLEND_SAMPLES} pts/coin)")
            log.info("  → trajectoire fluide aux coins LIFT et APPROCHE")
            log.info("=" * 60)

            # [1] HOME (joint space)
            log.info("[1] HOME")
            if not self._send_joint(HOME_JOINTS, "HOME"):
                return

            # [2] Aller à APPROCHE_PICK via IK joint space
            log.info("[2] APPROCHE_PICK (joint space)")
            ik_app_pk = self._ik(*P_APP_PICK, "APPROCHE_PICK")
            if ik_app_pk is None: return
            if not self._send_joint(ik_app_pk, "APPROCHE_PICK"):
                return

            # Capture orientation pour tout le cycle cartésien
            time.sleep(0.3)
            pose_app = self._fk()
            if pose_app is None:
                log.error("FK indisponible — abandon")
                return
            self._cycle_orientation = copy.deepcopy(pose_app.orientation)
            log.info("  Orientation figée pour le cycle")

            # [3] Ouverture pince
            log.info("[3] Ouverture pince")
            self._gripper(close=False)

            # [4] Descente APPROCHE_PICK → PICK (droit, pas de coin)
            log.info("[4] Descente vers PICK (cartésien droit)")
            if not self._execute_smooth([P_APP_PICK, P_PICK], "DESCENTE_PICK"):
                return

            # [5] Saisie
            log.info("[5] Saisie")
            self._gripper(close=True)

            # [6] PICK → LIFT_PICK → APPROCHE_PLACE → PLACE
            #     coins LIFT_PICK et APPROCHE_PLACE arrondis !
            log.info("[6] Trajectoire FLUIDE : PICK → LIFT → cross → APPROCHE_PLACE → PLACE")
            traj_main = smooth_path(
                [P_PICK, P_LIFT_PK, P_APP_PLC, P_PLACE],
                radius=r, n_samples=BLEND_SAMPLES,
            )
            log.info(f"  {len(traj_main)} waypoints après blending")
            if not self._execute_smooth(traj_main, "FLUIDE_PICK_TO_PLACE"):
                return

            # [7] Lâcher
            log.info("[7] Lâcher")
            self._gripper(close=False)

            # [8] PLACE → LIFT_PLACE (droit, pas de coin)
            log.info("[8] Remontée depuis PLACE (cartésien droit)")
            if not self._execute_smooth([P_PLACE, P_LIFT_PL], "LIFT_PLACE"):
                return

            # [9] Retour HOME (joint space)
            log.info("[9] Retour HOME")
            self._send_joint(HOME_JOINTS, "HOME_FINAL")

            log.info("=" * 60)
            log.info("  CYCLE OPTIMISÉ TERMINÉ — aucun arrêt brusque aux coins")
            log.info("=" * 60)
            _ok = True

        finally:
            if not _ok and self._securite_ok:
                log.info("⟲  Interruption — retour HOME…")
                self._send_joint(HOME_JOINTS, "HOME_RECOVERY")

    def run(self):
        pick_x  = self.get_parameter("pick_x").get_parameter_value().double_value
        pick_y  = self.get_parameter("pick_y").get_parameter_value().double_value
        pick_z  = self.get_parameter("pick_z").get_parameter_value().double_value
        place_x = self.get_parameter("place_x").get_parameter_value().double_value
        place_y = self.get_parameter("place_y").get_parameter_value().double_value
        place_z = self.get_parameter("place_z").get_parameter_value().double_value
        self._run_cycle(pick_x, pick_y, pick_z, place_x, place_y, place_z)


# ══════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceOptim()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        if node.wait_for_servers():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
