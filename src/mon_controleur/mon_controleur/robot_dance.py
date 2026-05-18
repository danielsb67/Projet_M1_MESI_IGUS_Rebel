#!/usr/bin/env python3
"""
Robot Dance — Igus Rebel
========================
Chorégraphie de démo : trace des formes géométriques, courbes paramétriques,
ou exécute une "joint dance" style stand robotique.

Modes (paramètre ROS `dance`) :
  shapes      : cercle, carré, étoile à 5 branches, hexagone, triangle
  lemniscate  : symbole infini ∞ + spirale ascendante
  joint_dance : chaque articulation se présente à tour de rôle
  parametric  : coeur paramétrique + rosace à 5 pétales
  all         : enchaîne tout (par défaut)

Tous les tracés en cartésien sont planifiés via /compute_cartesian_path
(trajectoires rectilignes en ligne). L'orientation du gripper est figée
en début de chorégraphie (capturée depuis la FK courante).

Usage :
  ros2 run mon_controleur robot_dance --ros-args -p dance:=shapes
  ros2 launch mon_controleur sim_robot_dance.launch.py dance:=all
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

from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetCartesianPath


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

BASE_FRAME  = "igus_rebel_base_link"
ARM_GROUP   = "rebel_arm"
EE_LINK     = "gripper_tip_link"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# Position de référence — bras un peu replié, gripper pointe vers l'avant en bas
HOME_JOINTS = [0.15, 0.4, 0.7, 0.05, 1.0, 1.1]

# Centre de la scène de danse (repère base) — face au robot, à mi-hauteur
DANCE_CENTER_X = 0.40
DANCE_CENTER_Y = 0.00
DANCE_CENTER_Z = 0.30

# Résolution chemin cartésien
CART_MAX_STEP    = 0.005   # 5 mm — trajectoire lisse
CART_JUMP_THRESH = 0.0
CART_MIN_FRAC    = 0.90    # accepte 90 % du chemin (les formes complexes ratent parfois 1-2 pts)

# Vitesses
VELOCITY_SCALE     = 0.40
ACCELERATION_SCALE = 0.25


# ══════════════════════════════════════════════════════════════════
#  GÉNÉRATEURS DE FORMES — renvoient des offsets (dx, dy, dz)
#  par rapport au centre DANCE_CENTER. Plan par défaut : XZ (vertical face au robot).
# ══════════════════════════════════════════════════════════════════

def _circle(n=48, radius=0.10):
    """Cercle dans le plan XZ (vertical, face robot)."""
    pts = []
    for i in range(n + 1):  # +1 pour refermer
        t = 2 * math.pi * i / n
        pts.append((radius * math.cos(t), 0.0, radius * math.sin(t)))
    return pts


def _square(side=0.15):
    """Carré dans XZ (centré)."""
    s = side / 2
    return [(s, 0, -s), (s, 0, s), (-s, 0, s), (-s, 0, -s), (s, 0, -s)]


def _triangle(radius=0.10):
    """Triangle équilatéral pointe vers le haut."""
    pts = []
    for i in range(4):
        t = math.pi / 2 + 2 * math.pi * i / 3
        pts.append((radius * math.cos(t), 0.0, radius * math.sin(t)))
    return pts


def _hexagon(radius=0.10):
    pts = []
    for i in range(7):
        t = 2 * math.pi * i / 6
        pts.append((radius * math.cos(t), 0.0, radius * math.sin(t)))
    return pts


def _star(n_branches=5, outer=0.12, inner=0.05):
    """Étoile à n branches (5 par défaut)."""
    pts = []
    total = 2 * n_branches
    for i in range(total + 1):
        r = outer if i % 2 == 0 else inner
        t = math.pi / 2 + 2 * math.pi * i / total
        pts.append((r * math.cos(t), 0.0, r * math.sin(t)))
    return pts


def _lemniscate(scale=0.13, n=120):
    """Lemniscate de Bernoulli (symbole ∞), plan XZ."""
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        denom = 1 + math.sin(t) ** 2
        x = scale * math.cos(t) / denom
        z = scale * math.sin(t) * math.cos(t) / denom
        pts.append((x, 0.0, z))
    return pts


def _spiral(r_max=0.10, height=0.18, turns=3.0, n=120):
    """Spirale ascendante depuis -height/2 jusqu'à +height/2."""
    pts = []
    for i in range(n + 1):
        u = i / n
        r = r_max * u
        theta = 2 * math.pi * turns * u
        pts.append((r * math.cos(theta), 0.0, -height / 2 + height * u))
    # Retour au centre depuis le sommet
    pts.append((0.0, 0.0, height / 2))
    return pts


def _heart(scale=0.007, n=100):
    """Coeur paramétrique (classique 16*sin³t)."""
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        x = scale * 16 * math.sin(t) ** 3
        z = scale * (13 * math.cos(t)
                     - 5 * math.cos(2 * t)
                     - 2 * math.cos(3 * t)
                     - math.cos(4 * t))
        pts.append((x, 0.0, z))
    return pts


def _rose(petals=5, radius=0.12, n=200):
    """Rosace polaire r = R*cos(k*θ), plan XZ."""
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        r = radius * math.cos(petals * t)
        pts.append((r * math.cos(t), 0.0, r * math.sin(t)))
    return pts


# ══════════════════════════════════════════════════════════════════
#  JOINT SHOWCASE — pour le mode joint_dance
# ══════════════════════════════════════════════════════════════════
# Chaque entrée : (joint_idx, [pose_idx1, pose_idx2, ...]) où chaque pose_idx
# est la liste des 6 valeurs articulaires.

def _joint_showcase_sequences():
    """Renvoie une liste de séquences, une par articulation."""
    h = list(HOME_JOINTS)
    seqs = []

    # joint1 : rotation base (gauche - droite - centre)
    s1 = [h[:], [-math.pi/2, *h[1:]], [math.pi/2, *h[1:]], h[:]]
    seqs.append(("joint1 — rotation base", s1))

    # joint2 : "plonge" épaule
    s2 = [h[:],
          [h[0], -math.pi/4, *h[2:]],
          [h[0],  math.pi/3, *h[2:]],
          h[:]]
    seqs.append(("joint2 — plongée épaule", s2))

    # joint3 : coude
    s3 = [h[:],
          [*h[:2],  math.pi/3, *h[3:]],
          [*h[:2], -math.pi/4, *h[3:]],
          h[:]]
    seqs.append(("joint3 — coude", s3))

    # joint4 : roll avant-bras
    s4 = [h[:],
          [*h[:3], -math.pi/2, *h[4:]],
          [*h[:3],  math.pi/2, *h[4:]],
          h[:]]
    seqs.append(("joint4 — roll avant-bras", s4))

    # joint5 : pitch poignet
    s5 = [h[:],
          [*h[:4],  math.pi/3, h[5]],
          [*h[:4], -math.pi/3, h[5]],
          h[:]]
    seqs.append(("joint5 — pitch poignet", s5))

    # joint6 : spin gripper
    s6 = [h[:],
          [*h[:5], -math.pi],
          [*h[:5],  math.pi],
          h[:]]
    seqs.append(("joint6 — spin gripper", s6))

    # FINALE : wave (tous les joints en cascade puis retour HOME)
    finale = [h[:]]
    for i in range(6):
        p = h[:]
        p[i] = h[i] + (math.pi / 4 if i != 0 else math.pi / 3)
        finale.append(p)
    finale.append(h[:])
    seqs.append(("FINALE — wave en cascade", finale))

    return seqs


# ══════════════════════════════════════════════════════════════════
class RobotDance(Node):

    def __init__(self):
        super().__init__("robot_dance")

        self.declare_parameter("dance", "all")
        self.declare_parameter("vel_scale", VELOCITY_SCALE)
        self.declare_parameter("acc_scale", ACCELERATION_SCALE)
        self.declare_parameter("center_x", DANCE_CENTER_X)
        self.declare_parameter("center_y", DANCE_CENTER_Y)
        self.declare_parameter("center_z", DANCE_CENTER_Z)

        self._cbg = ReentrantCallbackGroup()

        self._mg   = ActionClient(self, MoveGroup,        "/move_action",        callback_group=self._cbg)
        self._exec = ActionClient(self, ExecuteTrajectory,"/execute_trajectory", callback_group=self._cbg)

        self._ik_cli   = self.create_client(GetPositionIK,    "/compute_ik",             callback_group=self._cbg)
        self._fk_cli   = self.create_client(GetPositionFK,    "/compute_fk",             callback_group=self._cbg)
        self._cart_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=self._cbg)

        self._securite_ok    = True
        self._current_joints = list(HOME_JOINTS)
        self._dance_orientation = None

        self.create_subscription(Bool,       "/securite_mouvement", self._securite_cb,   10, callback_group=self._cbg)
        self.create_subscription(JointState, "/joint_states",        self._joint_state_cb, 10, callback_group=self._cbg)

        self.get_logger().info("RobotDance prêt — let's dance!")

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

    def _center(self):
        return (
            float(self.get_parameter("center_x").get_parameter_value().double_value),
            float(self.get_parameter("center_y").get_parameter_value().double_value),
            float(self.get_parameter("center_z").get_parameter_value().double_value),
        )

    # ── IK : XYZ → joints ───────────────────────────────────────
    def _ik(self, x, y, z, label=""):
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
        rs.joint_state.position = list(self._current_joints)
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

    # ── Goal joint space (Pilz PTP) ─────────────────────────────
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

    # ── Tracé cartésien (multi-waypoints) ───────────────────────
    def _trace_path(self, offsets, label: str) -> bool:
        """Trace une forme : offsets = liste de (dx, dy, dz) relatifs au centre.
        L'orientation est figée (_dance_orientation)."""
        if not self._securite_ok:
            self.get_logger().error(f"SÉCURITÉ — {label} bloqué.")
            return False

        if self._dance_orientation is None:
            pose = self._fk()
            if pose is None:
                self.get_logger().error("FK indisponible — orientation non capturée.")
                return False
            self._dance_orientation = copy.deepcopy(pose.orientation)

        cx, cy, cz = self._center()
        waypoints = []
        for dx, dy, dz in offsets:
            p = Pose()
            p.position.x = cx + dx
            p.position.y = cy + dy
            p.position.z = cz + dz
            p.orientation = copy.deepcopy(self._dance_orientation)
            waypoints.append(p)

        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.group_name       = ARM_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = waypoints
        req.max_step         = CART_MAX_STEP
        req.jump_threshold   = CART_JUMP_THRESH
        req.avoid_collisions = True
        req.max_velocity_scaling_factor     = self._vel()
        req.max_acceleration_scaling_factor = self._acc()

        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(self._current_joints)
        req.start_state = rs

        self.get_logger().info(f"  → [cartésien] {label} ({len(waypoints)} pts)")
        res = self._wait(self._cart_cli.call_async(req), timeout=20.0)
        if res is None:
            self.get_logger().error(f"  ✗ compute_cartesian_path timeout : {label}")
            return False
        if res.fraction < CART_MIN_FRAC:
            self.get_logger().warn(
                f"  ⚠ {label} : chemin partiel ({res.fraction*100:.0f}%) — exécution quand même")

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

    # ── Préparation : aller au centre de la scène ───────────────
    def _go_to_dance_center(self) -> bool:
        cx, cy, cz = self._center()
        self.get_logger().info(f"Positionnement au centre de la scène ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        joints = self._ik(cx, cy, cz, "DANCE_CENTER")
        if joints is None:
            return False
        if not self._send_joint(joints, "DANCE_CENTER"):
            return False
        # Capture orientation après arrivée
        time.sleep(0.5)
        pose = self._fk()
        if pose is not None:
            self._dance_orientation = copy.deepcopy(pose.orientation)
            self.get_logger().info("Orientation de danse figée.")
        return True

    # ══════════════════════════════════════════════════════════
    #  CHORÉGRAPHIES
    # ══════════════════════════════════════════════════════════

    def dance_shapes(self):
        log = self.get_logger()
        log.info("═" * 58)
        log.info("  🎨  FORMES GÉOMÉTRIQUES")
        log.info("═" * 58)
        shapes = [
            ("CERCLE",   _circle(n=48, radius=0.10)),
            ("CARRÉ",    _square(side=0.16)),
            ("TRIANGLE", _triangle(radius=0.10)),
            ("HEXAGONE", _hexagon(radius=0.10)),
            ("ÉTOILE ★", _star(n_branches=5, outer=0.12, inner=0.05)),
        ]
        for name, pts in shapes:
            if not self._trace_path(pts, name):
                log.warn(f"Forme {name} échouée — on continue")
            time.sleep(0.3)

    def dance_lemniscate(self):
        log = self.get_logger()
        log.info("═" * 58)
        log.info("  ♾️   LEMNISCATE + SPIRALE")
        log.info("═" * 58)
        if not self._trace_path(_lemniscate(scale=0.13, n=120), "LEMNISCATE ∞"):
            log.warn("Lemniscate échoué — on continue")
        time.sleep(0.5)
        if not self._trace_path(_spiral(r_max=0.10, height=0.18, turns=3.0, n=120), "SPIRALE"):
            log.warn("Spirale échouée")

    def dance_joints(self):
        log = self.get_logger()
        log.info("═" * 58)
        log.info("  🤖  JOINT DANCE — chaque articulation se présente")
        log.info("═" * 58)
        # Retour à HOME d'abord
        self._send_joint(HOME_JOINTS, "HOME")
        for name, sequence in _joint_showcase_sequences():
            log.info(f"  ▶ {name}")
            for pose in sequence:
                if not self._send_joint(pose, "→"):
                    log.warn(f"  étape échouée pour {name}")
                    break
            time.sleep(0.3)

    def dance_parametric(self):
        log = self.get_logger()
        log.info("═" * 58)
        log.info("  ❤️   COURBES PARAMÉTRIQUES")
        log.info("═" * 58)
        if not self._trace_path(_heart(scale=0.007, n=100), "COEUR ❤"):
            log.warn("Coeur échoué")
        time.sleep(0.5)
        if not self._trace_path(_rose(petals=5, radius=0.12, n=200), "ROSACE 5 pétales"):
            log.warn("Rosace échouée")

    # ── Point d'entrée ──────────────────────────────────────────
    def wait_for_servers(self) -> bool:
        log = self.get_logger()
        log.info("Attente /move_group…")
        if not self._mg.wait_for_server(timeout_sec=15.0):
            log.error("/move_group indisponible."); return False
        log.info("Attente /execute_trajectory…")
        if not self._exec.wait_for_server(timeout_sec=10.0):
            log.error("/execute_trajectory indisponible."); return False
        for cli, name in [(self._ik_cli, "/compute_ik"),
                          (self._fk_cli, "/compute_fk"),
                          (self._cart_cli, "/compute_cartesian_path")]:
            log.info(f"Attente {name}…")
            if not cli.wait_for_service(timeout_sec=10.0):
                log.error(f"{name} indisponible."); return False
        log.info("Synchronisation sécurité…")
        for _ in range(30):
            time.sleep(0.05)
        log.info("Retour HOME au démarrage…")
        self._send_joint(HOME_JOINTS, "HOME_INIT")
        time.sleep(1.0)
        return True

    def run(self):
        dance = self.get_parameter("dance").get_parameter_value().string_value.lower()
        log = self.get_logger()
        log.info(f"🎵  Choreographie demandée : '{dance}'")

        if not self._go_to_dance_center():
            log.error("Impossible d'atteindre le centre de la scène — arrêt.")
            return

        if dance in ("shapes", "all"):
            self.dance_shapes()
        if dance in ("lemniscate", "all"):
            self.dance_lemniscate()
        if dance in ("parametric", "all"):
            self.dance_parametric()
        if dance in ("joint_dance", "all"):
            self.dance_joints()

        log.info("═" * 58)
        log.info("  🎤  SPECTACLE TERMINÉ — retour HOME")
        log.info("═" * 58)
        self._send_joint(HOME_JOINTS, "HOME_FINAL")


# ══════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = RobotDance()
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
