#!/usr/bin/env python3
"""
Pick & Place INTELLIGENT — Igus Rebel + Schunk EGP25
Stratégie : /compute_ik (XYZ → joints) puis joint goal MoveIt.
OMPL est volontairement évité : il crashe (segfault) sur cette installation Humble.

Modifier PICK_X/Y/Z et PLACE_X/Y/Z pour choisir vos points.

Pré-requis :
  ros2 launch igus_rebel_moveit_config demo.launch.py \
      hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none camera:=none load_base:=false
"""
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped

# ══════════════════════════════════════════════════════════════════
#  VOS POINTS — modifier ici (en mètres, repère base du robot)
# ══════════════════════════════════════════════════════════════════

PICK_X,  PICK_Y,  PICK_Z  = 0.40, -0.15,  0.015  # Point A : centre objet (bout pince)
PLACE_X, PLACE_Y, PLACE_Z = 0.40,  0.15,  0.015  # Point B : dépôt  (bout pince)

APPROACH_OFFSET = 0.12   # remontée avant descente sur l'objet (m)
LIFT_OFFSET     = 0.12   # levée après saisie (m)

# ══════════════════════════════════════════════════════════════════
#  CONFIG ROBOT (ne pas modifier sauf si changement de config)
# ══════════════════════════════════════════════════════════════════

BASE_FRAME  = "igus_rebel_base_link"
ARM_GROUP   = "rebel_arm"
EE_LINK     = "gripper_tip_link"   # bout physique de la pince (18.5 cm du flange)
GRIPPER_SRV = "/gripper/command"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_JOINTS = [0.001, 0.92729345, 0.85, -0.00034906, 1.25, 1.1]  # position réelle mesurée

Z_MIN_M = 0.005  # plancher sécurité bout pince (m) — table ≈ Z=0, marge 5 mm

# ── Orientation pince vers le bas ─────────────────────────────────
# (0.5, 0.5, -0.5, 0.5) = rotation qui met l'axe X de l'effecteur vers -Z monde.
# Si la pince n'est pas vers le bas, essayer dans l'ordre :
#   (0.0, 0.7071, 0.0, 0.7071)  /  (0.7071, 0.0, 0.0, 0.7071)  /  (1,0,0,0)
GRIPPER_DOWN_Q = (0.0, 0.7071, 0.0, 0.7071) # (x, y, z, w)

# ── Blocage joint4 et joint6 ──────────────────────────────────────
# Valeurs AUTO-DÉRIVÉES de HOME_JOINTS → toujours cohérentes avec le robot réel.
# Ne pas mettre de valeur manuelle ici : ça crée des conflits IK.
JOINT4_LOCKED    = False
JOINT4_FIXED_DEG = math.degrees(HOME_JOINTS[JOINT_NAMES.index("joint4")])
JOINT4_TOLERANCE = 5.0    # tolérance ± degrés

JOINT6_LOCKED    = False
JOINT6_FIXED_DEG = math.degrees(HOME_JOINTS[JOINT_NAMES.index("joint6")])
JOINT6_TOLERANCE = 5.0

# ── Limites articulaires (avertissements post-IK) ─────────────────
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "joint1": (-60.0,  60.0),
    "joint2": ( -75.0,   75.0),
    "joint3": ( -120.0,   120.0),
    "joint4": (-180.0,  180.0),
    "joint5": (-120.0,  120.0),
    "joint6": (-180.0,  180.0),
}

# Seuil de détection des grands sauts articulaires (rad, somme sur 6 joints)
JUMP_WARN_RAD = 2.5


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _normalize(angle: float) -> float:
    """Ramène un angle dans [-π, π] pour éviter les tours superflus."""
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


VELOCITY_SCALE     = 0.1
ACCELERATION_SCALE = 0.05
GRIPPER_WAIT       = 1.5


# ══════════════════════════════════════════════════════════════════
class PickPlaceIA(Node):

    def __init__(self):
        super().__init__("pick_place_ia")
        self._mg          = ActionClient(self, MoveGroup, "/move_action")
        self._ik_cli      = self.create_client(GetPositionIK, '/compute_ik')
        self._gripper_cli = self.create_client(SetBool, GRIPPER_SRV)
        self._securite_ok    = True
        self._current_joints = list(HOME_JOINTS)
        self.create_subscription(Bool,       '/securite_mouvement', self._securite_cb,   10)
        self.create_subscription(JointState, '/joint_states',       self._joint_state_cb, 10)

    def _securite_cb(self, msg: Bool):
        self._securite_ok = msg.data

    def _joint_state_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        self._current_joints = [name_to_pos.get(n, 0.0) for n in JOINT_NAMES]

    # ── Spin helper ───────────────────────────────────────────────

    def _wait(self, future, timeout=50.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    # ── IK : XYZ → angles articulaires ───────────────────────────

    def _ik_to_joints(self, x, y, z, label='', seed=None, _no_retry=False):
        """Convertit une cible XYZ en angles articulaires via /compute_ik.
        seed : liste de positions articulaires utilisée comme point de départ IK.
               Si None, utilise la position courante du robot.
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name       = ARM_GROUP
        req.ik_request.ik_link_name     = EE_LINK
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec      = 5

        # Position uniquement — orientation non imposée ici.
        # joint4 + joint6 bloqués définissent déjà l'orientation de la pince.
        # Imposer GRIPPER_DOWN_Q en plus crée 6 contraintes pour 4 DDL → NO_IK_SOLUTION.
        qx, qy, qz, qw = GRIPPER_DOWN_Q
        
        ps = PoseStamped()
        ps.header.frame_id    = BASE_FRAME
        ps.pose.position.x    = float(x)
        ps.pose.position.y    = float(y)
        ps.pose.position.z    = float(z)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        req.ik_request.pose_stamped = ps

        # Seed state : guide l'IK vers la solution la plus proche
        seed_positions = seed if seed is not None else self._current_joints
        rs = RobotState()
        rs.joint_state.name     = list(JOINT_NAMES)
        rs.joint_state.position = list(seed_positions)
        req.ik_request.robot_state = rs

        # Blocage joint4 + joint6 : réduit l'IK à 4-DDL (joints 1,2,3,5)
        locked = []
        if JOINT4_LOCKED:
            jc4                 = JointConstraint()
            jc4.joint_name      = "joint4"
            jc4.position        = _deg2rad(JOINT4_FIXED_DEG)
            jc4.tolerance_above = _deg2rad(JOINT4_TOLERANCE)
            jc4.tolerance_below = _deg2rad(JOINT4_TOLERANCE)
            jc4.weight          = 1.0
            locked.append(jc4)
        if JOINT6_LOCKED:
            jc6                 = JointConstraint()
            jc6.joint_name      = "joint6"
            jc6.position        = _deg2rad(JOINT6_FIXED_DEG)
            jc6.tolerance_above = _deg2rad(JOINT6_TOLERANCE)
            jc6.tolerance_below = _deg2rad(JOINT6_TOLERANCE)
            jc6.weight          = 1.0
            locked.append(jc6)
        if locked:
            lock_c = Constraints()
            lock_c.joint_constraints = locked
            req.ik_request.constraints = lock_c

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

        # Normalisation [-π, π] : élimine les tours superflus (ex: 270° → -90°)
        positions = [_normalize(p) for p in positions]

        # Détection de grand saut par rapport au seed
        seed_ref = seed if seed is not None else self._current_joints
        jump = sum(abs(_normalize(a - b)) for a, b in zip(positions, seed_ref))
        if jump > JUMP_WARN_RAD:
            self.get_logger().warn(
                f'  ⚠ {label} : saut articulaire total {jump:.2f} rad '
                f'— trajectoire potentiellement longue'
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
                f'    Causes possibles : coordonnées trop éloignées, contrainte joint4/joint6\n'
                f'    trop stricte, ou orientation GRIPPER_DOWN_Q incompatible avec cette pose.\n'
                f'    Essayez de réduire JOINT4_LOCKED/JOINT6_LOCKED ou d\'ajuster les coords.'
            )
            return None

        self.get_logger().info(
            f'  IK {label} → {[f"{math.degrees(p):.1f}°" for p in positions]}'
        )
        return positions

    # ── Goal articulaire ──────────────────────────────────────────

    def _joint_goal(self, positions, with_path_constraints=False) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 5.0
        req.max_velocity_scaling_factor     = VELOCITY_SCALE
        req.max_acceleration_scaling_factor = ACCELERATION_SCALE

        # Goal : position cible exacte
        c = Constraints()
        for name, pos in zip(JOINT_NAMES, positions):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]

        # Path constraints : restreint la plage de chaque joint pendant la trajectoire.
        # Activé uniquement pour les moves issus de l'IK (pas HOME qui a joint4=-175°
        # hors des plages standard).
        if with_path_constraints:
            pc = Constraints()
            for name in JOINT_NAMES:
                lo_rad = _deg2rad(JOINT_LIMITS_DEG[name][0])
                hi_rad = _deg2rad(JOINT_LIMITS_DEG[name][1])
                mid    = (lo_rad + hi_rad) / 2.0
                half   = (hi_rad - lo_rad) / 2.0
                jc             = JointConstraint()
                jc.joint_name  = name
                jc.position    = mid
                jc.tolerance_below = half
                jc.tolerance_above = half
                jc.weight      = 1.0
                pc.joint_constraints.append(jc)
            req.path_constraints = pc

        goal.planning_options.plan_only = False
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

    def _send(self, goal, label: str) -> bool:
        if not self._securite_ok:
            self.get_logger().error(f'  SÉCURITÉ active — "{label}" bloqué.')
            return False

        self.get_logger().info(f"  → {label}")
        gh = self._wait(self._mg.send_goal_async(goal), timeout=12.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ Goal rejeté : {label}")
            return False

        result_future = gh.get_result_async()
        deadline = time.time() + 60.0
        while not result_future.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if not self._securite_ok:
                self.get_logger().error(
                    f'  ARRÊT D\'URGENCE pendant "{label}" — annulation goal.'
                )
                gh.cancel_goal_async()
                return False

        if not result_future.done():
            self.get_logger().warn(f"  ✗ Timeout : {label}")
            return False
        if result_future.result().result.error_code.val == 1:
            self.get_logger().info(f"  ✓ {label}")
            return True
        self.get_logger().warn(
            f"  ✗ {label} — code MoveIt {result_future.result().result.error_code.val}"
        )
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

    # ── Cycle principal ───────────────────────────────────────────

    def run(self):
        log = self.get_logger()

        log.info("Attente du serveur /move_group...")
        if not self._mg.wait_for_server(timeout_sec=15.0):
            log.error("/move_group non disponible — lancez demo.launch.py d'abord.")
            return

        log.info("Attente du service /compute_ik...")
        if not self._ik_cli.wait_for_service(timeout_sec=10.0):
            log.error("/compute_ik non disponible.")
            return

        # ── Flush messages entrants (état sécurité réel) ──────────
        # Sans ce spin, _securite_ok reste à True même si le nœud
        # sécurité diffuse déjà False au moment du démarrage.
        log.info("Synchronisation état sécurité...")
        for _ in range(30):          # ~1.5 s à 20 ms / iteration
            rclpy.spin_once(self, timeout_sec=0.05)
        if not self._securite_ok:
            log.error(
                "Le nœud sécurité bloque le mouvement au démarrage. "
                "Vérifiez que le robot est en position sûre (Z ≥ 0.19 m) "
                "et relancez — ou tapez 'go' dans le terminal du nœud sécurité."
            )
            return

        # ── Validation Z ──────────────────────────────────────────
        for label, z in [
            ("PICK",           PICK_Z),
            ("APPROCHE_PICK",  PICK_Z  + APPROACH_OFFSET),
            ("LIFT_PICK",      PICK_Z  + LIFT_OFFSET),
            ("APPROCHE_PLACE", PLACE_Z + APPROACH_OFFSET),
            ("PLACE",          PLACE_Z),
            ("LIFT_PLACE",     PLACE_Z + LIFT_OFFSET),
        ]:
            if not self._z_safe(z, label):
                log.error("Corrigez les coordonnées avant de relancer."); return

        # ── Pré-calcul IK enchaîné (chaque seed = solution précédente) ──
        log.info("Calcul IK de toutes les étapes (pince vers le bas)...")
        sequence = [
            ("APPROCHE_PICK",  PICK_X,  PICK_Y,  PICK_Z  + APPROACH_OFFSET),
            ("PICK",           PICK_X,  PICK_Y,  PICK_Z),
            ("LIFT_PICK",      PICK_X,  PICK_Y,  PICK_Z  + LIFT_OFFSET),
            ("APPROCHE_PLACE", PLACE_X, PLACE_Y, PLACE_Z + APPROACH_OFFSET),
            ("PLACE",          PLACE_X, PLACE_Y, PLACE_Z),
            ("LIFT_PLACE",     PLACE_X, PLACE_Y, PLACE_Z + LIFT_OFFSET),
        ]
        joints = {}
        seed = list(HOME_JOINTS)   # point de départ : HOME
        for label, x, y, z in sequence:
            j = self._ik_to_joints(x, y, z, label, seed=seed)
            if j is None:
                log.error(f"IK impossible pour {label} — vérifiez les coordonnées."); return
            joints[label] = j
            seed = j   # la solution devient le seed de l'étape suivante
        log.info("✅ IK calculée — trajectoires optimisées.")

        log.info("=" * 60)
        log.info(f"  PICK  A = ({PICK_X:.3f}, {PICK_Y:.3f}, {PICK_Z:.3f}) m")
        log.info(f"  PLACE B = ({PLACE_X:.3f}, {PLACE_Y:.3f}, {PLACE_Z:.3f}) m")
        log.info(f"  Plancher sécurité Z ≥ {Z_MIN_M} m")
        log.info("=" * 60)

        # 1. HOME
        log.info("[1] Aller à HOME")
        if not self._send(self._joint_goal(HOME_JOINTS), "HOME"):
            log.error("HOME échoué — arrêt."); return

        # 2. Ouvrir pince
        log.info("[2] Ouverture pince")
        self._gripper(close=False)

        # 3. Approche Pick
        log.info(f"[3] Approche Pick (A + {APPROACH_OFFSET*100:.0f} cm)")
        if not self._send(self._joint_goal(joints["APPROCHE_PICK"]), "APPROCHE_PICK"):
            log.error("Approche Pick échouée — arrêt."); return

        # 4. Descente sur A
        log.info("[4] Descente sur A (PICK)")
        if not self._send(self._joint_goal(joints["PICK"]), "PICK"):
            log.error("Pick échoué — arrêt."); return

        # 5. Fermer pince
        log.info("[5] Saisie de l'objet")
        self._gripper(close=True)

        # 6. Levée
        log.info(f"[6] Levée ({LIFT_OFFSET*100:.0f} cm)")
        self._send(self._joint_goal(joints["LIFT_PICK"]), "LIFT_PICK")

        # 7. Approche Place
        log.info(f"[7] Approche Place (B + {APPROACH_OFFSET*100:.0f} cm)")
        if not self._send(self._joint_goal(joints["APPROCHE_PLACE"]), "APPROCHE_PLACE"):
            log.error("Approche Place échouée — arrêt."); return

        # 8. Descente sur B
        log.info("[8] Descente sur B (PLACE)")
        if not self._send(self._joint_goal(joints["PLACE"]), "PLACE"):
            log.error("Place échouée — arrêt."); return

        # 9. Ouvrir pince
        log.info("[9] Lâcher de l'objet")
        self._gripper(close=False)

        # 10. Remontée et HOME
        log.info("[10] Remontée et retour HOME")
        self._send(self._joint_goal(joints["LIFT_PLACE"]), "LIFT_PLACE")
        self._send(self._joint_goal(HOME_JOINTS), "HOME")

        log.info("=" * 60)
        log.info("  CYCLE TERMINÉ AVEC SUCCÈS ✅")
        log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = PickPlaceIA()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
