#!/usr/bin/env python3
import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import tf2_ros

ERREUR_CALIBRATION_Z = -0.01
Z_MARGE_M  = -0.003
Z_STOP_M   = Z_MARGE_M         + ERREUR_CALIBRATION_Z
Z_RESUME_M = Z_MARGE_M + 0.015 + ERREUR_CALIBRATION_Z
Z_MIN_M    = Z_STOP_M

Z_TIMER_HZ       = 100
Z_DEBOUNCE_COUNT = 2
Z_GRACE_PERIOD_S = 2.0

# Délai avant d'appliquer les arrêts sur limites articulaires (laisser pick_place_ia
# envoyer la commande HOME depuis une configuration hors-limites au démarrage)
JOINT_GRACE_PERIOD_S = 35.0

BASE_FRAME = "igus_rebel_base_link"
EE_FRAME   = "gripper_tip_link"

# ── Limites articulaires (alignées sur pick_place_ia.py / URDF) ──────────────
# Marge logicielle avant la butée physique : avertissement à SOFT_DEG, arrêt à 0°
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "joint1": (-179.0,  179.0),
    "joint2": ( -80.0,  140.0),
    "joint3": ( -80.0,  140.0),
    "joint4": (-179.0,  179.0),
    "joint5": ( -90.0,   90.0),
    "joint6": (-179.0,  179.0),
}
JOINT_SOFT_MARGIN_DEG = 3.0   # zone d'alerte avant la butée
JOINT_LIMIT_DEBOUNCE  = 3     # lectures consécutives avant arrêt

# ── Détection blocage (stall) ─────────────────────────────────────────────────
# Si une vitesse est commandée mais la position ne bouge pas → obstacle / blocage
STALL_VEL_THRESHOLD  = 0.05   # rad/s min pour considérer une commande active
STALL_POS_THRESHOLD  = 0.003  # rad de déplacement minimum attendu sur la fenêtre
STALL_DEBOUNCE       = 15     # lectures consécutives (~300 ms à 50 Hz)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SecuriteRobot(Node):
    def __init__(self):
        super().__init__("securite_des_robots")

        self._mouvement_autorise = True
        self._lock = threading.Lock()

        self.joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_vel_max = [0.79] * len(self.joints)

        self.last_cmd_vel  = [0.0] * len(self.joints)
        self.last_cmd_time = self.get_clock().now()

        self.cmd_publisher  = self.create_publisher(JointState, '/joint_vel_cmd_safe', 10)
        self._pub_securite  = self.create_publisher(Bool, '/securite_mouvement', 10)
        self.create_timer(0.1, self._publish_securite)

        self.cmd_subscription = self.create_subscription(
            JointState, '/joint_vel_cmd_raw', self.cmd_callback, 10
        )

        self.timeout_s = 0.3
        self.timer = self.create_timer(0.02, self.timer_callback)

        # --- TF surveillance Z ---
        self._tf_buffer        = tf2_ros.Buffer()
        self._tf_listener      = tf2_ros.TransformListener(self._tf_buffer, self)
        self._z_bloque         = False
        self._z_sous_seuil_cpt = 0
        self._startup_time     = self.get_clock().now()
        self.create_timer(1.0 / Z_TIMER_HZ, self._verifier_z)

        # --- Surveillance limites articulaires ---
        self._limit_bloque    = False
        self._limit_cpt: dict[str, int] = {j: 0 for j in self.joints}

        # --- Détection stall ---
        self._stall_bloque    = False
        self._stall_cpt       = 0
        self._last_positions: dict[str, float] = {}

        self.create_subscription(
            JointState, '/joint_states', self._verifier_positions, 10
        )

        self._input_thread = threading.Thread(target=self._lire_commandes, daemon=True)
        self._input_thread.start()

        self.get_logger().info(
            f"✓ Sécurité démarrée : raw → safe | clavier : 'stop' / 'go' | "
            f"plancher Z ≥ {Z_STOP_M:.3f} m | limites articulaires + stall actifs"
        )

    # ------------------------------------------------------------------
    def _verifier_z(self):
        elapsed = (self.get_clock().now() - self._startup_time).nanoseconds * 1e-9
        if elapsed < Z_GRACE_PERIOD_S:
            return
        try:
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
            z  = tf.transform.translation.z
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        with self._lock:
            z_r = round(z, 4)
            if z_r < Z_STOP_M:
                self._z_sous_seuil_cpt += 1
                if self._z_sous_seuil_cpt >= Z_DEBOUNCE_COUNT and not self._z_bloque:
                    self._z_bloque = True
                    self._mouvement_autorise = False
                    self.get_logger().error(
                        f'✗ SÉCURITÉ Z : {z:.4f} m < {Z_STOP_M:.4f} m — ARRÊT !'
                    )
            else:
                self._z_sous_seuil_cpt = 0
                if z_r >= Z_RESUME_M and self._z_bloque:
                    self._z_bloque = False
                    self._mouvement_autorise = True
                    self.get_logger().info(
                        f'✓ SÉCURITÉ Z : {z:.4f} m ≥ {Z_RESUME_M:.4f} m — mouvement réautorisé'
                    )

    # ------------------------------------------------------------------
    def _verifier_positions(self, msg: JointState):
        if not msg.name or not msg.position:
            return

        pos_map = dict(zip(msg.name, msg.position))
        elapsed = (self.get_clock().now() - self._startup_time).nanoseconds * 1e-9
        in_grace = elapsed < JOINT_GRACE_PERIOD_S

        with self._lock:
            # ── Limites articulaires ──────────────────────────────────
            violation_dure = None
            for joint, pos in pos_map.items():
                if joint not in JOINT_LIMITS_DEG:
                    continue
                lo_deg, hi_deg = JOINT_LIMITS_DEG[joint]
                pos_deg = math.degrees(pos)
                soft_lo = lo_deg + JOINT_SOFT_MARGIN_DEG
                soft_hi = hi_deg - JOINT_SOFT_MARGIN_DEG

                if pos_deg <= lo_deg or pos_deg >= hi_deg:
                    if in_grace:
                        # Pendant le grace period : avertissement seulement,
                        # pick_place_ia a le temps d'envoyer la commande HOME
                        self.get_logger().warn(
                            f'⚠ [GRACE] {joint} hors limites : {pos_deg:.1f}° '
                            f'[{lo_deg:.0f}°, {hi_deg:.0f}°] — HOME en cours...'
                        )
                    else:
                        self._limit_cpt[joint] += 1
                        if self._limit_cpt[joint] >= JOINT_LIMIT_DEBOUNCE and not self._limit_bloque:
                            violation_dure = (joint, pos_deg, lo_deg, hi_deg)
                elif pos_deg < soft_lo or pos_deg > soft_hi:
                    self._limit_cpt[joint] = 0
                    if not in_grace:
                        self.get_logger().warn(
                            f'⚠ {joint} proche butée : {pos_deg:.1f}° '
                            f'(limites [{lo_deg:.0f}°, {hi_deg:.0f}°])'
                        )
                else:
                    self._limit_cpt[joint] = 0

            if violation_dure:
                j, p, lo, hi = violation_dure
                self._limit_bloque = True
                self._mouvement_autorise = False
                self.get_logger().error(
                    f'✗ LIMITE ARTICULAIRE : {j} = {p:.1f}° hors [{lo:.0f}°, {hi:.0f}°] — ARRÊT !'
                )

            # ── Détection stall ───────────────────────────────────────
            if self._last_positions:
                stall_joints = []
                for i, joint in enumerate(self.joints):
                    cmd_v = self.last_cmd_vel[i]
                    if abs(cmd_v) < STALL_VEL_THRESHOLD:
                        continue  # pas de commande active sur ce joint
                    prev = self._last_positions.get(joint)
                    curr = pos_map.get(joint)
                    if prev is None or curr is None:
                        continue
                    if abs(curr - prev) < STALL_POS_THRESHOLD:
                        stall_joints.append(joint)

                if stall_joints:
                    self._stall_cpt += 1
                    if self._stall_cpt >= STALL_DEBOUNCE and not self._stall_bloque:
                        self._stall_bloque = True
                        self._mouvement_autorise = False
                        self.get_logger().error(
                            f'✗ STALL détecté sur {stall_joints} '
                            f'— robot bloqué ! Tapez "go" pour reprendre.'
                        )
                else:
                    self._stall_cpt = 0

            self._last_positions = pos_map

    # ------------------------------------------------------------------
    def _publish_securite(self):
        msg = Bool()
        with self._lock:
            msg.data = self._mouvement_autorise
        self._pub_securite.publish(msg)

    def _lire_commandes(self):
        while True:
            try:
                commande = input().strip().lower()
            except EOFError:
                break

            with self._lock:
                if commande == 'stop':
                    self._mouvement_autorise = False
                    self.get_logger().warn('⚠ SÉCURITÉ : mouvement bloqué (stop)')
                elif commande == 'go':
                    self._mouvement_autorise = True
                    self._z_bloque           = False
                    self._z_sous_seuil_cpt   = 0
                    self._limit_bloque       = False
                    self._limit_cpt          = {j: 0 for j in self.joints}
                    self._stall_bloque       = False
                    self._stall_cpt          = 0
                    self.get_logger().info('✓ SÉCURITÉ : mouvement autorisé (go)')
                else:
                    self.get_logger().warn(
                        f"⚠ Commande inconnue : '{commande}' — utilisez 'stop' ou 'go'"
                    )

    # ------------------------------------------------------------------
    def cmd_callback(self, msg: JointState):
        self.last_cmd_time = self.get_clock().now()

        if len(msg.velocity) >= len(self.joints):
            self.last_cmd_vel = list(msg.velocity[:len(self.joints)])
            return

        if msg.name and msg.velocity and len(msg.name) == len(msg.velocity):
            name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity)}
            self.last_cmd_vel = [float(name_to_vel.get(j, 0.0)) for j in self.joints]
            return

        self.get_logger().warn('⚠ JointState sans vitesses exploitables — zéros forcés')
        self.last_cmd_vel = [0.0] * len(self.joints)

    def timer_callback(self):
        now = self.get_clock().now()

        with self._lock:
            autorise = self._mouvement_autorise

        stale = (now - self.last_cmd_time) > Duration(seconds=self.timeout_s)
        if not autorise or stale:
            safe_vel = [0.0] * len(self.joints)
        else:
            safe_vel = []
            for i, v in enumerate(self.last_cmd_vel):
                vmax  = self.joint_vel_max[i]
                v_safe = clamp(v, -vmax, vmax)
                if v_safe != v:
                    self.get_logger().warn(
                        f'⚠ {self.joints[i]} : v={v:.3f} → {v_safe:.3f} rad/s (limite ±{vmax:.3f})'
                    )
                safe_vel.append(v_safe)

        out = JointState()
        out.header.stamp = now.to_msg()
        out.name         = self.joints
        out.velocity     = safe_vel
        self.cmd_publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SecuriteRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
