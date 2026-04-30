#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import tf2_ros

Z_STOP_M   = 0.004         # plancher bas  : en-dessous → arrêt d'urgence
Z_RESUME_M = 0.007         # plancher haut : au-dessus → re-autorisation (hysteresis 3 mm)
Z_MIN_M    = Z_STOP_M      # alias conservé pour compatibilité logs
BASE_FRAME = "igus_rebel_base_link"
EE_FRAME   = "gripper_tip_link"   # bout physique de la pince


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SecuriteRobot(Node):
    def __init__(self):
        super().__init__("securite_des_robots")

        # --- Etat de sécurité global (True = autorisé) ---
        self._mouvement_autorise = True
        self._lock = threading.Lock()

        # --- Config joints ---
        self.joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_vel_max = [0.79] * len(self.joints)

        # --- Last command ---
        self.last_cmd_vel = [0.0] * len(self.joints)
        self.last_cmd_time = self.get_clock().now()

        # --- Publisher safe vitesses ---
        self.cmd_publisher = self.create_publisher(JointState, '/joint_vel_cmd_safe', 10)

        # --- Publisher état sécurité (10 Hz) ---
        self._pub_securite = self.create_publisher(Bool, '/securite_mouvement', 10)
        self.create_timer(0.1, self._publish_securite)

        # --- Subscriber raw ---
        self.cmd_subscription = self.create_subscription(
            JointState,
            '/joint_vel_cmd_raw',
            self.cmd_callback,
            10
        )

        # --- Watchdog timeout ---
        self.timeout_s = 0.3

        # --- Timer publish vitesses (50 Hz) ---
        self.timer = self.create_timer(0.02, self.timer_callback)

        # --- TF : surveillance position Z de l'effecteur ---
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._z_bloque    = False
        self.create_timer(0.05, self._verifier_z)  # 20 Hz

        # --- Thread lecture clavier (stop / go) ---
        self._input_thread = threading.Thread(target=self._lire_commandes, daemon=True)
        self._input_thread.start()

        self.get_logger().info(
            "Safety node started: /joint_vel_cmd_raw -> /joint_vel_cmd_safe | "
            "Commandes clavier : 'stop' | 'go'"
        )

    # ------------------------------------------------------------------
    def _verifier_z(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_FRAME, rclpy.time.Time()
            )
            z = tf.transform.translation.z
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return  # TF pas encore disponible, on attend

        with self._lock:
            if z < Z_MIN_M and not self._z_bloque:
                self._z_bloque = True
                self._mouvement_autorise = False
                self.get_logger().error(
                    f'SECURITE : Z = {z:.3f} m < {Z_MIN_M} m — ARRÊT D\'URGENCE !'
                )
            elif z >= Z_MIN_M and self._z_bloque:
                self._z_bloque = False
                self._mouvement_autorise = True
                self.get_logger().info(
                    f'SECURITE : Z = {z:.3f} m — zone sûre, mouvement réautorisé.'
                )

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
                    self.get_logger().warn('SECURITE : Mouvement BLOQUE (stop reçu)')
                elif commande == 'go':
                    self._mouvement_autorise = True
                    self.get_logger().info('SECURITE : Mouvement AUTORISE (go reçu)')
                else:
                    self.get_logger().warn(
                        f"Commande inconnue : '{commande}'. Utilisez 'stop' ou 'go'."
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

        self.get_logger().warn("Received JointState without usable velocities; forcing zeros.")
        self.last_cmd_vel = [0.0] * len(self.joints)

    def timer_callback(self):
        now = self.get_clock().now()

        with self._lock:
            autorise = self._mouvement_autorise

        # Stop d'urgence clavier OU watchdog
        stale = (now - self.last_cmd_time) > Duration(seconds=self.timeout_s)
        if not autorise or stale:
            safe_vel = [0.0] * len(self.joints)
        else:
            safe_vel = []
            for i, v in enumerate(self.last_cmd_vel):
                vmax = self.joint_vel_max[i]
                v_safe = clamp(v, -vmax, vmax)
                if v_safe != v:
                    self.get_logger().warn(
                        f"Joint {self.joints[i]}: v={v:.3f} clamped to {v_safe:.3f} (limit={vmax:.3f})"
                    )
                safe_vel.append(v_safe)

        out = JointState()
        out.header.stamp = now.to_msg()
        out.name = self.joints
        out.velocity = safe_vel
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
