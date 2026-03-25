#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SecuriteRobot(Node):
    def __init__(self):
        super().__init__("securite_des_robots")

        # --- Config joints ---
        self.joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_vel_max = [0.79] * len(self.joints)

        # --- Last command ---
        self.last_cmd_vel = [0.0] * len(self.joints)
        self.last_cmd_time = self.get_clock().now()

        # --- Publisher safe ---
        self.cmd_publisher = self.create_publisher(JointState, '/joint_vel_cmd_safe', 10)

        # --- Subscriber raw ---
        self.cmd_subscription = self.create_subscription(
            JointState,
            '/joint_vel_cmd_raw',        # <-- topic d'entrée (raw)
            self.cmd_callback,
            10
        )

        # --- Watchdog timeout ---
        self.timeout_s = 0.3  # si pas de commande depuis 0.3s => stop

        # --- Timer publish (50 Hz) ---
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.get_logger().info("Safety node started: gating /joint_vel_cmd_raw -> /joint_vel_cmd_safe")

    def cmd_callback(self, msg: JointState):
        # Met à jour le timestamp
        self.last_cmd_time = self.get_clock().now()

        # Cas simple: msg.velocity contient directement 6 valeurs dans l'ordre
        if len(msg.velocity) >= len(self.joints):
            self.last_cmd_vel = list(msg.velocity[:len(self.joints)])
            return

        # Sinon, si msg.name est fourni, on remappe proprement
        if msg.name and msg.velocity and len(msg.name) == len(msg.velocity):
            name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity)}
            self.last_cmd_vel = [float(name_to_vel.get(j, 0.0)) for j in self.joints]
            return

        # Si message invalide
        self.get_logger().warn("Received JointState without usable velocities; forcing zeros.")
        self.last_cmd_vel = [0.0] * len(self.joints)

    def timer_callback(self):
        now = self.get_clock().now()

        # Watchdog: si commande trop vieille -> stop
        stale = (now - self.last_cmd_time) > Duration(seconds=self.timeout_s)
        if stale:
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

