#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class SecuriteTrajectoire(Node):
    def __init__(self):
        super().__init__('securite_trajectoire')

        # ---- Joints (doivent matcher le contrôleur/URDF) ----
        self.joints = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']

        # ---- Limites positions (rad) ----
        self.pos_limits = {
            'joint_1': (-3.124,  3.124),
            'joint_2': (-1.553,  2.444),
            'joint_3': (-1.553,  2.444),
            'joint_4': (-3.124,  3.124),
            'joint_5': (-1.658,  1.658),
            'joint_6': (-3.124,  3.124),
        }

        # ---- Limites vitesses (rad/s) ----
        self.vmax = {j: 0.79 for j in self.joints}

        # ---- Position actuelle (depuis /joint_states) ----
        self.q_current = {j: None for j in self.joints}

        self.sub_js = self.create_subscription(JointState, '/joint_states', self.cb_joint_states, 10)

        # Input raw trajectory
        self.sub_raw = self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_raw',
            self.cb_raw_traj,
            10
        )

        # Output safe trajectory to controller
        self.pub_safe = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.get_logger().info("Trajectory safety gate ON: /joint_trajectory_raw -> controller")

    def cb_joint_states(self, msg: JointState):
        # map name->position
        if not msg.name or not msg.position:
            return
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        for j in self.joints:
            if j in name_to_pos:
                self.q_current[j] = float(name_to_pos[j])

    def _time_from_start_to_sec(self, t):
        return float(t.sec) + float(t.nanosec) * 1e-9

    def _sec_to_time_from_start(self, sec_float):
        sec = int(math.floor(sec_float))
        nanosec = int((sec_float - sec) * 1e9)
        d = Duration(seconds=0.0).to_msg()
        d.sec = sec
        d.nanosec = nanosec
        return d

    def cb_raw_traj(self, traj: JointTrajectory):
        if not traj.points:
            self.get_logger().warn("Received empty trajectory.")
            return

        # Vérifie joint_names
        if not traj.joint_names:
            self.get_logger().error("Trajectory has no joint_names -> rejected.")
            return

        # On impose l'ordre canonique self.joints
        name_to_idx = {n: i for i, n in enumerate(traj.joint_names)}
        missing = [j for j in self.joints if j not in name_to_idx]
        if missing:
            self.get_logger().error(f"Missing joints in trajectory: {missing} -> rejected.")
            return

        # Point de départ pour calcul vitesse implicite
        # Si on n'a pas joint_states, on ne peut pas calculer un T_safe parfaitement
        have_current = all(self.q_current[j] is not None for j in self.joints)

        safe = JointTrajectory()
        safe.header.stamp = self.get_clock().now().to_msg()
        safe.joint_names = list(self.joints)

        last_time = 0.0
        q_prev = {j: (self.q_current[j] if have_current else 0.0) for j in self.joints}

        for k, pt in enumerate(traj.points):
            if len(pt.positions) < len(traj.joint_names):
                self.get_logger().error("Point positions size mismatch -> rejected.")
                return

            # 1) Reorder + clamp positions
            q_des = {}
            for j in self.joints:
                raw_q = float(pt.positions[name_to_idx[j]])
                lo, hi = self.pos_limits[j]
                q_des[j] = clamp(raw_q, lo, hi)
                if q_des[j] != raw_q:
                    self.get_logger().warn(f"{j}: pos {raw_q:.3f} clamped to {q_des[j]:.3f}")

            # 2) Time_from_start safe (respect vmax) + monotonic
            t_raw = self._time_from_start_to_sec(pt.time_from_start)
            if t_raw <= last_time:
                t_raw = last_time + 0.1  # impose croissant

            # calc T required from q_prev -> q_des
            if have_current:
                dt_needed = 0.0
                for j in self.joints:
                    dq = abs(q_des[j] - q_prev[j])
                    dtj = dq / max(self.vmax[j], 1e-6)
                    dt_needed = max(dt_needed, dtj)
                # dt minimal pour ce segment
                dt_seg = t_raw - last_time
                if dt_seg < dt_needed:
                    t_raw = last_time + dt_needed
                    self.get_logger().warn(
                        f"Point {k}: time increased to {t_raw:.2f}s to respect vmax"
                    )

            # 3) Build safe point
            spt = JointTrajectoryPoint()
            spt.positions = [q_des[j] for j in self.joints]
            spt.time_from_start = self._sec_to_time_from_start(t_raw)

            safe.points.append(spt)

            # update
            last_time = t_raw
            q_prev = q_des

        self.pub_safe.publish(safe)
        self.get_logger().info("Safe trajectory published.")
        

def main(args=None):
    rclpy.init(args=args)
    node = SecuriteTrajectoire()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

