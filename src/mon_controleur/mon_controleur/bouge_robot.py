#!/usr/bin/env python3
"""
Script pour faire bouger le bras Igus Rebel RÉEL.
Mouvement petit et sécurisé à partir de la position actuelle.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import time

CONTROLLER_NAME = "rebel_arm_controller"

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

# Durée longue = mouvement lent = sécurisé
MOVE_DURATION = 8.0


class IgusRebelMover(Node):
    def __init__(self):
        super().__init__("igus_rebel_mover")
        action_topic = f"/{CONTROLLER_NAME}/follow_joint_trajectory"
        self.get_logger().info(f"Connexion à : {action_topic}")
        self._action_client = ActionClient(self, FollowJointTrajectory, action_topic)
        self._current_positions = None
        self._joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )

    def _joint_state_callback(self, msg):
        self._current_positions = dict(zip(msg.name, msg.position))

    def wait_for_server(self, timeout_sec=15.0):
        self.get_logger().info("En attente du serveur d'action...")
        if not self._action_client.wait_for_server(timeout_sec):
            self.get_logger().error("Serveur non disponible !")
            return False
        self.get_logger().info("Serveur connecté !")
        return True

    def send_trajectory(self, target_positions, duration_sec=8.0):
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9),
        )
        trajectory.points.append(point)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.get_logger().info(
            f"Envoi vers : {[f'{p:.3f}' for p in target_positions]} "
            f"(durée : {duration_sec}s)"
        )

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal REJETÉ !")
            return False

        self.get_logger().info("Goal accepté, mouvement en cours...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Mouvement terminé !")
        return True

    def get_current_positions(self):
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._current_positions:
                break
        if self._current_positions:
            positions = []
            for name in JOINT_NAMES:
                if name in self._current_positions:
                    positions.append(self._current_positions[name])
                else:
                    self.get_logger().warn(f"Joint '{name}' non trouvé !")
                    return None
            return positions
        return None


def main():
    rclpy.init()
    node = IgusRebelMover()

    try:
        if not node.wait_for_server(timeout_sec=15.0):
            return

        # Lire la position actuelle du robot
        current = node.get_current_positions()
        if not current:
            node.get_logger().error("Impossible de lire la position actuelle !")
            return

        node.get_logger().info(f"Position actuelle : {[f'{p:.3f}' for p in current]}")

        # Position B = position actuelle + petit mouvement sur joint1 seulement (0.15 rad ~ 9 degrés)
        position_b = list(current)
        position_b[0] = current[0] + 0.15  # joint1 tourne un peu

        node.get_logger().info(f"Position cible  :  {[f'{p:.3f}' for p in position_b]}")
        node.get_logger().info("Seul joint1 va bouger de ~9 degrés")

        input("\n>>> APPUIE SUR ENTRÉE pour bouger joint1 de +0.15 rad...")
        node.send_trajectory(position_b, duration_sec=MOVE_DURATION)
        time.sleep(2.0)

        input("\n>>> APPUIE SUR ENTRÉE pour revenir à la position initiale...")
        node.send_trajectory(current, duration_sec=MOVE_DURATION)

        node.get_logger().info("Séquence terminée !")

    except KeyboardInterrupt:
        node.get_logger().info("Interrompu.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
