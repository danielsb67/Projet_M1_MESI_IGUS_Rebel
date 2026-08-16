#!/usr/bin/env python3
"""Publie une trajectoire articulaire vers une pose 'reach forward' et attend."""
import sys
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

TARGET = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                             else "0.3,1.0,0.6,0.0,1.0,0.0".split(","))]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0


class Mover(Node):
    def __init__(self):
        super().__init__("mover")
        self.pub = self.create_publisher(
            JointTrajectory, "/rebel_arm_controller/joint_trajectory", 10)

    def send(self):
        msg = JointTrajectory()
        msg.joint_names = [f"joint{i}" for i in range(1, 7)]
        pt = JointTrajectoryPoint()
        pt.positions = TARGET
        pt.time_from_start = Duration(sec=int(DUR), nanosec=int((DUR % 1) * 1e9))
        msg.points = [pt]
        # publie quelques fois (le controleur doit etre abonné)
        for _ in range(10):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"trajectoire envoyée vers {TARGET} (durée {DUR}s)")


def main():
    rclpy.init()
    n = Mover()
    n.send()
    end = n.get_clock().now().nanoseconds + int((DUR + 1.5) * 1e9)
    while rclpy.ok() and n.get_clock().now().nanoseconds < end:
        rclpy.spin_once(n, timeout_sec=0.2)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
