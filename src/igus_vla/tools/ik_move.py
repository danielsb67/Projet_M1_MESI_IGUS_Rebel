#!/usr/bin/env python3
"""Calcule l'IK d'une pose de pick (orientation verticale) via /compute_ik,
puis envoie la solution articulaire au contrôleur. Reproduit l'approche expert."""
import sys
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

BASE_FRAME = "igus_rebel_base_link"
ARM_GROUP = "rebel_arm"
DOWNWARD_QUAT = (0.0, -0.70710678, 0.0, 0.70710678)
X = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
Y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
Z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05


class IKMove(Node):
    def __init__(self):
        super().__init__("ik_move")
        self.cli = self.create_client(GetPositionIK, "/compute_ik")
        self.pub = self.create_publisher(
            JointTrajectory, "/rebel_arm_controller/joint_trajectory", 10)

    def run(self):
        if not self.cli.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("/compute_ik indisponible"); return False
        req = GetPositionIK.Request()
        req.ik_request.group_name = ARM_GROUP
        req.ik_request.avoid_collisions = False
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = X, Y, Z
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = DOWNWARD_QUAT
        req.ik_request.pose_stamped = ps
        req.ik_request.timeout = Duration(sec=2)
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            self.get_logger().error(f"IK échec (code={getattr(res,'error_code',None)})")
            return False
        js = res.solution.joint_state
        jmap = dict(zip(js.name, js.position))
        order = [f"joint{i}" for i in range(1, 7)]
        target = [jmap[j] for j in order if j in jmap]
        self.get_logger().info(f"IK OK -> {[round(v,3) for v in target]}")
        msg = JointTrajectory()
        msg.joint_names = order
        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.time_from_start = Duration(sec=3)
        msg.points = [pt]
        for _ in range(15):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        end = self.get_clock().now().nanoseconds + int(5e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.2)
        return True


def main():
    rclpy.init()
    n = IKMove()
    ok = n.run()
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
