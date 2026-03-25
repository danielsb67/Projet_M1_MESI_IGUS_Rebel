#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RVizJointStateCmd(Node):
    def __init__(self):
        super().__init__('rviz_jointstate_cmd')

        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        self.joints = ['joint1','joint2','joint3','joint4','joint5','joint6']
        self.pos = [2.0, 0.8, 0.8, -1.9, 0.7, 0.3]  # radians

        # publie en continu pour que RViz reste à jour
        self.timer = self.create_timer(0.05, self.cb)  # 20 Hz
        self.get_logger().info("Publishing JointState on /joint_states for RViz.")

    def cb(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joints
        msg.position = self.pos
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RVizJointStateCmd()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

