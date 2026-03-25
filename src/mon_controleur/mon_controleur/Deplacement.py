#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class Deplacement(Node):
    def __init__(self):
        super().__init__('deplacement_du_robot')
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Publishing joint states...")
        self.q0 = [0.7, 0.5, -0.6, 0.0, 0.9, 0.0]
        self.q1 = [0.1, 0.0, 0.5, 0.0, 0.0, 0.7]
        self.t = 0.0
        self.duration = 5.0

    def timer_callback(self):
        self.t += 0.1
        alpha = min(self.t / self.duration, 1.0)
        q = [(1 - alpha) * a + alpha * b for a, b in zip(self.q0, self.q1)]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = q
        self.js_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Deplacement()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
