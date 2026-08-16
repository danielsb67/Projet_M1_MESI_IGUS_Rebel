#!/usr/bin/env python3
"""Capture une frame de /front_camera/image (sensor_msgs/Image) -> PNG, puis quitte."""
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/front.png"
TOPIC = sys.argv[2] if len(sys.argv) > 2 else "/front_camera/image"


class Grab(Node):
    def __init__(self):
        super().__init__("grab_front")
        self.bridge = CvBridge()
        self.done = False
        self.create_subscription(Image, TOPIC, self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        if self.done:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(OUT, img)
        self.get_logger().info(f"saved {OUT} ({msg.width}x{msg.height}, enc={msg.encoding})")
        self.done = True


def main():
    rclpy.init()
    node = Grab()
    end = node.get_clock().now().nanoseconds + int(15e9)
    while rclpy.ok() and not node.done and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    ok = node.done
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
