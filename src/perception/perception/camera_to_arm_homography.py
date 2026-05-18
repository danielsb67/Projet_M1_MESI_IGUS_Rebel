#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

import numpy as np
import cv2


class CameraToArmHomography(Node):

    def __init__(self):
        super().__init__('camera_to_arm_homography')

        self.sub = self.create_subscription(
            PointStamped,
            '/vision/roulette_point',
            self.point_callback,
            10
        )

        self.pub = self.create_publisher(
            PointStamped,
            '/arm/roulette_point',
            10
        )

        camera_points = np.array([
            [63, 429],     # A
            [576, 426],    # B
            [592, 45],     # C
            [47, 39]       # D
        ], dtype=np.float32)

        arm_points = np.array([
            [0.27, 0.52],     # A
            [-0.10, 0.54],    # B
            [-0.18, 0.24],    # C
            [0.26, 0.23]      # D
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(
            camera_points,
            arm_points
        )

        self.get_logger().info(
            "Camera -> Arm Homography started with filter"
        )

    def point_callback(self, msg):

        cx = msg.point.x
        cy = msg.point.y

        pixel_point = np.array(
            [[[cx, cy]]],
            dtype=np.float32
        )

        arm_point = cv2.perspectiveTransform(
            pixel_point,
            self.H
        )

        y_arm = float(arm_point[0][0][0])
        x_arm = float(arm_point[0][0][1])

        # ==================================================
        # Filtre simple
        # ==================================================
        if not (0.20 <= x_arm <= 0.30 and 0.20 <= y_arm <= 0.27):
            self.get_logger().warn(
                f"Point ignoré -> X={x_arm:.3f} m, Y={y_arm:.3f} m"
            )
            return

        z_arm = 0.018

        out = PointStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "arm_base"

        out.point.x = x_arm
        out.point.y = y_arm
        out.point.z = z_arm

        self.pub.publish(out)

        self.get_logger().info(
            f"Arm point publié -> "
            f"X={x_arm:.3f} m, "
            f"Y={y_arm:.3f} m"
        )


def main(args=None):

    rclpy.init(args=args)

    node = CameraToArmHomography()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
