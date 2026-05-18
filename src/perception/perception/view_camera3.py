#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2


class VisionYOLO(Node):

    def __init__(self):
        super().__init__('vision_yolo_node')

        self.bridge = CvBridge()
        self.model = YOLO('/home/dbal/projet_igus/src/yolo/yolo/best.pt')

        self.color_image = None
        self.depth_image = None

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        self.pub_point = self.create_publisher(
            PointStamped,
            '/vision/roulette_point',
            10
        )

        self.pub_result = self.create_publisher(
            String,
            '/vision/roulette_result',
            10
        )

        self.pub_debug_image = self.create_publisher(
            Image,
            '/vision/debug_image',
            10
        )

        self.timer = self.create_timer(0.1, self.process_image)

        self.get_logger().info(
            "YOLO + Depth node started without cv2.imshow"
        )

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

    def process_image(self):

        if self.color_image is None or self.depth_image is None:
            return

        frame = self.color_image.copy()

        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        if len(results[0].boxes) == 0:
            result_msg = String()
            result_msg.data = "detected=False"
            self.pub_result.publish(result_msg)

            self.publish_debug_image(frame)
            return

        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if cy < 0 or cy >= self.depth_image.shape[0]:
            return

        if cx < 0 or cx >= self.depth_image.shape[1]:
            return

       

        roi_size = 10

        x_min = max(cx - roi_size, 0)
        x_max = min(cx + roi_size, self.depth_image.shape[1])
        y_min = max(cy - roi_size, 0)
        y_max = min(cy + roi_size, self.depth_image.shape[0])

        depth_roi = self.depth_image[y_min:y_max, x_min:x_max]

        valid_depths = depth_roi[depth_roi > 0]

        if valid_depths.size == 0:
            self.get_logger().warn("Depth invalide")
            self.publish_debug_image(frame)
            return

        depth_m = float(valid_depths.mean()) / 1000.0

        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "camera_color_frame"

        point_msg.point.x = float(cx)
        point_msg.point.y = float(cy)
        point_msg.point.z = depth_m

        self.pub_point.publish(point_msg)

        result_msg = String()
        result_msg.data = (
            f"detected=True, "
            f"cx={cx}, cy={cy}, depth={depth_m:.3f}"
        )
        self.pub_result.publish(result_msg)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"depth={depth_m:.2f} m",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        self.publish_debug_image(frame)

    def publish_debug_image(self, frame):
        img_msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_color_frame"

        self.pub_debug_image.publish(img_msg)


def main(args=None):

    rclpy.init(args=args)
    node = VisionYOLO()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
