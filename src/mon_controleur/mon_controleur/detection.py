#!/usr/bin/env python3

import json
import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')

        # Parameters
        self.rgb_topic = self.declare_parameter(
            'rgb_topic',
            '/camera/color/image_raw'
        ).value

        self.model_path = self.declare_parameter(
            'model_path',
            'yolo11s.pt'
        ).value

        self.conf_threshold = float(
            self.declare_parameter('conf_threshold', 0.25).value
        )

        self.show_debug = bool(
            self.declare_parameter('show_debug', False).value
        )

        # QoS
        qos_profile = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Bridge
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            qos_profile
        )

        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            '/image_yolo/image',
            qos_profile
        )

        self.object_pub = self.create_publisher(
            String,
            '/image_yolo/objects',
            10
        )

        # Load model
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.get_logger().info('YOLO model loaded successfully.')

    def image_callback(self, msg: Image):
        # ROS Image -> OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Cannot convert ROS image to OpenCV: {e}')
            return

        # Inference
        try:
            results = self.model.predict(
                source=cv_image,
                conf=self.conf_threshold,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference failed: {e}')
            return

        if not results:
            return

        result = results[0]
        annotated_image = result.plot()

        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                class_name = self.model.names[class_id]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_pixel": [
                        (x1 + x2) / 2.0,
                        (y1 + y2) / 2.0
                    ]
                })

        # Publish JSON detections
        object_info = {
            "timestamp": self.get_clock().now().nanoseconds * 1e-9,
            "objects": detections
        }

        object_msg = String()
        object_msg.data = json.dumps(object_info)
        self.object_pub.publish(object_msg)

        if detections:
            names = [d["class_name"] for d in detections]
            self.get_logger().info(f'Detected objects: {names}')

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding='bgr8'
            )
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Cannot convert OpenCV image back to ROS: {e}')

        # Optional debug window
        if self.show_debug:
            cv2.imshow("YOLO Detection", annotated_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down YOLO node...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
