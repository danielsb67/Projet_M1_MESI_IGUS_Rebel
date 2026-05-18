#!/usr/bin/env python3

import gc
import json
import cv2
import torch
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')

        # Parameters
        self.rgb_topic = self.declare_parameter(
            'rgb_topic',
            '/camera/camera/color/image_raw'
        ).value

        self.model_path = self.declare_parameter(
            'model_path',
            '/home/dbal/projet_igus/src/yolo/yolo/best.pt'
        ).value

        self.conf_threshold = float(
            self.declare_parameter('conf_threshold', 0.25).value
        )

        self.show_debug = bool(
            self.declare_parameter('show_debug', False).value
        )

        # QoS for camera image
        qos_profile = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.bridge = CvBridge()

        # Load YOLO model FIRST — must be ready before any image arrives
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.get_logger().info('YOLO model loaded successfully.')

        # Publisher: annotated YOLO image
        self.image_pub = self.create_publisher(
            Image,
            '/image_yolo/image',
            10
        )

        # Publisher: JSON object info
        self.object_pub = self.create_publisher(
            String,
            '/image_yolo/objects',
            10
        )

        # Publisher: Detection2DArray — QoS RELIABLE (depth=10) so all
        # subscribers using the default QoS integer (RELIABLE) can connect.
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/image_yolo/detections',
            10
        )

        # Subscriber: RealSense RGB image — created LAST, after model + publishers are ready
        self.image_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            qos_profile
        )

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f'Cannot convert ROS image to OpenCV: {e}')
            return

        try:
            with torch.no_grad():
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

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                class_name = self.model.names[class_id]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                # JSON detection
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_pixel": [cx, cy]
                })

                # ROS Detection2D message
                det = Detection2D()
                det.header = msg.header

                det.bbox.center.position.x = float(cx)
                det.bbox.center.position.y = float(cy)
                det.bbox.size_x = float(width)
                det.bbox.size_y = float(height)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(class_name)
                hyp.hypothesis.score = float(confidence)

                det.results.append(hyp)
                detections_msg.detections.append(det)

        # Publish JSON detections
        object_info = {
            "timestamp": self.get_clock().now().nanoseconds * 1e-9,
            "objects": detections
        }

        object_msg = String()
        object_msg.data = json.dumps(object_info)
        self.object_pub.publish(object_msg)

        # Toujours publier Detection2DArray (même vide) pour ne pas bloquer les nœuds en aval
        self.detection_pub.publish(detections_msg)

        if detections:
            names = [d["class_name"] for d in detections]
            self.get_logger().info(
                f'Detected objects: {names}',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().warn(
                f'Aucune détection (conf>={self.conf_threshold:.2f}). '
                'Vérifiez : caméra active, objet visible, modèle correct.',
                throttle_duration_sec=5.0
            )

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated_image,
                encoding='bgr8'
            )
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(
                f'Cannot convert OpenCV image back to ROS: {e}'
            )

        if self.show_debug:
            cv2.imshow("YOLO Detection", annotated_image)
            cv2.waitKey(1)

        # Libère les tenseurs PyTorch pour éviter la fuite mémoire
        del results, result, annotated_image
        gc.collect()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
