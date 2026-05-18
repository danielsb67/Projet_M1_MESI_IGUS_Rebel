import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from vision_msgs.msg import Detection2DArray

import numpy as np
import cv2

import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time as RclpyTime


class Calcul_Coordonnee(Node):
    def __init__(self):
        super().__init__('detection_publisher')

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.object_image = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, qos_profile
        )
        self.object_camera = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_callback, qos_profile
        )
        self.cam_calib = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.cam_calibration, qos_profile
        )
        # Subscription aux détections YOLO — QoS 10 (RELIABLE) doit correspondre
        # au publisher detection_pub dans detection.py (aussi QoS 10)
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/image_yolo/detections', self.detection_callback, 10
        )
        self.object_position_sub = self.create_subscription(
            PointStamped, '/object_position_in_camera', self.object_pose, 10
        )

        # Publishers
        self.object_position_pub = self.create_publisher(PoseStamped, 'target_pose_cam', 10)
        self.coord_object = self.create_publisher(PointStamped, '/object_position_in_world', 10)

        # TF2
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)

        self.image = None
        self.camerainfo = None
        self.last_detections = None
        self.last_depth = None
        self.target_class_id = self.declare_parameter('num_class', 0).value

        self.K = None
        self.D = None

    def image_callback(self, msg: Image):
        self.image = msg

    def camera_callback(self, msg: CameraInfo):
        self.camerainfo = msg

    def cam_calibration(self, msg: CameraInfo):
        # Early-exit si déjà calibré pour ne lire qu'une seule fois
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)
        self.get_logger().info('Calibration caméra reçue')

    def detection_callback(self, msg: Detection2DArray):
        self.last_detections = msg

        for detect in msg.detections:
            if not detect.results:
                continue

            result = detect.results[0]
            class_id = result.hypothesis.class_id
            confidence = result.hypothesis.score

            if confidence < 0.25:   # abaissé de 0.90 → 0.25 pour correspondre au seuil YOLO
                continue

            self.get_logger().info(f'Objet détecté : {class_id} ({confidence:.2f})')

            # Utiliser bbox.center.position (API ROS2 Humble vision_msgs)
            u = detect.bbox.center.position.x
            v = detect.bbox.center.position.y

            Z = self.get_depth_at(u, v)
            result_3d = self.pixel_to_3d(u, v, Z)

            if result_3d is None:
                self.get_logger().warn('pixel_to_3d a échoué (calibration manquante ou Z=0)')
                continue

            X, Y, Z = result_3d

            point = PointStamped()
            point.header.stamp = self.get_clock().now().to_msg()
            point.header.frame_id = 'camera_color_optical_frame'
            point.point.x = X
            point.point.y = Y
            point.point.z = Z

            self.coord_object.publish(point)

    def depth_callback(self, msg: Image):
        self.last_depth = msg

    def get_depth_at(self, u: float, v: float) -> float:
        if self.last_depth is None:
            return 1.0
        try:
            depth_image = self.bridge.imgmsg_to_cv2(self.last_depth, desired_encoding='passthrough')
        except CvBridgeError:
            return 1.0

        u_i, v_i = int(round(u)), int(round(v))
        h, w = depth_image.shape[:2]
        if not (0 <= u_i < w and 0 <= v_i < h):
            return 1.0

        val = depth_image[v_i, u_i]
        if self.last_depth.encoding in ['16UC1', 'mono16']:
            return float(val) * 0.001
        return float(val)

    def pixel_to_3d(self, u: float, v: float, Z: float):
        if self.K is None or self.D is None:
            return None
        if Z <= 0.0:
            return None
        pts = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K, self.D, P=None)
        x_n, y_n = und[0, 0, 0], und[0, 0, 1]
        return float(x_n * Z), float(y_n * Z), float(Z)

    def object_pose(self, msg: PointStamped):
        """Reçoit un PointStamped dans le repère caméra et publie un PoseStamped."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'igus_rebel_base_link'
        pose.pose.position.x = msg.point.x
        pose.pose.position.y = msg.point.y
        pose.pose.position.z = msg.point.z
        pose.pose.orientation.w = 1.0
        self.object_position_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = Calcul_Coordonnee()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
