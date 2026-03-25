import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from builtin_interfaces.msg import Time

import math
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

        #-------QOS camera---------
        qos_profile = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Create publisher and subscribers
        self.object_image = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_profile
        )
        self.object_camera = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_callback, qos_profile
        )
        self.cam_calib = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.cam_calibration, 10
        )
        self.object_position_sub = self.create_subscription(
            PointStamped, '/object_position_in_camera', self.object_pose, 10
        )

        # Publishers
        self.object_position_pub = self.create_publisher(PoseStamped, 'target_pose_cam', 10)
        self.coord_object = self.create_publisher(PointStamped, '/object_position_in_world', 10)

        # Create tf2 buffer
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)

        # Variables to store the latest images
        self.image = None
        self.camerainfo = None
        self.last_detctions = None
        self.last_target = None
        self.target_class_id = self.declare_parameter("num_class", 0).value

        # Camera Calibration
        self.K = None
        self.D = None
        self.dist_model = None

    def image_callback(self, msg: Image):
        self.image = msg

    def camera_callback(self, msg: CameraInfo):
        self.camerainfo = msg

    def cam_calibration(self, msg):
        if self.K is None:
            self.get_logger().info('No calibration detected')
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)

    def detection_callback(self, msg: Detection2DArray):
        self.last_target = msg

        detect_image = self.last_target
        if detect_image is None:
            self.get_logger().info('No object detected')
            return

        for detect in detect_image.detections:
            self.get_logger().info("Detection object")

            if not detect.results:
                continue

            result = detect.results[0]
            class_id = result.hypothesis.class_id
            confidence = result.hypothesis.score

            self.get_logger().info(f"Detected {class_id} with confidence {confidence}")

            if confidence >= 0.90:
                self.get_logger().info("Object detected with approved confidence")

                u = detect.bbox.center.x
                v = detect.bbox.center.y

                self.get_logger().info(f"Bounding box center: (u={u}, v={v})")

                # Use depth callback to get Z value
                Z = self.get_depth_at(u, v)

                # Convert pixel to 3D
                X, Y, Z = self.pixel_to_3d(u, v, Z)

                # Create a PointStamped for the object position
                point = PointStamped()
                point.header.stamp = self.get_clock().now().to_msg()
                point.header.frame_id = "camera_link"
                point.point.x = X
                point.point.y = Y
                point.point.z = Z

                # Publish the object position in the world frame
                self.coord_object.publish(point)

    def depth_callback(self, msg):
        # Dummy depth callback
        pass

    def get_depth_at(self, u: float, v: float):
        # Just a dummy function to get depth at u, v from the depth image
        return 1.0  # Assume 1 meter for now, replace with actual depth retrieval

    def pixel_to_3d(self, u: float, v: float, Z: float):
        if self.K is None or self.D is None:
            return None
        if Z == 0.0:
            return None
        pts = np.array([[[u, v]]])
        und = cv2.undistortPoints(pts, self.K, self.D, P=None)
        x_n, y_n = und[0, 0, 0], und[0, 0, 1]

        X = x_n * Z
        Y = y_n * Z
        return float(X), float(Y), float(Z)

    def object_pose(self, pose):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = X
        pose.pose.position.y = Y
        pose.pose.position.z = Z

        pose.pose.orientation.x = 0.0  # Default orientation values
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0


def main(args=None):
    rclpy.init(args=args)
    node = Calcul_Coordonnee()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
