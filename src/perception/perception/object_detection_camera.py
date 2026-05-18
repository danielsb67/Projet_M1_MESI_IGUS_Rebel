#!/usr/bin/env python3

import numpy as np
import cv2
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


class ObjectPositionCamera(Node):
    def __init__(self):
        super().__init__('object_position_camera')

        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('detection_topic', '/image_yolo/detections')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('output_topic', '/object_position_in_camera')

        # Empty string means: accept any detected class
        # You can also put 'Roue' if you only want the wheel
        self.declare_parameter('target_class_id', '')
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('depth_window', 5)
        self.declare_parameter('position_buffer_size', 10)
        self.declare_parameter('stability_threshold', 0.015)
        self.declare_parameter('robot_base_frame', 'igus_rebel_base_link')

        detection_topic = self.get_parameter('detection_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.target_class_id = str(self.get_parameter('target_class_id').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.depth_window = int(self.get_parameter('depth_window').value)
        self.position_buffer_size = int(self.get_parameter('position_buffer_size').value)
        self.stability_threshold = float(self.get_parameter('stability_threshold').value)
        self.robot_base_frame = str(self.get_parameter('robot_base_frame').value)

        qos_camera = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_camera
        )

        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_camera
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            detection_topic,
            self.detection_callback,
            qos_camera
        )

        # Publishers
        self.point_pub = self.create_publisher(
            PointStamped,
            output_topic,
            10
        )
        self.point_robot_pub = self.create_publisher(
            PointStamped,
            '/object_position_in_robot',
            10
        )

        # Stored data
        self.K = None
        self.D = None
        self.camera_frame_id = 'camera_color_optical_frame'
        self.depth_msg = None

        self.camera_info_received = False
        self.camera_info_logged = False

        # Position stabilization buffer (Fonctionnalité 1)
        self.position_buffer = deque(maxlen=self.position_buffer_size)
        self._last_publish_time = 0.0  # timestamp dernière publication

        # TF2 listener (Fonctionnalité 2)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('ObjectPositionCamera node started')
        self.get_logger().info(f'Subscribing to camera_info: {camera_info_topic}')
        self.get_logger().info(f'Subscribing to depth: {depth_topic}')
        self.get_logger().info(f'Subscribing to detections: {detection_topic}')
        self.get_logger().info(f'Publishing object position to: {output_topic}')
        self.get_logger().info(
            f'Stabilization: buffer_size={self.position_buffer_size}, '
            f'stability_threshold={self.stability_threshold} m'
        )
        self.get_logger().info(
            f'TF2: camera→robot frame "{self.robot_base_frame}", '
            f'publishing to /object_position_in_robot'
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)
        self.camera_frame_id = (
            msg.header.frame_id
            if msg.header.frame_id
            else 'camera_color_optical_frame'
        )

        self.camera_info_received = True

        if not self.camera_info_logged:
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            self.get_logger().info(
                f'Camera calibration received: '
                f'fx={fx:.2f}, fy={fy:.2f}, '
                f'cx={cx:.2f}, cy={cy:.2f}, '
                f'frame={self.camera_frame_id}'
            )

            self.camera_info_logged = True

    def depth_callback(self, msg: Image):
        self.depth_msg = msg

    def detection_callback(self, msg: Detection2DArray):
        if self.K is None or self.D is None:
            self.get_logger().warn('No camera calibration received yet')
            return

        if self.depth_msg is None:
            self.get_logger().warn('No depth image received yet')
            return

        best_detection = None
        best_score = -1.0
        best_class_id = ''

        n_detections = len(msg.detections)
        if n_detections == 0:
            self.get_logger().warn(
                'Detection2DArray reçu mais vide (0 détections). '
                'YOLO ne détecte rien — vérifiez le modèle, la caméra et le seuil de confiance.',
                throttle_duration_sec=3.0
            )
            return

        for detection in msg.detections:
            if not detection.results:
                self.get_logger().warn(
                    'Une détection sans résultat (results vide) — ignorée.',
                    throttle_duration_sec=3.0
                )
                continue

            result = detection.results[0]
            class_id = str(result.hypothesis.class_id)
            score = float(result.hypothesis.score)

            # If target_class_id is empty, accept all classes
            if self.target_class_id and class_id != self.target_class_id:
                self.get_logger().debug(
                    f'Détection filtrée : classe "{class_id}" != cible "{self.target_class_id}"'
                )
                continue

            if score < self.min_confidence:
                self.get_logger().debug(
                    f'Détection filtrée : score {score:.2f} < seuil {self.min_confidence:.2f}'
                )
                continue

            if score > best_score:
                best_score = score
                best_detection = detection
                best_class_id = class_id

        if best_detection is None:
            self.get_logger().warn(
                f'Aucune détection retenue parmi {n_detections} '
                f'(cible="{self.target_class_id}", seuil={self.min_confidence:.2f}). '
                'Vérifiez target_class_id et min_confidence.',
                throttle_duration_sec=3.0
            )
            return

        u = float(best_detection.bbox.center.position.x)
        v = float(best_detection.bbox.center.position.y)

        z = self.get_depth_at(u, v)

        if z is None:
            self.get_logger().warn(f'No valid depth at pixel ({u:.1f}, {v:.1f})')
            return

        point_3d = self.pixel_to_3d(u, v, z)

        if point_3d is None:
            self.get_logger().warn('Failed to convert pixel to 3D point')
            return

        x_raw, y_raw, z_raw = point_3d

        self.get_logger().info(
            f'Object {best_class_id}: '
            f'u={u:.1f}, v={v:.1f}, '
            f'X={x_raw:.3f} m, Y={y_raw:.3f} m, Z={z_raw:.3f} m, '
            f'confidence={best_score:.2f}'
        )

        # --- Fonctionnalité 1 : filtre de stabilisation ---
        self.position_buffer.append((x_raw, y_raw, z_raw))

        xs = np.array([p[0] for p in self.position_buffer])
        ys = np.array([p[1] for p in self.position_buffer])
        zs = np.array([p[2] for p in self.position_buffer])

        std_z = float(np.std(zs))
        if std_z >= self.stability_threshold and len(self.position_buffer) >= self.position_buffer_size:
            self.get_logger().debug(
                f'Position instable (std_Z={std_z:.4f} m >= seuil {self.stability_threshold:.3f} m) '
                f'— publication supprimée.',
                throttle_duration_sec=2.0
            )
            return

        x = float(np.median(xs))
        y = float(np.median(ys))
        z = float(np.median(zs))

        # Limite de publication : 1 Hz max
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_publish_time < 1.0:
            return
        self._last_publish_time = now

        # Publish stable position in camera frame
        point_msg = PointStamped()
        point_msg.header.stamp = msg.header.stamp
        point_msg.header.frame_id = self.camera_frame_id
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z

        self.point_pub.publish(point_msg)

        # --- Fonctionnalité 2 : transformation TF2 vers robot base ---
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.camera_frame_id,
                rclpy.time.Time()
            )
            point_robot = do_transform_point(point_msg, transform)
            point_robot.header.frame_id = self.robot_base_frame
            self.point_robot_pub.publish(point_robot)

            self.get_logger().info(
                f'Object stable: '
                f'X={point_robot.point.x:.3f} m, '
                f'Y={point_robot.point.y:.3f} m, '
                f'Z={point_robot.point.z:.3f} m (robot frame, '
                f'n={len(self.position_buffer)}, std_Z={std_z:.4f} m)',
                throttle_duration_sec=2.0
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF non disponible ({self.robot_base_frame}): {e}',
                throttle_duration_sec=5.0
            )
            self.get_logger().info(
                f'Object stable: '
                f'X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m (camera frame, '
                f'n={len(self.position_buffer)}, std_Z={std_z:.4f} m)',
                throttle_duration_sec=2.0
            )

    def get_depth_at(self, u: float, v: float):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                self.depth_msg,
                desired_encoding='passthrough'
            )
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return None

        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]

        u_i = int(round(u))
        v_i = int(round(v))

        if u_i < 0 or u_i >= w or v_i < 0 or v_i >= h:
            return None

        r = self.depth_window

        u_min = max(0, u_i - r)
        u_max = min(w, u_i + r + 1)
        v_min = max(0, v_i - r)
        v_max = min(h, v_i + r + 1)

        roi = depth_image[v_min:v_max, u_min:u_max]

        if roi.size == 0:
            return None

        if self.depth_msg.encoding in ['16UC1', 'mono16']:
            roi_m = roi.astype(np.float32) * 0.001
        elif self.depth_msg.encoding in ['32FC1']:
            roi_m = roi.astype(np.float32)
        else:
            self.get_logger().warn(
                f'Encodage depth inconnu : "{self.depth_msg.encoding}". '
                'Attendu : 16UC1, mono16, ou 32FC1. '
                'Traitement comme float32 brut.',
                throttle_duration_sec=5.0
            )
            roi_m = roi.astype(np.float32)

        valid = roi_m[np.isfinite(roi_m) & (roi_m > 0.0)]

        if valid.size == 0:
            self.get_logger().warn(
                f'Toutes les valeurs depth dans la fenêtre ({2*self.depth_window+1}x'
                f'{2*self.depth_window+1}) autour de ({u:.0f},{v:.0f}) sont nulles '
                f'ou invalides. Encodage: {self.depth_msg.encoding}. '
                'Objet trop proche, trop loin, ou profondeur non alignée ?',
                throttle_duration_sec=3.0
            )
            return None

        return float(np.median(valid))

    def pixel_to_3d(self, u: float, v: float, z: float):
        if self.K is None or self.D is None:
            return None

        if z <= 0.0:
            return None

        pts = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.K, self.D, P=None)

        x_n = undistorted[0, 0, 0]
        y_n = undistorted[0, 0, 1]

        x = x_n * z
        y = y_n * z

        return float(x), float(y), float(z)


def main(args=None):
    rclpy.init(args=args)

    node = ObjectPositionCamera()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
