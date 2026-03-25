#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')

        # --- Paramètres caméra ---
        self.rgb_cam = self.declare_parameter("rgb_topic", "/camera/color/image_raw").value
        self.depth_cam = self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw").value
        self.cam_info_topic = self.declare_parameter("cam_info_topic", "/camera/color/camera_info").value

        # --- YOLO ---
        self.model_path = self.declare_parameter("model_path", "/home/theresia/yolo_vision/ultralytics").value
        self.input_size = self.declare_parameter("input_size", 640).value
        self.conf_threshold = self.declare_parameter("conf_threshold", 0.35).value
        self.nms_threshold = self.declare_parameter("nms_threshold", 0.45).value
        self.num_class = self.declare_parameter("num_class", 80).value

        # --- Profondeur ---
        self.depth_cam_min = float(self.declare_parameter("depth_min_cam", 0.3).value)
        self.depth_cam_max = float(self.declare_parameter("depth_max_cam", 5.0).value)
        self.depth_window = self.declare_parameter("depth_window", 5).value

        # --- Divers ---
        self.max_detection = self.declare_parameter("max_detection", 5).value
        self.intervalle_ms = self.declare_parameter("intervalle_ms", 33).value
        self.debug = self.declare_parameter("publish_debug_image", False).value
        self.use_gpu = self.declare_parameter("use_gpu", False).value

        # --- État interne ---
        self.bridge = CvBridge()
        self.proc_time = self.get_clock().now()
        self.K = None
        self.D = None
        self.got_intrinsics = False
        self.model_ready = False  # TODO: passer à True après chargement YOLO

        # --- QoS ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Subscribers ---
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_cam, self.rgb_cb, qos_profile=qos)
        self.depth_sub = self.create_subscription(
            Image, self.depth_cam, self.depth_cb, qos_profile=qos)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, self.cam_info_topic, self.camera_info_cb, qos)

        # --- Publishers ---
        self.target_pub = self.create_publisher(PointStamped, "/perception/target", 10)
        self.debug_pub = self.create_publisher(Image, "/perception/debug_image", 10)
        self.status_pub = self.create_publisher(String, "/perception/status", 10)

        # --- Stockage frames ---
        self.latest_rgb = None
        self.latest_depth = None

        self.get_logger().info("Detection node initialized")

    # ── Callbacks ──

    def camera_info_cb(self, msg: CameraInfo):
        """Stocke les intrinsèques caméra (K, D)."""
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)
        if not self.got_intrinsics:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            self.get_logger().info(
                f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, "
                f"cx={cx:.1f}, cy={cy:.1f}, "
                f"width={msg.width}, height={msg.height}, "
                f"D_len={len(self.D)}"
            )
        self.got_intrinsics = True

    def rgb_cb(self, msg: Image):
        self.latest_rgb = msg
        self.try_process()

    def depth_cb(self, msg: Image):
        self.latest_depth = msg
        self.try_process()

    def try_process(self):
        """Traite quand on a RGB + Depth récents."""
        if self.latest_rgb is None or self.latest_depth is None:
            return
        if not self.got_intrinsics or not self.model_ready:
            return

        # FPS control
        now = self.get_clock().now()
        dt_ms = (now - self.proc_time).nanoseconds / 1e6
        if dt_ms < self.intervalle_ms:
            return
        self.proc_time = now

        rgb_msg = self.latest_rgb
        depth_msg = self.latest_depth

        # Convertir en OpenCV
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg)

        # Détection YOLO
        detections = self.detect(rgb_cv)
        if not detections:
            return

        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        best = detections[0]
        x_min, y_min, w, h = best["box"]
        u = int(x_min + w / 2)
        v = int(y_min + h / 2)

        # Profondeur
        z = self.sample_depth(depth_cv, depth_msg.encoding, u, v)
        if z < self.depth_cam_min or z > self.depth_cam_max:
            self.get_logger().warn(f"Invalid depth z={z:.2f}m at ({u},{v})")
            return

        # Déprojection pixel → 3D
        x, y = self.unproject(u, v, z)

        # Publier cible 3D
        target = PointStamped()
        target.header = rgb_msg.header
        target.point.x = x
        target.point.y = y
        target.point.z = z
        self.target_pub.publish(target)

        # Debug image
        if self.debug:
            debug_img = rgb_cv.copy()
            cv2.rectangle(debug_img, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
            cv2.circle(debug_img, (u, v), 4, (0, 0, 255), -1)
            label = f"{best['score']*100:.0f}% {z:.2f}m"
            cv2.putText(debug_img, label, (x_min, max(0, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = rgb_msg.header
            self.debug_pub.publish(debug_msg)

        self.status_pub.publish(String(data=f"1 object at z={z:.2f}m (x={x:.2f}, y={y:.2f})"))

    # ── Méthodes utilitaires (TODO: implémenter) ──

    def detect(self, rgb_cv):
        """YOLO inference. Retourne liste de {'box': [x,y,w,h], 'score': float, 'class_id': int}."""
        # TODO: charger et exécuter le modèle YOLO
        return []

    def sample_depth(self, depth_cv, encoding, u, v):
        """Échantillonne la profondeur autour de (u,v) avec une fenêtre."""
        half = self.depth_window // 2
        h, w = depth_cv.shape[:2]
        u = np.clip(u, half, w - half - 1)
        v = np.clip(v, half, h - half - 1)
        patch = depth_cv[v - half:v + half + 1, u - half:u + half + 1]

        if "16UC1" in encoding or "mono16" in encoding:
            patch = patch.astype(np.float64) / 1000.0  # mm → m
        valid = patch[(patch > self.depth_cam_min) & (patch < self.depth_cam_max)]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))

    def unproject(self, u, v, z):
        """Pixel (u,v) + profondeur z → coordonnées 3D caméra."""
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return x, y


def main():
    rclpy.init()
    node = ObjectDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()