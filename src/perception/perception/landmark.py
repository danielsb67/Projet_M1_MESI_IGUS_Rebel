#!/usr/bin/env python3
import os
import math
from collections import deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, TwistStamped
from cv_bridge import CvBridge

import mediapipe as mp
from mediapipe.tasks.python import vision

try:
    import tf2_ros
    from geometry_msgs.msg import PointStamped
    from tf2_geometry_msgs import do_transform_point
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unwrap_angle(dtheta):
    while dtheta > math.pi:
        dtheta -= 2.0 * math.pi
    while dtheta < -math.pi:
        dtheta += 2.0 * math.pi
    return dtheta


class IndexFingerTeleop(Node):
    """
    Index fingertip (MediaPipe landmark 8) controls translation.
    Circle motion (in XY plane) controls yaw (angular.z).
    Optional pinch clutch (thumb tip 4 <-> index tip 8).
    """

    def __init__(self):
        super().__init__('index_finger_teleop')

        # ---------------- Parameters ----------------
        self.rgb_topic = self.declare_parameter('rgb_topic', '/camera/color/image_raw').value
        self.depth_topic = self.declare_parameter('depth_topic', '/camera/depth/image_raw').value
        self.info_topic = self.declare_parameter('cam_info_topic', '/camera/color/camera_info').value

        # MediaPipe model path (REQUIRED for Tasks API)
        default_model = os.path.expanduser('~/Desktop/models/hand_landmarker.task')
        self.model_path = self.declare_parameter('model_path', default_model).value

        # Output mode: "twist" or "twist_stamped"
        self.output_mode = self.declare_parameter('output_mode', 'twist').value  # twist | twist_stamped
        self.cmd_topic = self.declare_parameter('cmd_topic', '/cmd_vel').value
        self.twist_frame = self.declare_parameter('twist_frame', 'base_link').value

        # Control gains and limits
        self.gain_lin = float(self.declare_parameter('gain_lin', 1.0).value)
        self.gain_yaw = float(self.declare_parameter('gain_yaw', 1.0).value)
        self.max_v = float(self.declare_parameter('max_v', 0.25).value)
        self.max_wz = float(self.declare_parameter('max_wz', 1.0).value)
        self.deadband_m = float(self.declare_parameter('deadband_m', 0.004).value)
        self.alpha = float(self.declare_parameter('smoothing_alpha', 0.3).value)

        # Depth usage
        self.use_depth = bool(self.declare_parameter('use_depth', True).value)
        self.depth_window = int(self.declare_parameter('depth_window', 2).value)

        # Pinch clutch
        self.use_pinch = bool(self.declare_parameter('use_pinch', True).value)
        self.pinch_on_m = float(self.declare_parameter('pinch_on_m', 0.035).value)
        self.pinch_off_m = float(self.declare_parameter('pinch_off_m', 0.045).value)

        # Circle detection for yaw
        self.circle_hist_len = int(self.declare_parameter('circle_hist_len', 10).value)
        self.r_min = float(self.declare_parameter('circle_r_min', 0.015).value)
        self.omega_min = float(self.declare_parameter('circle_omega_min', 0.7).value)

        # Optional TF transform camera_frame -> base_link
        self.use_tf = bool(self.declare_parameter('use_tf', False).value) and TF_AVAILABLE
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_color_optical_frame').value
        self.target_frame = self.declare_parameter('target_frame', 'base_link').value

        # Debug view
        self.show_debug = bool(self.declare_parameter('show_debug', False).value)

        # ---------------- QoS for camera ----------------
        self.qos_profile = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # ---------------- ROS interfaces ----------------
        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_cb, self.qos_profile)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_cb, self.qos_profile)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_cb, self.qos_profile)

        if self.output_mode == 'twist_stamped':
            self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        # ---------------- MediaPipe Tasks HandLandmarker ----------------
        if not os.path.isfile(self.model_path):
            self.get_logger().error(
                f"Model file not found: {self.model_path}\n"
                "Download it with:\n"
                "  mkdir -p ~/Desktop/models\n"
                "  wget -O ~/Desktop/models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
            )
            raise FileNotFoundError(self.model_path)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,   # we will call detect_for_video(...)
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # ---------------- Camera intrinsics ----------------
        self.fx = self.fy = None
        self.cx = self.cy = None

        # ---------------- Latest depth frame ----------------
        self.last_depth = None
        self.depth_is_mm = False

        # ---------------- State ----------------
        self.prev_tip = None
        self.prev_time = None
        self.vel_filt = np.zeros(3, dtype=np.float32)
        self.circle_xy = deque(maxlen=self.circle_hist_len)
        self.enabled = (not self.use_pinch)

        # ---------------- TF (optional) ----------------
        if self.use_tf:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"RGB: {self.rgb_topic}")
        self.get_logger().info(f"DEPTH: {self.depth_topic} | use_depth={self.use_depth}")
        self.get_logger().info(f"INFO: {self.info_topic}")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Output: mode={self.output_mode} topic={self.cmd_topic}")
        if self.use_tf:
            self.get_logger().info(f"TF: {self.camera_frame} -> {self.target_frame}")

    def destroy_node(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
        super().destroy_node()

    # ---------------- Callbacks ----------------
    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.last_depth = depth
        self.depth_is_mm = (depth.dtype == np.uint16)

    # ---------------- Helpers ----------------
    def depth_at(self, u: int, v: int):
        if self.last_depth is None:
            return None
        h, w = self.last_depth.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        r = self.depth_window
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)

        win = self.last_depth[v0:v1, u0:u1].astype(np.float32)
        if self.depth_is_mm:
            win = win / 1000.0

        win = win[np.isfinite(win)]
        win = win[win > 0.0]
        if win.size == 0:
            return None
        return float(np.median(win))

    def pixel_to_3d(self, u: int, v: int, Z: float):
        X = (u - self.cx) / self.fx * Z
        Y = (v - self.cy) / self.fy * Z
        return np.array([X, Y, Z], dtype=np.float32)

    def maybe_tf_point(self, p_cam: np.ndarray):
        if not self.use_tf:
            return p_cam

        try:
            t = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
            ps = PointStamped()
            ps.header.frame_id = self.camera_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.point.x, ps.point.y, ps.point.z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
            out = do_transform_point(ps, t)
            return np.array([out.point.x, out.point.y, out.point.z], dtype=np.float32)
        except Exception:
            return p_cam

    def circle_yaw_rate(self, dt: float):
        if len(self.circle_xy) < 6:
            return 0.0

        pts = np.array(self.circle_xy, dtype=np.float32)
        c = pts.mean(axis=0)
        vec = pts - c
        r = np.linalg.norm(vec, axis=1).mean()
        if r < self.r_min:
            return 0.0

        th_prev = math.atan2(vec[-2, 1], vec[-2, 0])
        th_now = math.atan2(vec[-1, 1], vec[-1, 0])
        dth = unwrap_angle(th_now - th_prev)
        omega = dth / max(dt, 1e-3)

        if abs(omega) < self.omega_min:
            return 0.0
        return omega

    def publish_cmd(self, v_xyz: np.ndarray, wz: float):
        if self.output_mode == 'twist_stamped':
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.twist_frame
            msg.twist.linear.x = float(v_xyz[0])
            msg.twist.linear.y = float(v_xyz[1])
            msg.twist.linear.z = float(v_xyz[2])
            msg.twist.angular.z = float(wz)
            self.cmd_pub.publish(msg)
        else:
            msg = Twist()
            msg.linear.x = float(v_xyz[0])
            msg.linear.y = float(v_xyz[1])
            msg.linear.z = float(v_xyz[2])
            msg.angular.z = float(wz)
            self.cmd_pub.publish(msg)

    # ---------------- Main pipeline ----------------
    def rgb_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(self.get_clock().now().nanoseconds / 1e6)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        v_cmd = np.zeros(3, dtype=np.float32)
        wz_cmd = 0.0

        if not result.hand_landmarks:
            self.prev_tip = None
            self.prev_time = None
            self.vel_filt[:] = 0.0
            self.circle_xy.clear()
            if self.use_pinch:
                self.enabled = False
            self.publish_cmd(v_cmd, wz_cmd)
            if self.show_debug:
                cv2.imshow("index_teleop", frame)
                cv2.waitKey(1)
            return

        lm = result.hand_landmarks[0]  # first hand

        idx = lm[8]
        thb = lm[4]

        u8, v8 = int(idx.x * w), int(idx.y * h)
        u4, v4 = int(thb.x * w), int(thb.y * h)

        if (self.fx is None) or (self.last_depth is None) or (not self.use_depth):
            self.publish_cmd(v_cmd, wz_cmd)
            return

        Z8 = self.depth_at(u8, v8)
        Z4 = self.depth_at(u4, v4)
        if (Z8 is None) or (Z4 is None):
            self.publish_cmd(v_cmd, wz_cmd)
            return

        p8_cam = self.pixel_to_3d(u8, v8, Z8)
        p4_cam = self.pixel_to_3d(u4, v4, Z4)

        p8 = self.maybe_tf_point(p8_cam)
        p4 = self.maybe_tf_point(p4_cam)

        # ---- Pinch clutch ----
        if self.use_pinch:
            pinch_dist = float(np.linalg.norm(p8 - p4))
            if not self.enabled and pinch_dist < self.pinch_on_m:
                self.enabled = True
                self.prev_tip = p8
                self.prev_time = self.get_clock().now().nanoseconds * 1e-9
                self.vel_filt[:] = 0.0
                self.circle_xy.clear()
            elif self.enabled and pinch_dist > self.pinch_off_m:
                self.enabled = False
                self.prev_tip = None
                self.prev_time = None
                self.vel_filt[:] = 0.0
                self.circle_xy.clear()

        if not self.enabled:
            self.publish_cmd(v_cmd, wz_cmd)
            if self.show_debug:
                cv2.circle(frame, (u8, v8), 8, (0, 0, 255), -1)
                cv2.imshow("index_teleop", frame)
                cv2.waitKey(1)
            return

        # ---- dt ----
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is None:
            self.prev_time = now
            self.prev_tip = p8
            self.publish_cmd(v_cmd, wz_cmd)
            return

        dt = max(1e-3, now - self.prev_time)
        self.prev_time = now

        if self.prev_tip is None:
            self.prev_tip = p8
            self.publish_cmd(v_cmd, wz_cmd)
            return

        dp = p8 - self.prev_tip
        self.prev_tip = p8

        v_raw = dp / dt

        v_db = self.deadband_m / dt
        v_raw[np.abs(v_raw) < v_db] = 0.0

        self.vel_filt = (1.0 - self.alpha) * self.vel_filt + self.alpha * v_raw

        v_cmd = self.gain_lin * self.vel_filt
        v_cmd[0] = clamp(v_cmd[0], -self.max_v, self.max_v)
        v_cmd[1] = clamp(v_cmd[1], -self.max_v, self.max_v)
        v_cmd[2] = clamp(v_cmd[2], -self.max_v, self.max_v)

        # ---- Circle -> yaw ----
        self.circle_xy.append((float(p8[0]), float(p8[1])))
        omega = self.circle_yaw_rate(dt)
        wz_cmd = clamp(self.gain_yaw * omega, -self.max_wz, self.max_wz)

        self.publish_cmd(v_cmd, wz_cmd)

        if self.show_debug:
            cv2.circle(frame, (u8, v8), 8, (0, 255, 0), -1)
            cv2.circle(frame, (u4, v4), 6, (255, 0, 0), -1)
            cv2.putText(frame, f"enabled={self.enabled}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"v=({v_cmd[0]:.2f},{v_cmd[1]:.2f},{v_cmd[2]:.2f}) wz={wz_cmd:.2f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("index_teleop", frame)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = IndexFingerTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

