#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseStamped

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point


class ObjectPositionTransform(Node):
    def __init__(self):
        super().__init__('object_position_transform')

        self.declare_parameter('input_topic', '/object_position_in_camera')
        self.declare_parameter('output_point_topic', '/object_position_in_world')
        self.declare_parameter('output_pose_topic', '/target_pose_world')
        self.declare_parameter('target_frame', 'igus_rebel_base_link')

        input_topic = self.get_parameter('input_topic').value
        output_point_topic = self.get_parameter('output_point_topic').value
        output_pose_topic = self.get_parameter('output_pose_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PointStamped, input_topic, self.point_callback, 10)

        self.point_pub = self.create_publisher(PointStamped, output_point_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, output_pose_topic, 10)

        self.get_logger().info(
            f'ObjectPositionTransform démarré : '
            f'{input_topic} → {output_point_topic} (frame: {self.target_frame})'
        )
        self.get_logger().info(
            'IMPORTANT : la transform caméra→robot doit être fournie par le driver '
            'RealSense ou un static_transform_publisher externe.'
        )

    def point_callback(self, msg: PointStamped):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.5)
            )

            transformed_point = do_transform_point(msg, transform)
            transformed_point.header.frame_id = self.target_frame

            self.point_pub.publish(transformed_point)

            pose_msg = PoseStamped()
            pose_msg.header = transformed_point.header
            pose_msg.pose.position.x = transformed_point.point.x
            pose_msg.pose.position.y = transformed_point.point.y
            pose_msg.pose.position.z = transformed_point.point.z
            pose_msg.pose.orientation.w = 1.0

            self.pose_pub.publish(pose_msg)

            self.get_logger().info(
                f'Point transformé : '
                f'X={transformed_point.point.x:.3f} m, '
                f'Y={transformed_point.point.y:.3f} m, '
                f'Z={transformed_point.point.z:.3f} m '
                f'(frame: {self.target_frame})'
            )

        except TransformException as e:
            self.get_logger().warn(f'TF transform failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPositionTransform()
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
