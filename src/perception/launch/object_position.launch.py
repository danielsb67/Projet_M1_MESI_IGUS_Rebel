from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='perception',
            executable='detection',
            name='detection',
            output='screen',
            parameters=[
                {
                    'rgb_topic': '/camera/camera/color/image_raw',
                    'model_path': '/home/dbal/projet_igus/src/yolo/yolo/best.pt',
                    'conf_threshold': 0.25,
                    'show_debug': False,
                }
            ]
        ),

        Node(
            package='perception',
            executable='obj_camera',
            name='object_detection_camera',
            output='screen',
            parameters=[
                {
                    'detection_topic': '/image_yolo/detections',
                    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                    'camera_info_topic': '/camera/camera/color/camera_info',
                    'output_topic': '/object_position_in_camera',
                    'target_class_id': '',
                    'min_confidence': 0.25,
                }
            ]
        ),

        Node(
            package='perception',
            executable='obj_robot',
            name='object_detection_robot',
            output='screen',
            parameters=[
                {
                    'input_topic': '/object_position_in_camera',
                    'output_point_topic': '/object_position_in_world',
                    'output_pose_topic': '/target_pose_world',
                    'target_frame': 'igus_rebel_base_link',
                }
            ]
        ),

    ])
