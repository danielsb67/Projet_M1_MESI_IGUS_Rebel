from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    place_x_arg = DeclareLaunchArgument('place_x', default_value='0.4',   description='X dépôt (m)')
    place_y_arg = DeclareLaunchArgument('place_y', default_value='-0.15', description='Y dépôt (m)')
    place_z_arg = DeclareLaunchArgument('place_z', default_value='-0.01', description='Z dépôt (m)')

    return LaunchDescription([
        place_x_arg,
        place_y_arg,
        place_z_arg,

        # ── 1. Sécurité ─────────────────────────────────────────────
        Node(
            package='mon_controleur',
            executable='securite',
            name='securite',
            output='screen',
            emulate_tty=True,
        ),

        # ── 2. Cage de collision MoveIt ──────────────────────────────
        Node(
            package='mon_controleur',
            executable='workspace_scene',
            name='workspace_scene',
            output='screen',
        ),

        # ── 3. Détection YOLO + profondeur → /vision/roulette_point ──
        Node(
            package='perception',
            executable='view_camera',
            name='vision_yolo_node',
            output='screen',
        ),

        # ── 4. Homographie pixel → repère robot → /arm/roulette_point ─
        Node(
            package='perception',
            executable='homography',
            name='camera_to_arm_homography',
            output='screen',
        ),

        # ── 5. Pick & Place IA — mode réactif ────────────────────────
        Node(
            package='mon_controleur',
            executable='pick_place_ia',
            name='pick_place_ia',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'mode':    'reactive',
                'place_x': LaunchConfiguration('place_x'),
                'place_y': LaunchConfiguration('place_y'),
                'place_z': LaunchConfiguration('place_z'),
            }],
        ),
    ])
