#!/usr/bin/env python3
"""
Launch tout-en-un pour le pick & place — caméra + perception YOLO.

Lance dans l'ordre :
  1. realsense2_camera : flux RGB + profondeur alignée à 5 FPS (bande passante réduite)
  2. detection         : YOLO → /image_yolo/detections + /image_yolo/image  (après 2 s)
  3. obj_camera        : coordonnées 3D dans le repère caméra → /object_position_in_camera

Utilisation :
  ros2 launch perception pick_detection.launch.py
  ros2 launch perception pick_detection.launch.py model_path:=/chemin/vers/model.pt
  ros2 launch perception pick_detection.launch.py conf_threshold:=0.6 show_debug:=True

Vérification :
  ros2 topic echo /image_yolo/detections          # détections brutes YOLO
  ros2 topic echo /object_position_in_camera      # coordonnées 3D (m, repère caméra)
  ros2 run rqt_image_view rqt_image_view          # image annotée /image_yolo/image
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # ── Arguments exposés ────────────────────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/dbal/projet_igus/src/yolo/yolo/best.pt',
        description='Chemin vers le modèle YOLO (.pt)'
    )

    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.5',
        description='Seuil de confiance YOLO (0.0 à 1.0)'
    )

    show_debug_arg = DeclareLaunchArgument(
        'show_debug',
        default_value='False',
        description='Afficher fenêtre OpenCV debug (True/False)'
    )

    # ── 1. Caméra RealSense D435 — 5 FPS (bande passante réduite) ────────
    # Profil : 640x480 @ 5 FPS pour RGB et profondeur
    # Infrarouge désactivé pour alléger le bus USB
    # respawn=True : redémarre automatiquement après une déconnexion USB
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            '/opt/ros/humble/share/realsense2_camera/launch/rs_launch.py'
        ),
        launch_arguments={
            'align_depth.enable':           'true',
            'depth_module.depth_profile':   '640x480x5',
            'rgb_camera.color_profile':     '640x480x5',
            'enable_infra1':                'false',
            'enable_infra2':                'false',
            'enable_color':                 'true',
            'enable_depth':                 'true',
            'respawn':                      'true',
            'respawn_delay':                '3.0',
        }.items(),
    )

    # ── 2. Nœuds de perception — démarrés 2 s après la caméra ───────────
    # Le délai laisse à la RealSense le temps de s'initialiser et de
    # commencer à publier avant que YOLO ne s'abonne aux topics.
    perception_nodes = TimerAction(
        period=2.0,
        actions=[

            # ── 2a. Détection YOLO ────────────────────────────────────
            # Publie : /image_yolo/detections  (Detection2DArray)
            #          /image_yolo/image       (Image annotée)
            #          /image_yolo/objects     (JSON String)
            Node(
                package='perception',
                executable='detection',
                name='yolo_detection',
                output='screen',
                respawn=True,
                respawn_delay=5.0,
                parameters=[{
                    'model_path':     LaunchConfiguration('model_path'),
                    'conf_threshold': LaunchConfiguration('conf_threshold'),
                    'show_debug':     LaunchConfiguration('show_debug'),
                }],
            ),

            # ── 2b. Coordonnées 3D dans le repère caméra ─────────────
            # Publie : /object_position_in_camera  (PointStamped, mètres)
            # Besoin : profondeur alignée + camera_info (fournis par RealSense)
            Node(
                package='perception',
                executable='obj_camera',
                name='object_position_camera',
                output='screen',
                respawn=True,
                respawn_delay=5.0,
                parameters=[{
                    'target_class_id': '',
                    'min_confidence':  LaunchConfiguration('conf_threshold'),
                }],
            ),
        ],
    )

    return LaunchDescription([
        model_path_arg,
        conf_threshold_arg,
        show_debug_arg,

        realsense_node,
        perception_nodes,
    ])
