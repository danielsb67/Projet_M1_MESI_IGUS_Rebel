#!/usr/bin/env python3
"""
Launch de test SANS robot — caméra RealSense seule.

Lance uniquement :
  1. detection  : YOLO → /image_yolo/detections + /image_yolo/image
  2. obj_camera : coordonnées 3D dans le repère caméra → /object_position_in_camera

Pré-requis (terminal séparé) :
  ros2 launch realsense2_camera rs_launch.py

Vérification en direct :
  ros2 topic echo /image_yolo/detections          # détections brutes YOLO
  ros2 topic echo /object_position_in_camera      # coordonnées 3D (m, repère caméra)
  ros2 run rqt_image_view rqt_image_view          # image annotée /image_yolo/image
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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

    target_class_arg = DeclareLaunchArgument(
        'target_class',
        default_value='',
        description='Classe YOLO cible (vide = toutes les classes)'
    )

    show_debug_arg = DeclareLaunchArgument(
        'show_debug',
        default_value='False',
        description='Afficher fenêtre OpenCV (True/False)'
    )

    return LaunchDescription([
        model_path_arg,
        conf_threshold_arg,
        target_class_arg,
        show_debug_arg,

        # ── 1. Détection YOLO ────────────────────────────────────────
        # Publie : /image_yolo/detections  (Detection2DArray)
        #          /image_yolo/image       (Image annotée)
        #          /image_yolo/objects     (JSON String)
        Node(
            package='perception',
            executable='detection',
            name='yolo_detection',
            output='screen',
            parameters=[{
                'model_path':     LaunchConfiguration('model_path'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'show_debug':     LaunchConfiguration('show_debug'),
            }],
        ),

        # ── 2. Coordonnées 3D dans le repère caméra ──────────────────
        # Publie : /object_position_in_camera  (PointStamped, mètres)
        # Besoin : profondeur alignée + camera_info (fournis par RealSense)
        Node(
            package='perception',
            executable='obj_camera',
            name='object_position_camera',
            output='screen',
            parameters=[{
                'target_class_id': LaunchConfiguration('target_class'),
                'min_confidence':  LaunchConfiguration('conf_threshold'),
            }],
        ),
    ])
