#!/usr/bin/env python3
"""
Launch complet : Pick & Place sécurisé avec perception YOLO + caméra RealSense.

Ce launch est AUTONOME pour la partie vision : il démarre lui-même le driver
RealSense (D435) avec l'infrarouge désactivé et un débit réduit (6 FPS) pour
alléger le bus USB / la communication DDS.

6 FPS = le minimum officiellement supporté par la D435 à 640x480 (valeurs
valides : 6, 15, 30, 60). Cela divise par 5 la bande passante USB par rapport
au 30 FPS par défaut et supprime les coupures "Frames didn't arrive within
5 seconds". NE PAS lancer rs_launch.py séparément en plus de ce launch.

Pré-requis (à lancer séparément AVANT ce launch) :
  ros2 launch igus_rebel_moveit_config demo.launch.py \
      hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none camera:=none load_base:=false

Pipeline :
  RealSense → detection (YOLO) → obj_camera (3D) → obj_robot (TF) → pick_place_ia (réactif)

Séquence de démarrage :
  t=0s : sécurité, cage collision, TF caméra→robot, driver RealSense
  t=3s : detection YOLO, obj_camera, obj_robot, pick_place_ia
         (le délai laisse à la caméra le temps de s'initialiser)

NOTE use_sim_time :
  Ce launch est pour le hardware réel. Aucun nœud ne reçoit use_sim_time=True (défaut=False).
  C'est intentionnel : l'horloge système (wall clock) est utilisée.
  NE PAS ajouter use_sim_time=True ici sauf en mode simulation.

NOTE topics caméra RealSense :
  Le driver publie sous le namespace /camera/camera/... (camera_name=camera par défaut).
  detection → /camera/camera/color/image_raw
  obj_camera → /camera/camera/aligned_depth_to_color/image_raw + /camera/camera/color/camera_info

Montage caméra physique (valeurs par défaut figées ci-dessous) :
  Caméra surélevée devant le robot, axe optique partant de la verticale (-Z) et
  incliné de 30° vers le robot, dans le plan X-Z (aucune composante latérale).
    cam_x/y/z   = 0.56 / 0.0 / 0.40 m   (position de camera_link / igus_rebel_base_link)
    cam_pitch   = π/3 ≈ 1.0472 rad      (= π/2 - 30° d'inclinaison)
    cam_yaw     = π ≈ 3.14159 rad       (la plongée pointe vers le robot)
    cam_roll    = 0
  Si les objets détectés apparaissent en Y miroir (gauche/droite inversés), la
  caméra est montée à l'envers : utiliser cam_pitch:=2.0944 cam_yaw:=0.0.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
        description='Classe cible YOLO (vide = accepter toutes les classes)'
    )

    # Bac de dépôt fixe : 25 cm devant le robot (Y), centré (X=0), à 10 cm de haut.
    # Choisi pour ne JAMAIS coïncider avec la zone de pick (objets posés vers X≈0.4).
    place_x_arg = DeclareLaunchArgument('place_x', default_value='0.0',  description='X dépôt (m) — centré sur le robot')
    place_y_arg = DeclareLaunchArgument('place_y', default_value='0.25', description='Y dépôt (m) — 25 cm sur le côté du robot')
    place_z_arg = DeclareLaunchArgument('place_z', default_value='0.10', description='Z dépôt (m) — 10 cm au-dessus du bac')

    # ── Montage caméra ────────────────────────────────────────────────
    cam_x_arg     = DeclareLaunchArgument('cam_x',     default_value='0.56',    description='X caméra dans repère robot (m)')
    cam_y_arg     = DeclareLaunchArgument('cam_y',     default_value='0.0',     description='Y caméra dans repère robot (m)')
    cam_z_arg     = DeclareLaunchArgument('cam_z',     default_value='0.40',    description='Z caméra dans repère robot (m)')
    cam_roll_arg  = DeclareLaunchArgument('cam_roll',  default_value='0.0',     description='Roll caméra (rad)')
    cam_pitch_arg = DeclareLaunchArgument('cam_pitch', default_value='1.0472',  description='Pitch caméra (rad). π/3 = inclinée 30° vers le robot (π/2 - 30°)')
    cam_yaw_arg   = DeclareLaunchArgument('cam_yaw',   default_value='3.14159', description='Yaw caméra (rad). π = la plongée pointe vers le robot')

    # ── Driver caméra RealSense D435 ─────────────────────────────────
    # 640x480 @ 6 FPS, infrarouge désactivé : allège le bus USB et la
    # bande passante DDS. respawn=True : redémarre après déconnexion USB.
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            '/opt/ros/humble/share/realsense2_camera/launch/rs_launch.py'
        ),
        launch_arguments={
            'align_depth.enable':         'true',
            'depth_module.depth_profile': '640x480x6',
            'rgb_camera.color_profile':   '640x480x6',
            'enable_infra1':              'false',
            'enable_infra2':              'false',
            'enable_color':               'true',
            'enable_depth':               'true',
            'respawn':                    'true',
            'respawn_delay':              '3.0',
        }.items(),
    )

    return LaunchDescription([
        model_path_arg,
        conf_threshold_arg,
        target_class_arg,
        place_x_arg,
        place_y_arg,
        place_z_arg,
        cam_x_arg,
        cam_y_arg,
        cam_z_arg,
        cam_roll_arg,
        cam_pitch_arg,
        cam_yaw_arg,

        # ── 1. Nœud sécurité ────────────────────────────────────────
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

        # ── 3. Transform statique caméra → robot (REQUIS pour obj_robot) ──
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_robot_tf',
            output='screen',
            arguments=[
                '--x',        LaunchConfiguration('cam_x'),
                '--y',        LaunchConfiguration('cam_y'),
                '--z',        LaunchConfiguration('cam_z'),
                '--roll',     LaunchConfiguration('cam_roll'),
                '--pitch',    LaunchConfiguration('cam_pitch'),
                '--yaw',      LaunchConfiguration('cam_yaw'),
                '--frame-id', 'igus_rebel_base_link',
                '--child-frame-id', 'camera_link',
            ],
        ),

        # ── 4. Driver caméra RealSense (IR désactivé, 6 FPS) ─────────
        realsense_node,

        # ── Perception + pick&place — démarrés 3 s après la caméra ───
        # Le délai laisse à la RealSense le temps de s'initialiser et de
        # commencer à publier avant que YOLO ne s'abonne aux topics.
        TimerAction(
            period=3.0,
            actions=[

                # ── 5. Détection YOLO ────────────────────────────────
                Node(
                    package='perception',
                    executable='detection',
                    name='yolo_detection',
                    output='screen',
                    parameters=[{
                        'model_path':     LaunchConfiguration('model_path'),
                        'conf_threshold': LaunchConfiguration('conf_threshold'),
                        'show_debug':     False,
                    }],
                ),

                # ── 6. Position objet en 3D dans le repère caméra ────
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

                # ── 7. Transform position caméra → repère robot ──────
                Node(
                    package='perception',
                    executable='obj_robot',
                    name='object_position_transform',
                    output='screen',
                    parameters=[{
                        'target_frame': 'igus_rebel_base_link',
                    }],
                ),

                # ── 8. Pick & Place IA — mode réactif ────────────────
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
            ],
        ),
    ])
