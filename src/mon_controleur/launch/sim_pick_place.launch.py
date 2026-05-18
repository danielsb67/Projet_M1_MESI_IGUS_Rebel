#!/usr/bin/env python3
"""
Simulation Gazebo + Pick & Place — sans robot réel, sans caméra.

Lance dans l'ordre :
  1. moveit_controller.launch.py : MoveIt2 + RViz + ros2_control (ignition)
  2. ignition.launch.py          : Gazebo Ignition + spawn robot
  3. securite + workspace_scene + pick_place_ia (après 20 s)

NOTE robot_state_publisher :
  moveit_controller.launch.py publie UN SEUL robot_state_publisher (avec use_sim_time=True
  car load_gazebo=true). ignition.launch.py ne lance PAS de second robot_state_publisher
  (il lance seulement ign_sim + ign_spawn_entity + ign_bridge). Pas de doublon RSP.
  Si le robot clignote dans RViz, vérifier que /joint_states est bien bridgé depuis
  Gazebo via bridge_moveit.yaml et que use_sim_time est cohérent sur tous les nœuds.

NOTE délai de démarrage :
  Le TimerAction est réglé à 20 s. Gazebo Ignition + move_group peuvent prendre 15-20 s
  à être opérationnels selon la machine (chargement du monde SDF, spawn URDF, planners).
  Si CONTROL_FAILED persiste au démarrage, augmenter à 25 s.

Utilisation :
  ws && ros2 launch mon_controleur sim_pick_place.launch.py

Coordonnées personnalisées :
  ros2 launch mon_controleur sim_pick_place.launch.py \\
      pick_x:=0.35 pick_y:=0.10 pick_z:=0.02 \\
      place_x:=0.40 place_y:=-0.15 place_z:=0.02
"""
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    # ── Argument principal : avec ou sans pick & place ───────────────
    with_pp_arg = DeclareLaunchArgument(
        'with_pick_place', default_value='true',
        description='false = simulation seule (Gazebo + MoveIt2 + RViz, sans pick & place)'
    )

    # ── Arguments coordonnées pick & place ───────────────────────────
    pick_x_arg  = DeclareLaunchArgument('pick_x',  default_value='0.4',
                                        description='X saisie (m, repère base robot)')
    pick_y_arg  = DeclareLaunchArgument('pick_y',  default_value='0.15',
                                        description='Y saisie (m)')
    pick_z_arg  = DeclareLaunchArgument('pick_z',  default_value='0.0',
                                        description='Z saisie (m)')
    place_x_arg = DeclareLaunchArgument('place_x', default_value='0.4',
                                        description='X dépôt (m)')
    place_y_arg = DeclareLaunchArgument('place_y', default_value='-0.15',
                                        description='Y dépôt (m)')
    place_z_arg = DeclareLaunchArgument('place_z', default_value='0.0',
                                        description='Z dépôt (m)')

    # ── 1. MoveIt2 + RViz + ros2_control ────────────────────────────
    # On inclut moveit_controller.launch.py DIRECTEMENT (pas via demo.launch.py)
    # pour pouvoir passer hardware_protocol, camera, mount, load_base, load_gazebo.
    # demo.launch.py ne forward que rviz_file → les autres args utilisent leurs
    # valeurs par défaut (camera=realsense → URDF cassé → robot_description vide).
    moveit_rviz = PathJoinSubstitution([
        FindPackageShare('igus_rebel_moveit_config'),
        'rviz', 'moveit.rviz',
    ])

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('igus_rebel_moveit_config'),
                'launch', 'moveit_controller.launch.py',
            ])
        ),
        launch_arguments={
            'hardware_protocol': 'ignition',    # contrôleurs via Gazebo
            'load_gazebo':       'true',         # active use_sim_time + frames Gazebo
            'end_effector':      'schunk_egp25',
            'mount':             'none',         # pas de support caméra sur le bras
            'camera':            'none',         # caméra sur support externe, pas sur le bras
            'load_base':         'false',        # bras seul, pas de base mobile
            'load_octomap':      'false',
            'rviz_file':         moveit_rviz,
        }.items(),
    )

    # ── 2. Gazebo Ignition — démarré 5 s après MoveIt pour que ros2_control_node
    #    soit enregistré avant que ign_ros2_control/IgnitionSystem tente de s'y connecter.
    #    Sans ce délai le plugin Ignition se connecte à un controller_manager pas encore
    #    prêt → transition zéro→position réelle → clignotement dans RViz.
    ignition_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('igus_rebel_gazebo_ignition'),
                        'launch', 'ignition.launch.py',
                    ])
                ),
                launch_arguments={
                    'moveit': 'true',
                }.items(),
            )
        ],
    )

    # ── 3. Nœuds pick & place — 25 s après le lancement (optionnel) ──────
    # 25 s = 5 s (délai Gazebo) + ~20 s (chargement monde SDF, spawn, bridge, move_group).
    # Sur machines lentes augmenter à 30 s si CONTROL_FAILED persiste.
    pick_place_nodes = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='mon_controleur',
                executable='securite',
                condition=IfCondition(LaunchConfiguration('with_pick_place')),
                name='securite',
                output='screen',
                emulate_tty=True,
                parameters=[{'use_sim_time': True}],
            ),
            Node(
                package='mon_controleur',
                executable='workspace_scene',
                name='workspace_scene',
                output='screen',
                condition=IfCondition(LaunchConfiguration('with_pick_place')),
                parameters=[{'use_sim_time': True, 'sim_mode': True}],
            ),
            Node(
                package='mon_controleur',
                executable='pick_place_ia',
                condition=IfCondition(LaunchConfiguration('with_pick_place')),
                name='pick_place_ia',
                output='screen',
                emulate_tty=True,
                parameters=[{
                    'use_sim_time': True,
                    'mode':      'static',
                    'pick_x':    LaunchConfiguration('pick_x'),
                    'pick_y':    LaunchConfiguration('pick_y'),
                    'pick_z':    LaunchConfiguration('pick_z'),
                    'place_x':   LaunchConfiguration('place_x'),
                    'place_y':   LaunchConfiguration('place_y'),
                    'place_z':   LaunchConfiguration('place_z'),
                    # Simulation : pas de danger physique → vitesse maximale
                    'vel_scale': 0.8,
                    'acc_scale': 0.4,
                }],
            ),
        ],
    )

    return LaunchDescription([
        with_pp_arg,
        pick_x_arg, pick_y_arg, pick_z_arg,
        place_x_arg, place_y_arg, place_z_arg,
        moveit_launch,
        ignition_launch,
        pick_place_nodes,
    ])
