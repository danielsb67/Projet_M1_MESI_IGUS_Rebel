#!/usr/bin/env python3
"""
Simulation Gazebo + Pick & Place CARTÉSIEN — sans robot réel, sans caméra.

Identique à sim_pick_place.launch.py mais lance `pick_place_cartesien` au lieu
de `pick_place_ia`. Les segments descente/montée pick/place sont exécutés en
ligne droite via /compute_cartesian_path (MoveIt). Les repositionnements
HOME→approche restent en joint space (Pilz PTP).

Utilisation :
  ws && ros2 launch mon_controleur sim_pick_place_cartesien.launch.py

Coordonnées personnalisées :
  ros2 launch mon_controleur sim_pick_place_cartesien.launch.py \\
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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    with_pp_arg = DeclareLaunchArgument(
        'with_pick_place', default_value='true',
        description='false = simulation seule (Gazebo + MoveIt2 + RViz)'
    )

    pick_x_arg  = DeclareLaunchArgument('pick_x',  default_value='0.4')
    pick_y_arg  = DeclareLaunchArgument('pick_y',  default_value='0.15')
    pick_z_arg  = DeclareLaunchArgument('pick_z',  default_value='0.0')
    place_x_arg = DeclareLaunchArgument('place_x', default_value='0.4')
    place_y_arg = DeclareLaunchArgument('place_y', default_value='-0.15')
    place_z_arg = DeclareLaunchArgument('place_z', default_value='0.0')

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
            'hardware_protocol': 'ignition',
            'load_gazebo':       'true',
            'end_effector':      'schunk_egp25',
            'mount':             'none',
            'camera':            'none',
            'load_base':         'false',
            'load_octomap':      'false',
            'rviz_file':         moveit_rviz,
        }.items(),
    )

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
                executable='pick_place_cartesien',
                condition=IfCondition(LaunchConfiguration('with_pick_place')),
                name='pick_place_cartesien',
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
