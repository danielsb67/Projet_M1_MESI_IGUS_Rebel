#!/usr/bin/env python3
"""
Simulation Gazebo + Robot Dance — démo chorégraphique.

Lance MoveIt + Gazebo Ignition + workspace_scene + robot_dance.

Utilisation :
  ros2 launch mon_controleur sim_robot_dance.launch.py
  ros2 launch mon_controleur sim_robot_dance.launch.py dance:=shapes
  ros2 launch mon_controleur sim_robot_dance.launch.py dance:=lemniscate
  ros2 launch mon_controleur sim_robot_dance.launch.py dance:=joint_dance
  ros2 launch mon_controleur sim_robot_dance.launch.py dance:=parametric
  ros2 launch mon_controleur sim_robot_dance.launch.py dance:=all   (défaut)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    dance_arg = DeclareLaunchArgument(
        'dance', default_value='all',
        description='Choreographie : shapes | lemniscate | joint_dance | parametric | all',
    )

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
                launch_arguments={'moveit': 'true'}.items(),
            )
        ],
    )

    dance_node = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='mon_controleur',
                executable='workspace_scene',
                name='workspace_scene',
                output='screen',
                parameters=[{'use_sim_time': True, 'sim_mode': True}],
            ),
            Node(
                package='mon_controleur',
                executable='robot_dance',
                name='robot_dance',
                output='screen',
                emulate_tty=True,
                parameters=[{
                    'use_sim_time': True,
                    'dance':     LaunchConfiguration('dance'),
                    'vel_scale': 0.6,
                    'acc_scale': 0.35,
                }],
            ),
        ],
    )

    return LaunchDescription([
        dance_arg,
        moveit_launch,
        ignition_launch,
        dance_node,
    ])
