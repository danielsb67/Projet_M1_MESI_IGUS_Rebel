#!/usr/bin/env python3
"""
visibility_sweep.launch.py — Mesure EMPIRIQUE de la zone visible par la caméra.

Ouvre la sim, balaie une grille de positions de roulette le plus vite possible
(cadence = vitesse caméra, pas d'attente à l'œil), détecte automatiquement si la
roulette est visible dans /front_camera/image, et écrit un CSV + une carte +
une proposition de zones à exclure.

Ce qu'on lance :
  - sim + robot + caméra (check_robot_in_world.launch.py) ;
  - gripper_shim     : téléporte la roulette aux positions reçues ;
  - visibility_sweep : balaie la grille, détecte, rapporte.

PAS d'expert, PAS de recorder. Aucune donnée d'entraînement écrite (juste le CSV
de visibilité sous datasets/visibility/).

Lancement :
  ros2 launch igus_vla visibility_sweep.launch.py
  ros2 launch igus_vla visibility_sweep.launch.py step:=0.05 headless:=true
"""
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.realpath(__file__))
SIM_LAUNCH_PATH = os.path.join(_HERE, "check_robot_in_world.launch.py")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false",
                              description="true = serveur Gazebo seul (sans GUI)"),
        DeclareLaunchArgument("step", default_value="0.06",
                              description="pas de la grille de balayage (m)"),
        DeclareLaunchArgument("settle", default_value="12.0",
                              description="délai démarrage sim avant le balayage (s)"),
        DeclareLaunchArgument("fresh_frames", default_value="2",
                              description="frames fraîches attendues par position"),
        DeclareLaunchArgument("pixel_thresh", default_value="60",
                              description="pixels orange min pour déclarer 'visible'"),
        DeclareLaunchArgument("place_x", default_value="0.0"),
        DeclareLaunchArgument("place_y", default_value="0.25"),
        DeclareLaunchArgument("world_name", default_value="default"),
        DeclareLaunchArgument("object_model", default_value="roulette"),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration("headless").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)
    object_model = LaunchConfiguration("object_model").perform(context)
    step = float(LaunchConfiguration("step").perform(context))
    settle = float(LaunchConfiguration("settle").perform(context))
    fresh_frames = int(LaunchConfiguration("fresh_frames").perform(context))
    pixel_thresh = int(LaunchConfiguration("pixel_thresh").perform(context))
    place_x = float(LaunchConfiguration("place_x").perform(context))
    place_y = float(LaunchConfiguration("place_y").perform(context))

    include_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(SIM_LAUNCH_PATH),
        launch_arguments={"headless": headless}.items(),
    )

    gripper_shim_node = Node(
        package="igus_vla",
        executable="gripper_shim",
        parameters=[{
            "world_name": world_name,
            "object_model": object_model,
            "object_z": 0.018,
            "object_x": 0.35,
            "object_y": -0.20,
        }],
        output="screen",
    )

    sweep_node = Node(
        package="igus_vla",
        executable="visibility_sweep",
        parameters=[{
            "step": step,
            "settle": settle,
            "fresh_frames": fresh_frames,
            "pixel_thresh": pixel_thresh,
            "place_x": place_x,
            "place_y": place_y,
            "object_z": 0.018,
        }],
        output="screen",
    )

    return [include_sim, gripper_shim_node, sweep_node]
