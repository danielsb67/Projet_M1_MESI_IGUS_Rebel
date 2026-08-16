#!/usr/bin/env python3
"""
preview_randomization.launch.py — APERÇU visuel de la randomisation pick.

But : ouvrir Gazebo et voir la roulette se téléporter TRÈS VITE à des positions
randomisées, SANS exécuter de cycle pick&place. Permet de valider d'un coup d'œil
la distribution des positions (anneau 0.15–0.54 m + exclusion du bac) avant de
lancer une vraie collecte de N épisodes.

Ce qu'on lance :
  - sim + robot + caméra (réutilise check_robot_in_world.launch.py) ;
  - gripper_shim         : téléporte l'objet à chaque position reçue ;
  - record_orchestrator  : en mode preview_only → boucle rapide de positions
                           randomisées publiées sur /object_position_in_world.

Ce qu'on NE lance PAS (vs record_demos) : l'expert pick_place_ia et le recorder.
Aucune donnée n'est écrite.

Lancement :
  ros2 launch igus_vla preview_randomization.launch.py
  ros2 launch igus_vla preview_randomization.launch.py preview_period:=0.3 preview_count:=50
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
        # Cadence et nombre de positions de l'aperçu
        DeclareLaunchArgument("preview_period", default_value="0.5",
                              description="secondes entre deux téléportations"),
        DeclareLaunchArgument("preview_count", default_value="0",
                              description="nombre de positions (0 = infini, stoppe à la main)"),
        DeclareLaunchArgument("preview_settle", default_value="12.0",
                              description="délai avant la 1re position (démarrage sim/shim)"),
        # Place (centre du bac) — sert à l'exclusion de zone dans le tirage
        DeclareLaunchArgument("place_x", default_value="0.0"),
        DeclareLaunchArgument("place_y", default_value="0.25"),
        # Sim / shim
        DeclareLaunchArgument("world_name", default_value="default"),
        DeclareLaunchArgument("object_model", default_value="roulette"),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration("headless").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)
    object_model = LaunchConfiguration("object_model").perform(context)

    preview_period = float(LaunchConfiguration("preview_period").perform(context))
    preview_count = int(LaunchConfiguration("preview_count").perform(context))
    preview_settle = float(LaunchConfiguration("preview_settle").perform(context))
    place_x = float(LaunchConfiguration("place_x").perform(context))
    place_y = float(LaunchConfiguration("place_y").perform(context))

    # ── 1. Bring-up sim (Gazebo + robot + caméra) ──
    include_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(SIM_LAUNCH_PATH),
        launch_arguments={"headless": headless}.items(),
    )

    # ── 2. gripper_shim : téléporte la roulette aux positions reçues ──
    gripper_shim_node = Node(
        package="igus_vla",
        executable="gripper_shim",
        parameters=[{
            "world_name": world_name,
            "object_model": object_model,
            "object_z": 0.018,
            # position initiale = zone sûre en y négatif (loin du bac)
            "object_x": 0.35,
            "object_y": -0.20,
        }],
        output="screen",
    )

    # ── 3. Orchestrateur en mode APERÇU (pas d'expert, pas de recorder) ──
    preview_node = Node(
        package="igus_vla",
        executable="record_orchestrator",
        parameters=[{
            "preview_only": True,
            "preview_period": preview_period,
            "preview_count": preview_count,
            "preview_settle": preview_settle,
            "randomize": True,
            "object_z": 0.018,
            "place_x": place_x,
            "place_y": place_y,
        }],
        output="screen",
    )

    return [include_sim, gripper_shim_node, preview_node]
