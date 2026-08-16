#!/usr/bin/env python3
"""
check_robot_in_world.launch.py — ÉTAPE 2 (vérification visuelle).

But : faire apparaître le robot igus ReBeL dans le monde VLA
(world_with_camera.sdf) pour valider :
  - le cadrage de la caméra fixe externe AVEC le bras dans le champ ;
  - que la sim tourne (spawn URDF + ign_ros2_control + /joint_states) avec ce monde.

Différences clés vs ignition.launch.py d'origine :
  - charge NOTRE monde (chemin absolu, relatif à ce fichier — pas besoin de build) ;
  - spawne le robot à l'ORIGINE (0,0,0) yaw=0  → repère monde ≈ repère base,
    cohérent avec l'objet roulette en (0.4, 0.15) et les coords de l'expert ;
  - bridge la caméra front (+ /clock) via gazebo/bridge_front_camera.yaml.

Réutilise moveit_controller.launch.py pour move_group + controller_manager +
robot_state_publisher + spawners (mêmes args que sim_pick_place.launch.py).

Lancement (sans build, par chemin) :
  ros2 launch src/igus_vla/launch/check_robot_in_world.launch.py
  ros2 launch src/igus_vla/launch/check_robot_in_world.launch.py headless:=true   # sans GUI
"""
import os
from os import environ

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Chemins des assets VLA, relatifs à CE fichier (le package n'a pas besoin d'être buildé)
_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)               # src/igus_vla
_GAZEBO_DIR = os.path.join(_PKG, "gazebo")
WORLD_PATH = os.path.join(_GAZEBO_DIR, "world_with_camera.sdf")
BRIDGE_PATH = os.path.join(_GAZEBO_DIR, "bridge_front_camera.yaml")
GUI_CONFIG_PATH = os.path.join(_GAZEBO_DIR, "gui_vla.config")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false",
                              description="true = serveur Gazebo seul (sans GUI)"),
        DeclareLaunchArgument("spawn_x", default_value="0.0"),
        DeclareLaunchArgument("spawn_y", default_value="0.0"),
        DeclareLaunchArgument("spawn_z", default_value="0.0"),
        DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration("headless").perform(context) == "true"

    # ── Resource paths Ignition : permettre la résolution des meshes model://...
    #    du robot (igus_rebel_description_ros2). Réplique ignition.launch.py.
    description_parent = os.path.dirname(
        get_package_share_directory("igus_rebel_description_ros2"))
    description_share = get_package_share_directory("igus_rebel_description_ros2")
    existing_resource = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    existing_gz = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_paths = ":".join(filter(None, [existing_resource,
                                            description_parent, description_share]))
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = resource_paths
    os.environ["GZ_SIM_RESOURCE_PATH"] = ":".join(filter(None, [existing_gz, resource_paths]))

    ign_env = {
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": ":".join([
            environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", default=""),
            environ.get("LD_LIBRARY_PATH", default=""),
        ]),
    }

    # ── 1. MoveIt2 + ros2_control + RSP + spawners (mêmes args que sim_pick_place) ──
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("igus_rebel_moveit_config"),
                "launch", "moveit_controller.launch.py",
            ])
        ),
        launch_arguments={
            "hardware_protocol": "ignition",
            "load_gazebo":       "true",
            "end_effector":      "schunk_egp25",
            "mount":             "none",
            "camera":            "none",
            "load_base":         "false",
            "load_octomap":      "false",
            "rviz_file":         "none",
        }.items(),
    )

    # ── 2. Gazebo + spawn robot + bridge, démarrés +5 s (laisser controller_manager prêt) ──
    if headless:
        ign_cmd = ["ign gazebo", "--verbose 1 -r -s", WORLD_PATH]
    else:
        ign_cmd = ["ign gazebo", "--verbose 1 -r --gui-config " + GUI_CONFIG_PATH, WORLD_PATH]

    ign_sim = ExecuteProcess(
        cmd=ign_cmd, output="log", additional_env=ign_env, shell=True,
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "/robot_description",
            "-name", "igus_rebel",
            "-x", LaunchConfiguration("spawn_x"),
            "-y", LaunchConfiguration("spawn_y"),
            "-z", LaunchConfiguration("spawn_z"),
            "-Y", LaunchConfiguration("spawn_yaw"),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "use_sim_time": True,
            "config_file": BRIDGE_PATH,
            "qos_overrides./tf_static.publisher.durability": "transient_local",
        }],
        output="screen",
    )

    # ── Bridge SERVICE set_pose : expose le /world/default/set_pose d'Ignition en
    #    service ROS (ros_gz_interfaces/SetEntityPose). Le gripper_shim l'appelle en
    #    call_async (~2 ms, non bloquant) pour la saisie cinématique — REMPLACE l'ancien
    #    subprocess `ign service` (~375 ms ⇒ l'objet traînait loin derrière la pince). ──
    set_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        # NB : format SIMPLIFIÉ (type ROS seul) — spécifier les types gz explicites
        # (@ignition.msgs.Pose@…Boolean) casse le parser du bridge (service non créé).
        arguments=[
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    gazebo_stack = TimerAction(
        period=5.0, actions=[ign_sim, spawn_robot, bridge, set_pose_bridge])

    return [moveit_launch, gazebo_stack]
