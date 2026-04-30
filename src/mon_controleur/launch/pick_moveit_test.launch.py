"""
Lance pick_moveit_test.
La config MoveIt2 est chargée directement dans le nœud Python (MoveItConfigsBuilder),
donc aucun params-file n'est nécessaire ici.

Pré-requis matériel/sim déjà lancés séparément :
  ros2 launch igus_rebel_moveit_config moveit_controller.launch.py end_effector:=schunk_egp25 load_base:=false
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pick_node = Node(
        package="mon_controleur",
        executable="pick_moveit_test",
        output="screen",
    )
    return LaunchDescription([pick_node])