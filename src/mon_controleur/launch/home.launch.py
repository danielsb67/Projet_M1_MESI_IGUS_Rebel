"""
Retour HOME en une commande
============================
Lance ensemble :
  • workspace_scene — publie les zones interdites (caméra, profilé, murs)
                      dans la planning scene de MoveIt.
  • go_homez        — envoie le robot en position HOME via MoveIt.

go_homez est démarré avec ~6 s de retard, le temps que workspace_scene
publie la cage de collision (il attend lui-même 3 s le démarrage de MoveIt).
Quand go_homez a terminé, tout le launch se ferme automatiquement.

Pré-requis (déjà lancé dans un autre terminal) :
  ros2 launch igus_rebel_moveit_config demo.launch.py \
      hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none camera:=none load_base:=false

Lancement :
  ros2 launch mon_controleur home.launch.py
"""
from launch import LaunchDescription
from launch.actions import TimerAction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    # 1. Publie les zones interdites dans la planning scene de MoveIt.
    workspace = Node(
        package="mon_controleur",
        executable="workspace_scene",
        output="screen",
    )

    # 2. Envoie le robot en HOME — retardé pour laisser la cage se publier.
    go_home = Node(
        package="mon_controleur",
        executable="go_homez",
        output="screen",
    )
    go_home_retarde = TimerAction(period=6.0, actions=[go_home])

    # 3. Dès que go_homez a fini, on ferme tout le launch.
    arret_a_la_fin = RegisterEventHandler(
        OnProcessExit(
            target_action=go_home,
            on_exit=[EmitEvent(event=Shutdown())],
        )
    )

    return LaunchDescription([workspace, go_home_retarde, arret_a_la_fin])
