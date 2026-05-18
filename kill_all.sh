#!/bin/bash
# Kill propre de tous les processus ROS2 + Gazebo Ignition résiduels.
# À lancer avant CHAQUE relance pour éviter les conflits de plugin gz_ros2_control.

echo "→ Kill ROS2 nodes..."
pkill -9 -f ros2 2>/dev/null
pkill -9 -f rviz2 2>/dev/null
pkill -9 -f move_group 2>/dev/null
pkill -9 -f robot_state_publisher 2>/dev/null
pkill -9 -f controller_manager 2>/dev/null
pkill -9 -f spawner 2>/dev/null

echo "→ Kill Gazebo / Ignition..."
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f ign-gazebo 2>/dev/null
pkill -9 -f ruby 2>/dev/null
pkill -9 -f gz-sim 2>/dev/null
pkill -9 -f gazebo 2>/dev/null
pkill -9 -f parameter_bridge 2>/dev/null
pkill -9 -f ros_gz 2>/dev/null

echo "→ Attente 3 sec pour libération des sockets DDS..."
sleep 3

echo "→ Vérification process restants :"
pgrep -af "ros2|gazebo|ign|move_group|rviz" || echo "  (aucun process résiduel — OK)"

echo "✓ Cleanup terminé. Vous pouvez relancer."
