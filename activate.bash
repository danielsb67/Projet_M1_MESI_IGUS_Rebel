#!/usr/bin/env bash
# Script d'activation du workspace — à sourcer dans chaque terminal :
#   source ~/projet_igus/activate.bash

# 1. Détection réseau : ethernet robot UP avec IP → labo, sinon → loopback
if ip addr show enp1s0 2>/dev/null | grep -q "inet "; then
    export CYCLONEDDS_URI=file://$HOME/cyclone.xml
    echo "[ROS] Mode LABO — réseau robot (enp1s0)"
else
    export CYCLONEDDS_URI=file://$HOME/cyclone_home.xml
    echo "[ROS] Mode MAISON — WiFi local (wlp0s20f3)"
fi

# 2. ROS2 base + workspace
source /opt/ros/humble/setup.bash
source /home/dbal/projet_igus/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=10

echo "[WS]  projet_igus activé — CYCLONEDDS_URI=$CYCLONEDDS_URI"
