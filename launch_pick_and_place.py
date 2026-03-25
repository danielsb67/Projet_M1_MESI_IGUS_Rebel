#!/usr/bin/env python3
"""
Lanceur automatique — Pick & Place Igus Rebel
Lance tous les terminaux nécessaires en une seule commande.

Usage:
    python3 launch_pick_and_place.py          → lance les 4 terminaux
    python3 launch_pick_and_place.py --stop   → arrête tout
"""

import subprocess
import sys
import os
import signal
import time

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — Adapte si besoin                          ║
# ╚══════════════════════════════════════════════════════════════╝

WORKSPACE = os.path.expanduser("~/projet_igus")

TERMINALS = [
    {
        "name": "1_Robot_MoveIt",
        "cmd": (
            f"cd {WORKSPACE} && source install/setup.bash && "
            "ros2 launch igus_rebel_moveit_config demo.launch.py "
            "load_base:=false camera:=none mount:=none end_effector:=none "
            "hardware_protocol:=cri"
        ),
        "wait_msg": "Attends 'You can start planning now!' puis passe au suivant",
    },
    {
        "name": "2_Securite",
        "cmd": (
            "source /opt/ros/humble/setup.bash && "
            "ros2 topic pub -r 10 /safety/emergency_stop std_msgs/msg/Bool "
            '"{data: false}"'
        ),
        "wait_msg": None,
    },
    {
        "name": "3_Pick_and_Place",
        "cmd": (
            f"cd {WORKSPACE} && source install/setup.bash && "
            "ros2 run mon_controleur pick_and_place"
        ),
        "wait_msg": "Attends 'PRÊT' puis envoie une cible depuis Terminal 4",
    },
]

# ╔══════════════════════════════════════════════════════════════╗
# ║  FONCTIONS                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

PIDFILE = "/tmp/pick_and_place_pids.txt"


def launch_terminal(name, bash_cmd):
    """Ouvre un gnome-terminal avec le nom et la commande donnés."""
    full_cmd = f'bash -c \'{bash_cmd}; echo ""; echo "=== {name} TERMINÉ — Appuie ENTRÉE pour fermer ==="; read\''
    proc = subprocess.Popen([
        "gnome-terminal",
        f"--title={name}",
        "--", "bash", "-c", full_cmd
    ])
    return proc.pid


def stop_all():
    """Arrête tous les terminaux lancés."""
    if not os.path.exists(PIDFILE):
        print("Aucun processus à arrêter.")
        return

    # Kill tous les process ROS2 liés
    print("Arrêt de tous les processus...")
    os.system("pkill -f 'ros2 launch igus_rebel_moveit_config'")
    os.system("pkill -f 'ros2 topic pub.*emergency_stop'")
    os.system("pkill -f 'ros2 run mon_controleur pick_and_place'")
    time.sleep(1)
    os.remove(PIDFILE)
    print("✅ Tout est arrêté.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        stop_all()
        return

    print("=" * 60)
    print("  LANCEUR PICK & PLACE — IGUS REBEL")
    print("=" * 60)
    print()

    pids = []

    for i, term in enumerate(TERMINALS):
        print(f"🚀 Lancement : {term['name']}")
        pid = launch_terminal(term["name"], term["cmd"])
        pids.append(pid)

        if term.get("wait_msg"):
            print(f"   ⏳ {term['wait_msg']}")
            input("   Appuie ENTRÉE quand c'est prêt...")
        else:
            time.sleep(2)

        print(f"   ✅ {term['name']} lancé")
        print()

    # Sauvegarder les PIDs
    with open(PIDFILE, "w") as f:
        for pid in pids:
            f.write(f"{pid}\n")

    print("=" * 60)
    print("  ✅ TOUT EST LANCÉ !")
    print("=" * 60)
    print()
    print("  Pour envoyer une cible, copie dans un nouveau terminal :")
    print()
    print("  source /opt/ros/humble/setup.bash")
    print('  ros2 topic pub --once /yolo/target_coordinates \\')
    print('    geometry_msgs/msg/PointStamped \\')
    print('    "{header: {frame_id: \'igus_rebel_base_link\'}, \\')
    print('     point: {x: 0.30, y: 0.0, z: 0.10}}"')
    print()
    print("  Pour tout arrêter : python3 launch_pick_and_place.py --stop")
    print()


if __name__ == "__main__":
    main()
