#!/usr/bin/env python3
"""
Pick & Place MANUEL — Igus Rebel + Schunk EGP25
================================================
Chaque étape attend une confirmation clavier avant d'agir.

Avant de lancer :
  1. Relever les positions avec : ros2 topic echo /joint_states --once
  2. Remplir les 5 positions ci-dessous (en radians)
  3. Lancer ROS2 : ros2 launch igus_rebel_moveit_config demo.launch.py hardware_protocol:=cri ...
  4. Lancer ce script : python3 pick_place_manuel.py
"""

import sys
import time
import rclpy

from mon_controleur.gripper_control import SchunkGripper
from mon_controleur.bouge_robot import IgusRebelMover

# SchunkGripper utilise maintenant le service ROS2 /gripper/command
# exposé par le driver C++ — plus de connexion TCP concurrente.

# ─────────────────────────────────────────────────────────────────
#  POSITIONS ARTICULAIRES  (radians)  — À REMPLIR
#  Ordre des joints : joint1, joint2, joint3, joint4, joint5, joint6
# ─────────────────────────────────────────────────────────────────

HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Position ~10 cm au-dessus de l'objet à saisir
# Original (°): [0.0, 33, 71.6, 0.0, 70, 0.0]
APPROCHE_PICK = [0.0, 0.5760, 1.24, 0.0, 1.2217, 0.0]

# Position de saisie (pince sur l'objet)
# Original (°): [0.0, 38.6, 79.3, 0.0, 63.8, 0.0]
PICK_JOINTS = [0.0, 0.6737, 1.3840, 0.0, 1.1135, 0.0]

# Position ~10 cm au-dessus de la zone de dépôt
# Original (°): [-170, 33, 71.6, 0.0, 70, 0.0]
APPROCHE_PLACE = [-2.9671, 0.5760, 1.2497, 0.0, 1.2217, 0.0]

# Position de dépôt (pince au-dessus de la destination)
# Original (°): [-170.2, 42.2, 74.7, 0.0, 57.8, 0.0]
PLACE_JOINTS = [-2.9706, 0.7365, 1.3238, 0.0, 1.0188, 0.0]

# ─────────────────────────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────────────────────────

MOVE_DURATION = 3.0      # secondes par mouvement (lent = sécurisé)
GRIPPER_WAIT  = 1.1      # secondes laissées à la pince pour agir

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def pause(etape: str, detail: str = ""):
    """Affiche l'étape et attend une confirmation clavier."""
    print(f"\n{'─'*55}")
    print(f"  ÉTAPE : {etape}")
    if detail:
        print(f"  {detail}")
    print(f"{'─'*55}")
    try:
        rep = input("  Appuie sur ENTRÉE pour continuer  (q + ENTRÉE pour quitter) : ")
        if rep.strip().lower() == 'q':
            print("Arrêt demandé par l'utilisateur.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nCtrl+C — arrêt.")
        sys.exit(0)


def bouger(robot: IgusRebelMover, joints: list, label: str):
    """Envoie une trajectoire et lève une exception si le mouvement échoue."""
    print(f"  → Mouvement vers {label} ...")
    ok = robot.send_trajectory(joints, duration_sec=MOVE_DURATION)
    if not ok:
        raise RuntimeError(f"Mouvement échoué : {label}")
    print(f"  ✓ {label} atteint")


# ─────────────────────────────────────────────────────────────────
#  SCRIPT PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*55)
    print("   PICK & PLACE MANUEL — Igus Rebel + Schunk EGP25")
    print("═"*55)

    # ── Init ROS2 ──
    rclpy.init()
    robot = IgusRebelMover()

    # ── Init gripper (partage la connexion CRI du driver via service ROS2) ──
    gripper = SchunkGripper(node=robot)

    try:
        # ── Connexion bras ──
        print("\n[1/2] Connexion au contrôleur du bras...")
        if not robot.wait_for_server(timeout_sec=15.0):
            print("ERREUR : contrôleur non disponible.")
            return

        current = robot.get_current_positions()
        if not current:
            print("ERREUR : impossible de lire /joint_states.")
            return
        print(f"  Position actuelle : {[f'{p:.3f}' for p in current]}")

        # ── Connexion pince ──
        print("\n[2/2] Attente du service /gripper/command...")
        gripper.connect(timeout_sec=10.0)

        # ═══════════════════════════════════════════
        #  CYCLE PICK & PLACE
        # ═══════════════════════════════════════════

        # 1. HOME
        pause("1 / 8 — Aller à HOME")
        bouger(robot, HOME_JOINTS, "HOME")

        # 2. Ouvrir la pince
        pause("2 / 8 — Ouvrir la pince")
        gripper.ouvrir()
        time.sleep(GRIPPER_WAIT)
        print("  ✓ Pince ouverte")

        # 3. Approche pick
        pause("3 / 8 — Approche au-dessus de l'objet",
              f"Position : {APPROCHE_PICK}")
        bouger(robot, APPROCHE_PICK, "APPROCHE_PICK")

        # 4. Descente pick
        pause("4 / 8 — Descendre sur l'objet",
              f"Position : {PICK_JOINTS}")
        bouger(robot, PICK_JOINTS, "PICK")

        # 5. Fermer la pince
        pause("5 / 8 — Fermer la pince (saisir l'objet)")
        gripper.fermer()
        time.sleep(GRIPPER_WAIT)
        print("  ✓ Pince fermée")

        # 6. Remonter
        pause("6 / 8 — Remonter (retour approche pick)")
        bouger(robot, APPROCHE_PICK, "APPROCHE_PICK (retour)")

        # 7. Approche dépôt
        pause("7 / 8 — Aller au-dessus de la zone de dépôt",
              f"Position : {APPROCHE_PLACE}")
        bouger(robot, APPROCHE_PLACE, "APPROCHE_PLACE")

        # 8. Descente dépôt
        pause("8 / 8 — Descendre sur la zone de dépôt",
              f"Position : {PLACE_JOINTS}")
        bouger(robot, PLACE_JOINTS, "PLACE")

        # 9. Ouvrir la pince (déposer)
        pause("9 / 9 — Ouvrir la pince (déposer l'objet)")
        gripper.ouvrir()
        time.sleep(GRIPPER_WAIT)
        print("  ✓ Objet déposé")

        # 10. Retour HOME
        pause("10 / 10 — Retour HOME")
        bouger(robot, APPROCHE_PLACE, "APPROCHE_PLACE (retour)")
        bouger(robot, HOME_JOINTS, "HOME")

        print("\n" + "═"*55)
        print("   ✅ CYCLE TERMINÉ AVEC SUCCÈS")
        print("═"*55 + "\n")

    except RuntimeError as e:
        print(f"\n❌ ERREUR : {e}")

    except KeyboardInterrupt:
        print("\nArrêt par Ctrl+C.")

    finally:
        gripper.disconnect()
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()