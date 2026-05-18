#!/usr/bin/env python3
"""
Retour HOME sécurisé — Igus Rebel
==================================
Envoie le robot en position HOME en passant par MoveIt (action /move_action).

Contrairement à un FollowJointTrajectory direct (qui interpole bêtement les
joints et peut balayer le profilé / la caméra), MoveIt vérifie la planning
scene : la trajectoire évite les zones interdites publiées par le nœud
workspace_scene (sol, plafond, murs et profilé protégeant la caméra).

Stratégie :
  1. Pilz PTP  — mouvement direct, REFUSÉ si le chemin traverse une zone interdite.
  2. STOMP     — repli : contourne réellement l'obstacle si le PTP a échoué.

Pré-requis (terminaux séparés) :
  ros2 launch igus_rebel_moveit_config demo.launch.py \
      hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none camera:=none load_base:=false
  ros2 run mon_controleur workspace_scene
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes

ARM_GROUP   = "rebel_arm"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# Position HOME (6 axes, radians) — bras vertical, hors des zones interdites.
POSITION_HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Mouvement doux (mêmes échelles que les déplacements HOME de pick_place_ia).
VEL_SCALE = 0.25
ACC_SCALE = 0.15


class RetourHome(Node):
    def __init__(self):
        super().__init__("go_home")
        self._mg = ActionClient(self, MoveGroup, "/move_action")

    # ------------------------------------------------------------------
    def _wait(self, future, timeout):
        """Attend la complétion d'un future en faisant tourner le nœud."""
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    # ------------------------------------------------------------------
    def _joint_goal(self, positions, pipeline, planner) -> MoveGroup.Goal:
        """Construit un goal articulaire MoveGroup (avec contrôle de collision)."""
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 3
        req.allowed_planning_time           = 15.0
        req.max_velocity_scaling_factor     = VEL_SCALE
        req.max_acceleration_scaling_factor = ACC_SCALE
        req.pipeline_id                     = pipeline
        req.planner_id                      = planner

        c = Constraints()
        for name, pos in zip(JOINT_NAMES, positions):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]

        goal.planning_options.plan_only = False
        return goal

    # ------------------------------------------------------------------
    def _send(self, goal, label) -> bool:
        """Envoie le goal et attend le résultat. True si succès."""
        gh = self._wait(self._mg.send_goal_async(goal), timeout=15.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"✗ Goal rejeté : {label}")
            return False

        result = self._wait(gh.get_result_async(), timeout=90.0)
        if result is None:
            self.get_logger().warn(f"✗ Timeout : {label} — annulation")
            gh.cancel_goal_async()
            return False

        code = result.result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"✓ {label} réussi")
            return True

        self.get_logger().warn(f"✗ {label} — code MoveIt {code}")
        return False

    # ------------------------------------------------------------------
    def aller_home(self) -> bool:
        # 1) Pilz PTP : mouvement direct. MoveIt le refuse si le chemin
        #    traverse une zone interdite (caméra / profilé / murs).
        self.get_logger().info("Planification HOME (Pilz PTP)...")
        if self._send(self._joint_goal(POSITION_HOME,
                                       "pilz_industrial_motion_planner", "PTP"),
                       "HOME (PTP)"):
            return True

        # 2) Repli STOMP : contourne réellement l'obstacle.
        self.get_logger().warn(
            "PTP impossible (zone interdite sur le chemin ?) — repli STOMP..."
        )
        return self._send(self._joint_goal(POSITION_HOME, "stomp", "STOMP"),
                           "HOME (STOMP)")


def main():
    # 1. Initialisation de ROS 2
    rclpy.init()

    # 2. Création du nœud de contrôle
    robot = RetourHome()

    try:
        robot.get_logger().info("Recherche du serveur MoveIt /move_action...")

        # 3. Vérification de la connexion à MoveIt
        if not robot._mg.wait_for_server(timeout_sec=10.0):
            robot.get_logger().error(
                "/move_action introuvable — as-tu bien lancé 'demo.launch.py' ?"
            )
            return

        robot.get_logger().info(
            "✅ MoveIt connecté ! Retour HOME (trajectoire vérifiée vs zones interdites)..."
        )

        # 4. Envoi de la commande HOME via MoveIt
        if robot.aller_home():
            robot.get_logger().info("🏁 Le robot est en position HOME !")
        else:
            robot.get_logger().error(
                "Échec du retour HOME. Vérifie que 'workspace_scene' tourne "
                "et que la position HOME est atteignable."
            )

    except KeyboardInterrupt:
        # Interception propre du Ctrl+C
        robot.get_logger().info("Arrêt d'urgence demandé (Ctrl+C).")
    finally:
        # Nettoyage propre sans double-shutdown
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
