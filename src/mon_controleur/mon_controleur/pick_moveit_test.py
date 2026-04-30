#!/usr/bin/env python3
"""
Pick & Place TEST — utilise l'action /move_group (STOMP/OMPL).
Plus robuste que compute_ik : gère IK + planification + exécution en un seul appel.

Pré-requis :
  ros2 launch igus_rebel_moveit_config demo.launch.py \
      hardware_protocol:=cri end_effector:=schunk_egp25 mount:=none camera:=none load_base:=false
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import SetBool
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, BoundingVolume, JointConstraint,
    RobotState,
)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    Constraints, PositionConstraint, BoundingVolume, JointConstraint,
    RobotState, OrientationConstraint # <--- Ajout ici
)

# ── Coordonnées de l'objet test ────────────────────────────────────
OBJ_X = 0.28
OBJ_Y = 0.0
OBJ_Z = 0.25

PREGRASP_OFFSET = 0.06
LIFT_OFFSET     = 0.08

# ── Config robot ───────────────────────────────────────────────────
BASE_FRAME  = "igus_rebel_base_link"
ARM_GROUP   = "rebel_arm"
EE_LINK     = "effecteur_simplifie_link"
GRIPPER_SRV = "/gripper/command"

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ──────────────────────────────────────────────────────────────────
class PickPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_moveit_test")
        self._mg      = ActionClient(self, MoveGroup, "/move_action")
        # On renomme self._gripper en self._gripper_cli
        self._gripper_cli = self.create_client(SetBool, GRIPPER_SRV) 
        self._current_joints = list(HOME_JOINTS)

    # ── Spin helper ───────────────────────────────────────────────
    def _wait(self, future, timeout=30.0):
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result() if future.done() else None

    # ── Construction d'un goal de pose cartésienne ────────────────
    def _pose_goal(self, x, y, z) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = ARM_GROUP
        req.num_planning_attempts           = 10
        req.allowed_planning_time           = 8.0
        req.max_velocity_scaling_factor     = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # Espace de travail
        wp = req.workspace_parameters
        wp.header.frame_id = BASE_FRAME
        wp.min_corner.x, wp.min_corner.y, wp.min_corner.z = -1.0, -1.0, -0.5
        wp.max_corner.x, wp.max_corner.y, wp.max_corner.z =  1.0,  1.0,  2.0

        

        # Contrainte de position (sphère 1 cm autour de la cible)
        pos = PositionConstraint()
        pos.header.frame_id = BASE_FRAME
        pos.link_name       = EE_LINK
        pos.weight          = 1.0

        sphere             = SolidPrimitive()
        sphere.type        = SolidPrimitive.SPHERE
        sphere.dimensions  = [0.05]

        target             = Pose()
        target.position.x  = float(x)
        target.position.y  = float(y)
        target.position.z  = float(z)
        target.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives      = [sphere]
        bv.primitive_poses = [target]
        pos.constraint_region = bv


        # --- NOUVEAU : Contrainte d'orientation ---
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name       = EE_LINK
        oc.orientation.w   = 1.0  # Orientation de base neutre
        # Tolérance de 3.14 radians (180°) sur tous les axes = MoveIt peut choisir 
        # n'importe quelle orientation pourvu qu'il atteigne la position (x,y,z).
        oc.absolute_x_axis_tolerance = 3.14
        oc.absolute_y_axis_tolerance = 3.14
        oc.absolute_z_axis_tolerance = 3.14
        oc.weight          = 1.0


        c = Constraints()
        c.position_constraints = [pos]
        req.goal_constraints   = [c]

        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3
        return goal

    # ── Construction d'un goal articulaire (HOME) ─────────────────
    def _joint_goal(self, positions) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name            = ARM_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor     = 0.3
        req.max_acceleration_scaling_factor = 0.3

        c = Constraints()
        for name, pos in zip(JOINT_NAMES, positions):
            jc                = JointConstraint()
            jc.joint_name     = name
            jc.position       = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]

        goal.planning_options.plan_only = False
        return goal

    # ── Envoi d'un goal MoveGroup ─────────────────────────────────
    def _send(self, goal, label: str) -> bool:
        self.get_logger().info(f"  → {label}")
        gh = self._wait(self._mg.send_goal_async(goal), timeout=10.0)
        if gh is None or not gh.accepted:
            self.get_logger().warn(f"  ✗ Goal rejeté : {label}")
            return False

        res = self._wait(gh.get_result_async(), timeout=40.0)
        if res is None:
            self.get_logger().warn(f"  ✗ Timeout : {label}")
            return False

        code = res.result.error_code.val
        if code == 1:
            # Mettre à jour l'état articulaire depuis la trajectoire planifiée
            traj = res.result.planned_trajectory
            if traj.joint_trajectory.points:
                self._current_joints = list(
                    traj.joint_trajectory.points[-1].positions
                )
            self.get_logger().info(f"  ✓ {label}")
            return True

        self.get_logger().warn(f"  ✗ {label} — erreur MoveIt code {code}")
        return False

    # ── Pince ─────────────────────────────────────────────────────
    def _gripper(self, close: bool):
        self.get_logger().info(f"  Pince : {'FERMER' if close else 'OUVRIR'}")
        # Utilise le nouveau nom ici
        if not self._gripper_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("  Service pince indisponible.")
            return
        req = SetBool.Request()
        req.data = close
        # Et ici aussi
        self._wait(self._gripper_cli.call_async(req), timeout=3.0)
        time.sleep(0.8)

    # ── Cycle principal ───────────────────────────────────────────
    def run(self):
        log = self.get_logger()
        x, y, z = OBJ_X, OBJ_Y, OBJ_Z

        log.info("Attente du serveur /move_group...")
        if not self._mg.wait_for_server(timeout_sec=15.0):
            log.error("/move_group non disponible — move_group lancé ?")
            return

        log.info("=" * 52)
        log.info(f"  PICK & PLACE TEST — objet ({x:.2f}, {y:.2f}, {z:.2f}) m")
        log.info("=" * 52)

        self._gripper(close=False)

        if not self._send(self._pose_goal(x, y, z + PREGRASP_OFFSET), "PREGRASP"):
            log.error("PREGRASP échoué — arrêt.")
            return

        if not self._send(self._pose_goal(x, y, z), "GRASP"):
            log.error("GRASP échoué — arrêt.")
            return

        self._gripper(close=True)

        self._send(self._pose_goal(x, y, z + LIFT_OFFSET), "LIFT")

        self._send(self._joint_goal(HOME_JOINTS), "HOME")

        self._gripper(close=False)

        log.info("=" * 52)
        log.info("  CYCLE TERMINE AVEC SUCCES")
        log.info("=" * 52)


# ──────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = PickPlaceNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()