#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          MASTER NODE — PICK AND PLACE — IGUS REBEL          ║
║                                                              ║
║  Architecture :                                              ║
║    - Reçoit coordonnées XYZ de YOLO (Collègue A)           ║
║    - Vérifie la sécurité (Collègue B)                       ║
║    - Calcule l'IK via MoveIt (/compute_ik)                  ║
║    - Exécute via rebel_arm_controller (FollowJointTraj)     ║
║                                                              ║
║  Prérequis :                                                 ║
║    Terminal 1 : ros2 launch igus_rebel_moveit_config         ║
║                 demo.launch.py hardware_protocol:=cri ...    ║
║    Terminal 2 : TF statique caméra → robot                  ║
║    Terminal 3 : python3 pick_and_place.py                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Messages ROS 2
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Quaternion
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration

# MoveIt 2 — Service IK
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState as MoveItRobotState

import time
import math


# ╔══════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — PLACEHOLDER : ADAPTE CES VALEURS       ║
# ╚══════════════════════════════════════════════════════════╝

# --- Robot ---
CONTROLLER_NAME = "rebel_arm_controller"
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
PLANNING_GROUP = "rebel_arm"  # PLACEHOLDER : vérifie avec ros2 param get /move_group robot_description_planning

# --- Topics d'entrée (tes collègues) ---
YOLO_TOPIC = "/yolo/target_coordinates"        # geometry_msgs/PointStamped (Collègue A)
SAFETY_TOPIC = "/safety/emergency_stop"         # std_msgs/Bool (Collègue B)

# --- Positions clés (en mètres, dans le repère igus_rebel_base_link) ---
# PLACEHOLDER : Adapte ces coordonnées à ton setup réel

# Position Home (angles articulaires connus et sûrs)
# Utilise la position actuelle de ton robot comme référence
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# PLACEHOLDER ↑ remplace par la position de repos de TON robot
# (celle que tu as lue avec ros2 topic echo /joint_states --once)

# Position d'approche : au-dessus de l'objet avant de descendre
APPROACH_HEIGHT_OFFSET = 0.10  # 10cm au-dessus de l'objet (en Z)

# Position de dépôt (où le robot lâche l'objet)
DROP_POSITION = {
    "x": 0.25,   # PLACEHOLDER : adapte X (mètres)
    "y": -0.20,  # PLACEHOLDER : adapte Y (mètres)
    "z": 0.10,   # PLACEHOLDER : adapte Z (mètres)
}

# Orientation de l'effecteur pour le pick (pince vers le bas)
# Quaternion : pointe vers le bas (rotation de 180° autour de X)
# PLACEHOLDER : adapte si ton gripper a une orientation différente
PICK_ORIENTATION = Quaternion(x=0.797, y=0.008, z=-0.010, w=0.603)

# --- Vitesse ---
MOVE_DURATION_FAST = 25.0    # secondes pour mouvements normaux
MOVE_DURATION_SLOW = 28.0    # secondes pour approche/pick (plus prudent)

# --- Sécurité ---
WORKSPACE_LIMITS = {
    "x_min": 0.05,  "x_max": 0.50,   # PLACEHOLDER : limites en X (mètres)
    "y_min": -0.35, "y_max": 0.35,    # PLACEHOLDER : limites en Y
    "z_min": -0.18,  "z_max": 0.55,    # PLACEHOLDER : limites en Z (z_min > 0 pour pas taper la table)
}


# ╔══════════════════════════════════════════════════════════╗
# ║  MASTER NODE                                             ║
# ╚══════════════════════════════════════════════════════════╝

class PickAndPlaceMaster(Node):
    def __init__(self):
        super().__init__("pick_and_place_master")
        self.get_logger().info("=" * 60)
        self.get_logger().info("  MASTER NODE — PICK AND PLACE — Démarrage")
        self.get_logger().info("=" * 60)

        self.callback_group = ReentrantCallbackGroup()

        # ── État interne ──
        self._current_joint_positions = None
        self._emergency_stop = False
        self._target_received = False
        self._target_point = None
        self._is_busy = False  # True pendant un cycle pick&place

        # ── Subscriber : positions articulaires actuelles ──
        self.create_subscription(
            JointState, "/joint_states",
            self._joint_state_cb, 10,
            callback_group=self.callback_group
        )

        # ── Subscriber : coordonnées YOLO (Collègue A) ──
        self.create_subscription(
            PointStamped, YOLO_TOPIC,
            self._yolo_target_cb, 10,
            callback_group=self.callback_group
        )
        self.get_logger().info(f"  Écoute YOLO sur : {YOLO_TOPIC}")

        # ── Subscriber : arrêt d'urgence (Collègue B) ──
        self.create_subscription(
            Bool, SAFETY_TOPIC,
            self._safety_cb, 10,
            callback_group=self.callback_group
        )
        self.get_logger().info(f"  Écoute Sécurité sur : {SAFETY_TOPIC}")

        # ── Action Client : FollowJointTrajectory (déjà éprouvé) ──
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            f"/{CONTROLLER_NAME}/follow_joint_trajectory",
            callback_group=self.callback_group
        )

        # ── Service Client : MoveIt Compute IK ──
        self._ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=self.callback_group
        )

        self.get_logger().info("  En attente des services...")
        self._wait_for_services()
        self.get_logger().info("=" * 60)
        self.get_logger().info("  PRÊT — En attente de détection YOLO")
        self.get_logger().info("=" * 60)

    # ──────────────────────────────────────────────
    #  CALLBACKS
    # ──────────────────────────────────────────────

    def _joint_state_cb(self, msg):
        """Met à jour les positions articulaires actuelles."""
        self._current_joint_positions = dict(zip(msg.name, msg.position))

    def _safety_cb(self, msg):
        """Reçoit l'état de l'arrêt d'urgence."""
        if msg.data and not self._emergency_stop:
            self.get_logger().error("⚠️  ARRÊT D'URGENCE ACTIVÉ !")
        if not msg.data and self._emergency_stop:
            self.get_logger().info("✅ Arrêt d'urgence désactivé — reprise possible")
        self._emergency_stop = msg.data

    def _yolo_target_cb(self, msg):
        """Reçoit les coordonnées de l'objet détecté par YOLO."""
        if self._is_busy:
            self.get_logger().info("Cycle en cours, détection ignorée.")
            return

        x, y, z = msg.point.x, msg.point.y, msg.point.z
        frame = msg.header.frame_id
        self.get_logger().info(f"📦 Détection YOLO : x={x:.3f} y={y:.3f} z={z:.3f} (repère: {frame})")

        # NOTE : Si frame_id = "camera_link", il faudrait transformer
        # les coordonnées vers "igus_rebel_base_link" via TF2.
        # Pour l'instant, on suppose que les coordonnées sont DÉJÀ
        # dans le repère du robot (ou transformées par le collègue).
        # Voir _transform_to_base_frame() pour la version avec TF2.

        self._target_point = msg.point
        self._target_received = True

    # ──────────────────────────────────────────────
    #  INITIALISATION
    # ──────────────────────────────────────────────

    def _wait_for_services(self):
        """Attend que les services soient disponibles."""
        # Action FollowJointTrajectory
        self.get_logger().info("  Attente du controller...")
        if not self._traj_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error("Controller non disponible !")
            raise RuntimeError("Controller non disponible")
        self.get_logger().info("  ✅ Controller connecté")

        # Service IK MoveIt
        self.get_logger().info("  Attente du service /compute_ik (MoveIt)...")
        if not self._ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error(
                "/compute_ik non disponible !\n"
                "Vérifie que move_group tourne (pas de crash libgeometric_shapes).\n"
                "Fix : sudo apt install ros-humble-geometric-shapes"
            )
            raise RuntimeError("/compute_ik non disponible")
        self.get_logger().info("  ✅ MoveIt IK connecté")

    # ──────────────────────────────────────────────
    #  INVERSE KINEMATICS (XYZ → angles joints)
    # ──────────────────────────────────────────────

    def compute_ik(self, x, y, z, orientation=None):
        """
        Calcule les angles articulaires pour atteindre la position (x, y, z).
        Utilise le service /compute_ik de MoveIt 2.

        Retourne : liste de 6 angles [j1, j2, j3, j4, j5, j6] ou None si échec
        """
        if orientation is None:
            orientation = PICK_ORIENTATION

        request = GetPositionIK.Request()

        # Construire la requête IK
        ik_request = PositionIKRequest()
        ik_request.group_name = PLANNING_GROUP
        ik_request.avoid_collisions = True

        # Pose cible
        pose = PoseStamped()
        pose.header.frame_id = "igus_rebel_base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation = orientation
        ik_request.pose_stamped = pose

        # État actuel du robot (seed pour l'IK)
        robot_state = MoveItRobotState()
        if self._current_joint_positions:
            robot_state.joint_state.name = JOINT_NAMES
            robot_state.joint_state.position = [
                self._current_joint_positions.get(j, 0.0) for j in JOINT_NAMES
            ]
        ik_request.robot_state = robot_state

        request.ik_request = ik_request

        self.get_logger().info(f"🔧 Calcul IK pour x={x:.3f} y={y:.3f} z={z:.3f}")

        # Appel au service
        future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if future.result() is None:
            self.get_logger().error("IK : pas de réponse du service")
            return None

        result = future.result()

        if result.error_code.val != 1:  # 1 = SUCCESS
            self.get_logger().error(
                f"IK : échec (code erreur : {result.error_code.val})\n"
                f"La position x={x:.3f} y={y:.3f} z={z:.3f} est peut-être hors d'atteinte."
            )
            return None

        # Extraire les angles
        joint_positions = []
        for name in JOINT_NAMES:
            if name in result.solution.joint_state.name:
                idx = list(result.solution.joint_state.name).index(name)
                joint_positions.append(result.solution.joint_state.position[idx])
            else:
                self.get_logger().error(f"Joint {name} absent de la solution IK !")
                return None

        self.get_logger().info(
            f"✅ IK résolu : {[f'{p:.3f}' for p in joint_positions]}"
        )
        return joint_positions

    # ──────────────────────────────────────────────
    #  MOUVEMENT (envoie les angles articulaires)
    # ──────────────────────────────────────────────

    def move_to_joints(self, joint_positions, duration_sec=5.0):
        """
        Envoie le robot aux positions articulaires données.
        Utilise l'action FollowJointTrajectory (déjà éprouvée).
        """
        # Vérification sécurité
        if self._emergency_stop:
            self.get_logger().error("⚠️  MOUVEMENT BLOQUÉ — Arrêt d'urgence actif !")
            return False

        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1) * 1e9),
        )
        trajectory.points.append(point)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.get_logger().info(
            f"🤖 Mouvement → {[f'{p:.2f}' for p in joint_positions]} "
            f"({duration_sec}s)"
        )

        future = self._traj_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejeté par le controller !")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("✅ Mouvement terminé")
        return True

    def move_to_cartesian(self, x, y, z, orientation=None, duration_sec=5.0):
        """
        Mouvement vers une position cartésienne (x, y, z).
        Calcule l'IK puis envoie la trajectoire.
        """
        joint_positions = self.compute_ik(x, y, z, orientation)
        if joint_positions is None:
            return False
        return self.move_to_joints(joint_positions, duration_sec)

    # ──────────────────────────────────────────────
    #  PINCE (GRIPPER)
    # ──────────────────────────────────────────────

    def gripper_close(self):
        """
        PLACEHOLDER : Ferme la pince.
        Remplace par la vraie commande quand tu auras le gripper configuré.

        Exemples de vraies implémentations :
          - Topic : ros2 topic pub /gripper/command ...
          - Service : self.gripper_client.call(close_request)
          - GPIO / Arduino via serial
        """
        self.get_logger().info("🔒 PINCE : FERMETURE (placeholder)")
        time.sleep(1.0)  # Simule le temps de fermeture

    def gripper_open(self):
        """
        PLACEHOLDER : Ouvre la pince.
        """
        self.get_logger().info("🔓 PINCE : OUVERTURE (placeholder)")
        time.sleep(1.0)  # Simule le temps d'ouverture

    # ──────────────────────────────────────────────
    #  VÉRIFICATIONS DE SÉCURITÉ
    # ──────────────────────────────────────────────

    def is_position_safe(self, x, y, z):
        """Vérifie que la position cible est dans l'espace de travail autorisé."""
        limits = WORKSPACE_LIMITS
        if not (limits["x_min"] <= x <= limits["x_max"]):
            self.get_logger().error(f"❌ X={x:.3f} hors limites [{limits['x_min']}, {limits['x_max']}]")
            return False
        if not (limits["y_min"] <= y <= limits["y_max"]):
            self.get_logger().error(f"❌ Y={y:.3f} hors limites [{limits['y_min']}, {limits['y_max']}]")
            return False
        if not (limits["z_min"] <= z <= limits["z_max"]):
            self.get_logger().error(f"❌ Z={z:.3f} hors limites [{limits['z_min']}, {limits['z_max']}]")
            return False
        return True

    def check_emergency_stop(self):
        """Vérifie l'arrêt d'urgence. Retourne True si OK."""
        if self._emergency_stop:
            self.get_logger().error("⚠️  ARRÊT D'URGENCE — Action annulée")
            return False
        return True

    # ──────────────────────────────────────────────
    #  CYCLE PICK AND PLACE COMPLET
    # ──────────────────────────────────────────────

    def execute_pick_and_place(self, target_x, target_y, target_z):
        """
        Cycle complet de Pick and Place :
          1. Vérification sécurité
          2. Aller au-dessus de l'objet (approche)
          3. Descendre vers l'objet (pick)
          4. Fermer la pince
          5. Remonter
          6. Aller au point de dépôt
          7. Ouvrir la pince (place)
          8. Retour Home
        """
        self._is_busy = True
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  CYCLE PICK & PLACE")
        self.get_logger().info(f"  Cible : x={target_x:.3f} y={target_y:.3f} z={target_z:.3f}")
        self.get_logger().info("=" * 60)

        try:
            # ── 1. Vérifications ──
            if not self.check_emergency_stop():
                return False
            if not self.is_position_safe(target_x, target_y, target_z):
                return False

            # ── 2. Approche : au-dessus de l'objet ──
            self.get_logger().info("📍 Étape 1/6 : Approche (au-dessus de l'objet)")
            approach_z = target_z + APPROACH_HEIGHT_OFFSET
            if not self.check_emergency_stop():
                return False
            if not self.move_to_cartesian(target_x, target_y, approach_z,
                                          duration_sec=MOVE_DURATION_FAST):
                self.get_logger().error("Échec approche — abandon")
                return False

            # ── 3. Descente vers l'objet ──
            self.get_logger().info("📍 Étape 2/6 : Descente vers l'objet")
            if not self.check_emergency_stop():
                return False
            if not self.move_to_cartesian(target_x, target_y, target_z,
                                          duration_sec=MOVE_DURATION_SLOW):
                self.get_logger().error("Échec descente — abandon")
                return False

            # ── 4. Fermer la pince ──
            self.get_logger().info("📍 Étape 3/6 : Fermeture pince")
            if not self.check_emergency_stop():
                return False
            self.gripper_close()

            # ── 5. Remonter ──
            self.get_logger().info("📍 Étape 4/6 : Remontée")
            if not self.check_emergency_stop():
                return False
            if not self.move_to_cartesian(target_x, target_y, approach_z,
                                          duration_sec=MOVE_DURATION_FAST):
                self.get_logger().error("Échec remontée — abandon")
                return False

            # ── 6. Aller au dépôt ──
            self.get_logger().info("📍 Étape 5/6 : Déplacement vers dépôt")
            if not self.check_emergency_stop():
                return False
            if not self.move_to_cartesian(
                    DROP_POSITION["x"], DROP_POSITION["y"], DROP_POSITION["z"],
                    duration_sec=MOVE_DURATION_FAST):
                self.get_logger().error("Échec dépôt — abandon")
                return False

            # ── 7. Ouvrir la pince ──
            self.get_logger().info("📍 Étape 6/6 : Ouverture pince (dépôt)")
            if not self.check_emergency_stop():
                return False
            self.gripper_open()

            # ── 8. Retour Home ──
            self.get_logger().info("🏠 Retour à la position Home")
            if not self.check_emergency_stop():
                return False
            self.move_to_joints(HOME_JOINTS, duration_sec=MOVE_DURATION_FAST)

            self.get_logger().info("=" * 60)
            self.get_logger().info("  ✅ CYCLE PICK & PLACE TERMINÉ AVEC SUCCÈS")
            self.get_logger().info("=" * 60)
            return True

        except Exception as e:
            self.get_logger().error(f"❌ Erreur pendant le cycle : {e}")
            return False

        finally:
            self._is_busy = False
            self._target_received = False

    # ──────────────────────────────────────────────
    #  BOUCLE PRINCIPALE
    # ──────────────────────────────────────────────

    def run(self):
        """Boucle principale : attend les détections et exécute les cycles."""
        self.get_logger().info("🔄 Boucle principale active — Ctrl+C pour quitter")

        # Attendre les premières données joints
        self.get_logger().info("Lecture position actuelle du robot...")
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._current_joint_positions:
                break

        if self._current_joint_positions:
            current = [self._current_joint_positions.get(j, 0.0) for j in JOINT_NAMES]
            self.get_logger().info(f"Position actuelle : {[f'{p:.3f}' for p in current]}")
        else:
            self.get_logger().warn("Impossible de lire la position actuelle")

        # Boucle d'attente
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)

            # Si une cible a été reçue → lancer le cycle
            if self._target_received and not self._is_busy:
                target = self._target_point
                self.get_logger().info(
                    f"🎯 Cible reçue ! Lancement du cycle Pick & Place..."
                )

                # Confirmation manuelle (SÉCURITÉ — retirer en production)
                try:
                    user_input = input(
                        f"\n>>> Cible détectée : x={target.x:.3f} y={target.y:.3f} z={target.z:.3f}\n"
                        f">>> Appuie sur ENTRÉE pour exécuter le pick & place (ou 'q' pour ignorer) : "
                    )
                    if user_input.lower() == 'q':
                        self.get_logger().info("Cible ignorée par l'utilisateur")
                        self._target_received = False
                        continue
                except EOFError:
                    pass

                self.execute_pick_and_place(target.x, target.y, target.z)


# ╔══════════════════════════════════════════════════════════╗
# ║  POINT D'ENTRÉE                                          ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    rclpy.init()
    node = PickAndPlaceMaster()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt par l'utilisateur (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
