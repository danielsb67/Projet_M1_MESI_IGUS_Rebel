#!/usr/bin/env python3
"""
Nœud ROS 2 — Cage de collision MoveIt (Planning Scene)
=======================================================
Publie six murs invisibles (sol, plafond, 4 côtés) dans la planning scene
de MoveIt dès le démarrage. Le planificateur OMPL/Pilz refusera toute
trajectoire dont un lien du robot toucherait ces murs (erreur -31).

Les dimensions sont calibrées pour l'Igus Rebel + Schunk EGP25 posé sur table.
Ajuste les constantes WORKSPACE_* pour ton installation physique.

Lancement (terminal séparé, après demo.launch.py) :
  ros2 run mon_controleur workspace_scene
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene, ObjectColor
from std_msgs.msg import ColorRGBA

# ──────────────────────────────────────────────────────────────────────────────
#  GÉOMÉTRIE DE LA CAGE — adapter à ton installation (en mètres)
#  Repère : igus_rebel_base_link (centre du robot, niveau table)
# ──────────────────────────────────────────────────────────────────────────────

PLANNING_FRAME = "igus_rebel_base_link"

Z_SOL     = -0.018   # plancher — aligné avec Z_STOP_M dans securite.py (-0.018 m)
Z_PLAFOND =  1.3    # hauteur max (Rebel reach ≈ 700 mm + marge sécurité)
X_AVANT   =  0.68    # mur frontal (+X, face au robot)
X_ARRIERE = -0.5    # mur dorsal  (-X, derrière le robot)
Y_GAUCHE  =  0.35    # mur gauche  (+Y)
Y_DROITE  = -0.35    # mur droit   (-Y)

EPAISSEUR =  0.02    # épaisseur de chaque dalle (m) — assez pour être détectée

# ── Obstacles fixes dans la scène ─────────────────────────────────────────────
# Profilé aluminium fixé sur la table, pile en face du robot (+X)
PROFIL_X  =  0.585   # centre en X (bord à 510 mm + demi-largeur 75 mm)
PROFIL_Y  =  0.0     # centré en Y
PROFIL_LX =  0.15    # 150 mm (profondeur)
PROFIL_LY =  0.15    # 150 mm (largeur)
PROFIL_LZ =  0.52    # 520 mm (420 mm réels + 100 mm de marge pour protéger la caméra)

# ──────────────────────────────────────────────────────────────────────────────


def _boite(x: float, y: float, z: float,
           lx: float, ly: float, lz: float) -> tuple[SolidPrimitive, Pose]:
    forme = SolidPrimitive()
    forme.type = SolidPrimitive.BOX
    forme.dimensions = [lx, ly, lz]

    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return forme, pose


def _construire_cage(frame: str, avec_profil: bool = True) -> list[CollisionObject]:
    largeur = X_AVANT - X_ARRIERE + 2 * EPAISSEUR
    profond = Y_GAUCHE - Y_DROITE  + 2 * EPAISSEUR
    hauteur = Z_PLAFOND - Z_SOL   + 2 * EPAISSEUR

    cx = (X_AVANT + X_ARRIERE) / 2.0
    cy = (Y_GAUCHE + Y_DROITE)  / 2.0
    cz = (Z_PLAFOND + Z_SOL)   / 2.0

    #            nom            x                              y                              z                          lx        ly        lz
    murs = [
        ("sol",         cx,                            cy,                            Z_SOL     - EPAISSEUR / 2, largeur, profond, EPAISSEUR),
        ("plafond",     cx,                            cy,                            Z_PLAFOND + EPAISSEUR / 2, largeur, profond, EPAISSEUR),
        ("mur_avant",   X_AVANT   + EPAISSEUR / 2,    cy,                            cz,                        EPAISSEUR, profond, hauteur),
        ("mur_arriere", X_ARRIERE - EPAISSEUR / 2,    cy,                            cz,                        EPAISSEUR, profond, hauteur),
        ("mur_gauche",  cx,                            Y_GAUCHE + EPAISSEUR / 2,     cz,                        largeur, EPAISSEUR, hauteur),
        ("mur_droite",  cx,                            Y_DROITE - EPAISSEUR / 2,     cz,                        largeur, EPAISSEUR, hauteur),
    ]

    objets = []
    for nom, x, y, z, lx, ly, lz in murs:
        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.id = nom
        obj.operation = CollisionObject.ADD
        forme, pose = _boite(x, y, z, lx, ly, lz)
        obj.primitives = [forme]
        obj.primitive_poses = [pose]
        objets.append(obj)

    if avec_profil:
        profil = CollisionObject()
        profil.header.frame_id = frame
        profil.id = "profil_avant"
        profil.operation = CollisionObject.ADD
        forme, pose = _boite(PROFIL_X, PROFIL_Y, PROFIL_LZ / 2,
                             PROFIL_LX, PROFIL_LY, PROFIL_LZ)
        profil.primitives = [forme]
        profil.primitive_poses = [pose]
        objets.append(profil)

    return objets


class WorkspaceScene(Node):
    def __init__(self):
        super().__init__("workspace_scene")

        self.declare_parameter("sim_mode", False)
        self._sim_mode = self.get_parameter("sim_mode").value

        qos = QoSProfile(depth=10,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(PlanningScene, "/planning_scene", qos)

        self.get_logger().info("⏳  Attente démarrage MoveIt Move Group (3 s)…")
        time.sleep(3.0)

        self._publier_cage()

    def _publier_cage(self):
        objets = _construire_cage(PLANNING_FRAME, avec_profil=not self._sim_mode)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objets

        for obj in objets:
            c = ObjectColor()
            c.id = obj.id
            c.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)
            scene.object_colors.append(c)

        self._pub.publish(scene)

        self.get_logger().info(
            f"✓ Cage publiée — {len(objets)} objets dans '{PLANNING_FRAME}'\n"
            f"   Sol     : Z ≥ {Z_SOL:.3f} m\n"
            f"   Plafond : Z ≤ {Z_PLAFOND:.3f} m\n"
            f"   X       : [{X_ARRIERE:.2f} … {X_AVANT:.2f}] m\n"
            f"   Y       : [{Y_DROITE:.2f} … {Y_GAUCHE:.2f}] m\n"
            f"   Profilé : X={PROFIL_X:.3f} m, Y={PROFIL_Y:.3f} m, "
            f"{int(PROFIL_LX*1000)}x{int(PROFIL_LY*1000)}x{int(PROFIL_LZ*1000)} mm\n"
            f"   → MoveIt refusera toute trajectoire hors de cette enveloppe."
        )


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceScene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()