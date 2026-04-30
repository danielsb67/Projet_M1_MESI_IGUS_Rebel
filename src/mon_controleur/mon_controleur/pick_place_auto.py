#!/usr/bin/env python3
"""
Pick & Place AUTOMATIQUE — Igus Rebel + Schunk EGP25 (Version 100% ROS 2)

Architecture :
  - Bras   : bouge_robot.IgusRebelMover  → ROS 2 FollowJointTrajectory
  - Pince  : Service ROS 2 natif         → /gripper/command (SetBool)
"""

import time
import rclpy
from std_srvs.srv import SetBool

# ON SUPPRIME L'IMPORT DE LA PINCE EN SOCKET !
from mon_controleur.bouge_robot import IgusRebelMover

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Positions en RADIANS
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
APPROCHE_PICK = [0.0, 0.5760, 1.24, 0.0, 1.2217, 0.0]
PICK = [0.0, 0.6737, 1.3840, 0.0, 1.1135, 0.0]
APPROCHE_PLACE = [-2.9671, 0.5760, 1.2497, 0.0, 1.2217, 0.0]
PLACE = [-2.9706, 0.58, 1.538, 0.0, 1.0188, 0.0]

MOVE_DURATION        = 5.0
MOVE_DURATION_SLOW   = 3.5
GRIPPER_WAIT         = 1   # Temps mécanique pour la pince


# ══════════════════════════════════════════════════════════════════
#  NŒUD PICK & PLACE AUTOMATIQUE
# ══════════════════════════════════════════════════════════════════

class PickPlaceAuto:

    def __init__(self):
        # Le robot (qui est un Nœud ROS 2)
        self.robot = IgusRebelMover()
        
        # Création du Client pour appeler le service de la pince
        self.gripper_client = self.robot.create_client(SetBool, '/gripper/command')

    # ── Helpers pour le Mouvement ──────────────────────────────────

    def _move(self, positions, duration=MOVE_DURATION, label=''):
        if label:
            self.robot.get_logger().info(f'  → {label}')
        ok = self.robot.send_trajectory(positions, duration_sec=duration)
        if not ok:
            raise RuntimeError(f'Mouvement échoué : {label}')
        time.sleep(0.3)

    # ── Helpers pour la Pince (via ROS 2) ──────────────────────────

    def _control_gripper(self, activer: bool, label: str):
        self.robot.get_logger().info(f'  → {label}')
        
        # Préparation du message SetBool
        req = SetBool.Request()
        req.data = activer
        
        # Appel asynchrone du service et attente de la réponse
        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self.robot, future)
        
        if future.result() is not None:
            self.robot.get_logger().info(f'    Pince : {future.result().message}')
        else:
            self.robot.get_logger().error('    Échec de l\'appel au service pince !')
            
        time.sleep(GRIPPER_WAIT)

    def _ouvrir(self):
        self._control_gripper(activer=False, label="Ouvrir pince (SetBool: False)")

    def _fermer(self):
        self._control_gripper(activer=True, label="Fermer pince (SetBool: True)")

    # ── Cycle principal ────────────────────────────────────────────

    def run(self):
        log = self.robot.get_logger()

        # ── Connexions ROS 2 (Bras et Pince) ────────────────────────
        log.info('Connexion au serveur d\'action du bras...')
        if not self.robot.wait_for_server(timeout_sec=15.0):
            log.error('Controller de bras non disponible — arrêt.')
            return

        log.info('Attente du service de la pince (/gripper/command)...')
        while not self.gripper_client.wait_for_service(timeout_sec=2.0):
            log.info('Service pince non disponible, on patiente...')
        log.info('✅ Service pince connecté !')

        current = self.robot.get_current_positions()
        if current:
            log.info(f'Position initiale : {[f"{p:.3f}" for p in current]}')
        else:
            log.warn('Impossible de lire la position actuelle.')

        log.info('=' * 55)
        log.info('  DÉMARRAGE CYCLE PICK & PLACE AUTOMATIQUE')
        log.info('=' * 55)

        try:
            # Plus besoin de ruses, on déroule la séquence pure !
            
            # 1. Home
            log.info('[1/9] Aller à HOME')
            self._move(HOME, MOVE_DURATION, 'HOME')

            # 2. Ouvrir la pince
            log.info('[2/9] Préparation pince')
            self._ouvrir()

            # 3. Approche pick
            log.info('[3/9] Approche au-dessus de l\'objet')
            self._move(APPROCHE_PICK, MOVE_DURATION, 'APPROCHE_PICK')

            # 4. Descente vers l'objet
            log.info('[4/9] Descente vers l\'objet (PICK)')
            self._move(PICK, MOVE_DURATION_SLOW, 'PICK')

            # 5. Fermer la pince
            log.info('[5/9] Saisie de l\'objet')
            self._fermer()

            # 6. Remonter
            log.info('[6/9] Remontée')
            self._move(APPROCHE_PICK, MOVE_DURATION, 'REMONTÉE_PICK')

            # 7. Approche dépôt
            log.info('[7/9] Déplacement vers la zone de dépôt')
            self._move(APPROCHE_PLACE, MOVE_DURATION, 'APPROCHE_PLACE')

            # 8. Descente au dépôt
            log.info('[8/9] Descente vers le dépôt (PLACE)')
            self._move(PLACE, MOVE_DURATION_SLOW, 'PLACE')

            # 9. Ouvrir la pince
            log.info('[9/9] Lâcher de l\'objet')
            self._ouvrir()

            # ── Retour HOME ─────────────────────────────────────────
            log.info('[fin] Remontée et retour HOME')
            self._move(APPROCHE_PLACE, MOVE_DURATION, 'REMONTÉE_PLACE')
            self._move(HOME, MOVE_DURATION, 'HOME')

            log.info('=' * 55)
            log.info('  CYCLE TERMINÉ AVEC SUCCÈS ✅')
            log.info('=' * 55)

        except RuntimeError as e:
            log.error(f'ERREUR pendant le cycle : {e}')
        except KeyboardInterrupt:
            log.info('Arrêt demandé (Ctrl+C).')


# ══════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = PickPlaceAuto()
    try:
        node.run()
    finally:
        node.robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()