#!/usr/bin/env python3

import rclpy
from mon_controleur.bouge_robot import IgusRebelMover

def main():
    # 1. Initialisation de ROS 2
    rclpy.init()
    
    # 2. Création du nœud de contrôle
    robot = IgusRebelMover()
    
    try:
        robot.get_logger().info("Recherche du serveur d'action du robot...")
        
        # 3. Vérification de la connexion
        if not robot.wait_for_server(timeout_sec=10.0):
            robot.get_logger().error("Serveur introuvable. As-tu bien lancé le 'ros2 launch' ?")
            return
            
        robot.get_logger().info("✅ Robot connecté ! Lancement du retour au HOME...")
        
        # 4. Les coordonnées de la position HOME (6 axes en radians)
        POSITION_HOME = [0.0001, 0.5, 0.5, 0.0 , 0.1, 1.55]
        
        # 5. Envoi de la commande avec une durée de 5 secondes pour un mouvement doux
        robot.send_trajectory(POSITION_HOME, duration_sec=5.0)
        
        robot.get_logger().info("🏁 Le robot est en position HOME !")

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