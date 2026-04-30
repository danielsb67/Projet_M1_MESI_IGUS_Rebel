#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  IGUS REBEL + SCHUNK EGP 25 — BIBLIOTHÈQUE DE CONTRÔLE COMPLÈTE
═══════════════════════════════════════════════════════════════════════

3 couches empilées :
  1. IgusRebelCRI      — Communication TCP/CRI bas niveau (socket)
  2. SchunkEGP25       — Contrôle pince (hérite de IgusRebelCRI)
  3. RebelPickAndPlace  — Séquences pick & place (mouvement + pince)

Auteur  : Maître — M1 PAIP, Université de Strasbourg
Robot   : igus Rebel 6DOF, CRI protocol, 192.168.3.11:3920
Pince   : SCHUNK EGP 25-N-N-B (entrées digitales 24V, pas IO-Link)
Contrôle: CMD DOUT 30 = DOut31 (ouvrir), CMD DOUT 31 = DOut32 (fermer)
"""

import socket
import time
import threading
import struct
import math


# ╔══════════════════════════════════════════════════════════════════╗
# ║  OPTION 1 : CLASSE CRI BAS NIVEAU + PINCE                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class IgusRebelCRI:
    """
    Couche de communication TCP avec le contrôleur igus via protocole CRI.
    
    Le protocole CRI (Commonplace Robotics Interface) est un protocole
    texte sur TCP. Chaque message a le format :
        CRISTART <counter> <commande> CRIEND\n
    
    OBLIGATION : envoyer ALIVEJOG toutes les ~200ms sinon le contrôleur
    coupe la connexion après ~1s. C'est un watchdog de sécurité.
    """

    def __init__(self, ip='192.168.3.11', port=3920):
        self.ip = ip
        self.port = port
        self.sock = None
        self._counter = 0          # Compteur CRI (incrémenté à chaque message)
        self._lock = threading.Lock()  # Thread-safety pour le socket
        self._alive_running = False
        self._alive_thread = None
        self._connected = False

    # ── Connexion / Déconnexion ──────────────────────────────────

    def connect(self):
        """
        Établit la connexion TCP et démarre le heartbeat ALIVEJOG.
        
        Séquence :
          1. Ouvrir le socket TCP vers 192.168.3.11:3920
          2. Démarrer le thread ALIVEJOG (watchdog obligatoire)
          3. Attendre 1s que le contrôleur nous enregistre
          4. Demander le statut de client actif (CMD SetActive true)
             → nécessaire pour pouvoir envoyer des commandes
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.ip, self.port))
        
        # Démarrer le heartbeat AVANT toute commande
        self._alive_running = True
        self._alive_thread = threading.Thread(target=self._alive_loop, daemon=True)
        self._alive_thread.start()
        
        time.sleep(1.0)  # Laisser le serveur CRI nous enregistrer
        
        # Prendre le contrôle actif
        # Le contrôleur igus n'accepte les commandes que du client "actif".
        # Si le driver ROS2 tourne aussi, il y a compétition — on reprend
        # le contrôle à chaque commande critique.
        self._send('CMD SetActive true')
        time.sleep(0.5)
        
        self._connected = True
        return True

    def disconnect(self):
        """Arrête le heartbeat et ferme le socket proprement."""
        self._alive_running = False
        if self._alive_thread:
            self._alive_thread.join(timeout=1.0)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self):
        return self._connected

    # ── Communication CRI ────────────────────────────────────────

    def _send(self, command):
        """
        Envoie un message CRI formaté.
        
        Format : CRISTART <counter> <command> CRIEND\n
        
        Le lock garantit qu'un seul thread écrit sur le socket à la fois
        (le thread ALIVEJOG et le thread principal envoient en parallèle).
        """
        with self._lock:
            self._counter = (self._counter % 9999) + 1
            msg = f"CRISTART {self._counter} {command} CRIEND\n"
            try:
                self.sock.sendall(msg.encode('utf-8'))
                return True
            except Exception as e:
                print(f"[CRI] Erreur envoi: {e}")
                return False

    def _receive(self, timeout=0.5):
        """Lit la réponse du serveur (best-effort, non bloquant)."""
        try:
            self.sock.settimeout(timeout)
            data = self.sock.recv(8192)
            return data.decode('utf-8', errors='ignore')
        except socket.timeout:
            return ""
        except Exception:
            return ""

    def _alive_loop(self):
        """
        Thread heartbeat : envoie ALIVEJOG toutes les 200ms.
        
        ALIVEJOG a 9 paramètres flottants (les 6 axes + 3 réservés).
        Tous à 0.0 = "je suis vivant mais ne demande aucun mouvement jog".
        Si on arrête d'envoyer, le contrôleur nous déconnecte après ~1s.
        """
        c = 5000  # Compteur séparé pour ne pas interférer avec _send
        while self._alive_running:
            try:
                with self._lock:
                    msg = f"CRISTART {c} ALIVEJOG 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 CRIEND\n"
                    self.sock.sendall(msg.encode('utf-8'))
                c = (c % 9999) + 1
            except Exception:
                break
            time.sleep(0.2)

    def request_active(self):
        """
        Reprend le statut de client actif.
        
        Utile quand le driver ROS2 (igus_rebel_hw_controller) tourne
        en parallèle et reprend automatiquement le contrôle actif.
        On doit re-demander avant chaque commande critique.
        """
        self._send('CMD SetActive true')
        time.sleep(0.3)


class SchunkEGP25(IgusRebelCRI):
    """
    Contrôle de la pince SCHUNK EGP 25-N-N-B via les DOut du module igus.
    
    Hérite de IgusRebelCRI pour réutiliser la connexion TCP/CRI.
    
    Mapping hardware (confirmé par test) :
      - CMD DOUT 30 true  → active DOut31 dans iRC → OUVRIR la pince
      - CMD DOUT 31 true  → active DOut32 dans iRC → FERMER la pince
      
    Convention CRI : numéro CRI = numéro iRC - 1
    
    Logique de commutation :
      Pour ouvrir  : d'abord désactiver DOut32 (fermer), puis activer DOut31 (ouvrir)
      Pour fermer  : d'abord désactiver DOut31 (ouvrir), puis activer DOut32 (fermer)
      → Évite un court-circuit logique où les deux seraient actifs simultanément.
    """

    # Numéros CRI (= numéro iRC - 1)
    DOUT_OPEN  = 30   # DOut31 dans iRC = ouvrir
    DOUT_CLOSE = 31   # DOut32 dans iRC = fermer

    SETTLE_TIME = 0.5  # Temps d'attente après commande (EGP 25 = ~0.09s, on prend de la marge)

    def ouvrir(self):
        """
        Ouvre la pince : désactive FERMER puis active OUVRIR.
        Attend SETTLE_TIME pour que le mouvement se termine.
        """
        self.request_active()
        self._send(f'CMD DOUT {self.DOUT_CLOSE} false')
        time.sleep(0.05)
        self._send(f'CMD DOUT {self.DOUT_OPEN} true')
        time.sleep(self.SETTLE_TIME)

    def fermer(self):
        """
        Ferme la pince : désactive OUVRIR puis active FERMER.
        Attend SETTLE_TIME pour que le mouvement se termine.
        """
        self.request_active()
        self._send(f'CMD DOUT {self.DOUT_OPEN} false')
        time.sleep(0.05)
        self._send(f'CMD DOUT {self.DOUT_CLOSE} true')
        time.sleep(self.SETTLE_TIME)

    def relacher(self):
        """Coupe les deux sorties — la pince reste dans sa position actuelle (pas de courant)."""
        self._send(f'CMD DOUT {self.DOUT_OPEN} false')
        self._send(f'CMD DOUT {self.DOUT_CLOSE} false')

    def disconnect(self):
        """Relâche la pince avant de déconnecter."""
        self.relacher()
        time.sleep(0.2)
        super().disconnect()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  OPTION 2 : PICK & PLACE PUR PYTHON (SANS ROS)                ║
# ╚══════════════════════════════════════════════════════════════════╝

class RebelPickAndPlace(SchunkEGP25):
    """
    Contrôle complet du bras igus Rebel + pince via CRI pur.
    
    Ajoute les commandes de mouvement articulaire et cartésien
    au contrôle de la pince. Tout passe par le même socket TCP.
    
    Commandes de mouvement CRI utilisées :
      - CMD Move Joint <j1> <j2> <j3> <j4> <j5> <j6> <v>
        → Mouvement articulaire (angles en degrés, vitesse en %)
      - CMD Move Cart <x> <y> <z> <a> <b> <c> <v>
        → Mouvement cartésien (position en mm, angles en degrés, vitesse en %)
    """

    # Limites de sécurité (mètres → converti en mm pour CRI)
    WORKSPACE = {
        'x_min': 50, 'x_max': 500,    # mm
        'y_min': -350, 'y_max': 350,   # mm
        'z_min': -150, 'z_max': 550,   # mm (base à 200mm au-dessus de la table)
    }

    def move_joint(self, joints_deg, velocity=30):
        """
        Mouvement articulaire.
        
        Args:
            joints_deg: liste de 6 angles en DEGRÉS [j1, j2, j3, j4, j5, j6]
            velocity: vitesse en % (1-100), défaut 30% pour la sécurité
            
        CRI attend : CMD Move Joint j1 j2 j3 j4 j5 j6 0 0 0 v
        Les 3 zéros sont pour les axes externes (non utilisés).
        """
        if len(joints_deg) != 6:
            raise ValueError("6 angles requis")
        
        j = joints_deg
        v = max(1, min(100, velocity))
        
        self.request_active()
        cmd = f"CMD Move Joint {j[0]:.2f} {j[1]:.2f} {j[2]:.2f} {j[3]:.2f} {j[4]:.2f} {j[5]:.2f} 0 0 0 {v}"
        self._send(cmd)
        print(f"  → Move Joint [{j[0]:.1f}, {j[1]:.1f}, {j[2]:.1f}, {j[3]:.1f}, {j[4]:.1f}, {j[5]:.1f}] v={v}%")

    def move_cartesian(self, x, y, z, a=0, b=0, c=0, velocity=30):
        """
        Mouvement cartésien.
        
        Args:
            x, y, z: position en MILLIMÈTRES (repère base robot)
            a, b, c: orientation en DEGRÉS (Euler ZYX)
            velocity: vitesse en % (1-100)
            
        CRI attend : CMD Move Cart x y z a b c 0 0 0 v
        """
        # Vérification des limites
        ws = self.WORKSPACE
        if not (ws['x_min'] <= x <= ws['x_max']):
            print(f"  ❌ X={x:.1f}mm hors limites [{ws['x_min']}, {ws['x_max']}]")
            return False
        if not (ws['y_min'] <= y <= ws['y_max']):
            print(f"  ❌ Y={y:.1f}mm hors limites [{ws['y_min']}, {ws['y_max']}]")
            return False
        if not (ws['z_min'] <= z <= ws['z_max']):
            print(f"  ❌ Z={z:.1f}mm hors limites [{ws['z_min']}, {ws['z_max']}]")
            return False
        
        v = max(1, min(100, velocity))
        
        self.request_active()
        cmd = f"CMD Move Cart {x:.2f} {y:.2f} {z:.2f} {a:.2f} {b:.2f} {c:.2f} 0 0 0 {v}"
        self._send(cmd)
        print(f"  → Move Cart [{x:.1f}, {y:.1f}, {z:.1f}] a={a:.1f} b={b:.1f} c={c:.1f} v={v}%")
        return True

    def move_home(self, velocity=30):
        """Retour à la position home (tous les joints à 0°)."""
        print("🏠 Retour Home")
        self.move_joint([0, 0, 0, 0, 0, 0], velocity)

    def wait_motion(self, duration):
        """
        Attend la fin d'un mouvement.
        
        NOTE : Le protocole CRI n'a pas de feedback de fin de mouvement
        dans la version de base. On utilise un délai estimé.
        Une version avancée parserait les messages STATUS pour vérifier
        que la position cible est atteinte.
        """
        time.sleep(duration)

    def pick_and_place(self, pick_xyz, drop_xyz, approach_h=100, velocity=30):
        """
        Cycle complet Pick & Place.
        
        Args:
            pick_xyz: [x, y, z] position de l'objet en MILLIMÈTRES
            drop_xyz: [x, y, z] position de dépôt en MILLIMÈTRES
            approach_h: hauteur d'approche au-dessus de la cible (mm)
            velocity: vitesse (%)
            
        Séquence :
          1. Ouvrir la pince (préparer)
          2. Aller au-dessus de l'objet (approche)
          3. Descendre vers l'objet
          4. Fermer la pince (saisir)
          5. Remonter
          6. Aller au-dessus du dépôt
          7. Descendre au dépôt
          8. Ouvrir la pince (relâcher)
          9. Remonter
        """
        px, py, pz = pick_xyz
        dx, dy, dz = drop_xyz

        print("=" * 50)
        print(f"  PICK & PLACE")
        print(f"  Pick : [{px:.1f}, {py:.1f}, {pz:.1f}] mm")
        print(f"  Drop : [{dx:.1f}, {dy:.1f}, {dz:.1f}] mm")
        print("=" * 50)

        # Estimation du temps de mouvement (très approximatif)
        move_time = 5.0   # secondes par mouvement (ajuster selon velocity)
        descend_time = 3.0

        # 1. Préparer la pince
        print("\n📍 1/9 Ouvrir pince")
        self.ouvrir()

        # 2. Approche pick (au-dessus)
        print("\n📍 2/9 Approche pick")
        self.move_cartesian(px, py, pz + approach_h, velocity=velocity)
        self.wait_motion(move_time)

        # 3. Descente pick
        print("\n📍 3/9 Descente pick")
        self.move_cartesian(px, py, pz, velocity=max(10, velocity // 2))
        self.wait_motion(descend_time)

        # 4. Saisir
        print("\n📍 4/9 Fermer pince")
        self.fermer()
        time.sleep(0.5)

        # 5. Remonter
        print("\n📍 5/9 Remontée")
        self.move_cartesian(px, py, pz + approach_h, velocity=velocity)
        self.wait_motion(move_time)

        # 6. Approche drop
        print("\n📍 6/9 Approche drop")
        self.move_cartesian(dx, dy, dz + approach_h, velocity=velocity)
        self.wait_motion(move_time)

        # 7. Descente drop
        print("\n📍 7/9 Descente drop")
        self.move_cartesian(dx, dy, dz, velocity=max(10, velocity // 2))
        self.wait_motion(descend_time)

        # 8. Relâcher
        print("\n📍 8/9 Ouvrir pince")
        self.ouvrir()
        time.sleep(0.5)

        # 9. Remonter
        print("\n📍 9/9 Remontée finale")
        self.move_cartesian(dx, dy, dz + approach_h, velocity=velocity)
        self.wait_motion(move_time)

        print("\n✅ Cycle Pick & Place terminé !")
        print("=" * 50)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  OPTION 3 : NŒUD ROS 2 HUMBLE                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# L'import ROS2 est conditionnel pour permettre l'utilisation
# de la bibliothèque sans ROS (Options 1 et 2)
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import PointStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


if ROS2_AVAILABLE:

    class RebelGripperNode(Node):
        """
        Nœud ROS 2 pour le contrôle de la pince SCHUNK EGP 25.
        
        Topics écoutés :
          /gripper/command (std_msgs/String) : "open", "close", "release"
          
        Ce nœud peut coexister avec le driver ROS2 du bras igus.
        Il reprend le statut actif CRI uniquement pour les commandes pince,
        puis le driver reprend automatiquement le contrôle pour les mouvements.
        """

        def __init__(self):
            super().__init__('rebel_gripper_node')
            
            # Paramètres ROS2 (configurables au lancement)
            self.declare_parameter('robot_ip', '192.168.3.11')
            self.declare_parameter('robot_port', 3920)
            
            ip = self.get_parameter('robot_ip').value
            port = self.get_parameter('robot_port').value
            
            # Créer la pince
            self.gripper = SchunkEGP25(ip=ip, port=port)
            
            # Subscriber pour les commandes
            self.create_subscription(
                String, '/gripper/command', self._cmd_callback, 10)
            
            # Connexion
            self.get_logger().info(f'Connexion CRI → {ip}:{port}...')
            try:
                self.gripper.connect()
                self.get_logger().info('✅ Pince EGP 25 connectée !')
                self.get_logger().info('   Commandes : open | close | release')
            except Exception as e:
                self.get_logger().error(f'❌ Connexion échouée : {e}')

        def _cmd_callback(self, msg):
            """Callback pour les commandes pince."""
            cmd = msg.data.strip().lower()
            
            if cmd == 'open':
                self.get_logger().info('🔓 Ouverture pince')
                self.gripper.ouvrir()
            elif cmd == 'close':
                self.get_logger().info('🔒 Fermeture pince')
                self.gripper.fermer()
            elif cmd == 'release':
                self.get_logger().info('⏹️  Relâchement pince')
                self.gripper.relacher()
            else:
                self.get_logger().warn(f'Commande inconnue : "{cmd}"')

        def destroy_node(self):
            self.gripper.disconnect()
            super().destroy_node()


    class RebelPickAndPlaceNode(Node):
        """
        Nœud ROS 2 complet : mouvement bras + pince.
        
        Topics écoutés :
          /gripper/command (String)          : "open", "close", "release"
          /pick_place/target (PointStamped)  : coordonnées XYZ de l'objet
          /pick_place/command (String)       : "home", "zero", "pick"
          
        NOTE : Ce nœud prend le contrôle CRI complet du robot.
        Ne PAS l'utiliser en même temps que le driver ROS2 igus
        (igus_rebel_hw_controller) car ils entrent en compétition
        pour le statut de client actif.
        """

        def __init__(self):
            super().__init__('rebel_pick_and_place_node')
            
            self.declare_parameter('robot_ip', '192.168.3.11')
            self.declare_parameter('robot_port', 3920)
            self.declare_parameter('drop_x', 250.0)
            self.declare_parameter('drop_y', -200.0)
            self.declare_parameter('drop_z', 100.0)
            
            ip = self.get_parameter('robot_ip').value
            port = self.get_parameter('robot_port').value
            
            self.robot = RebelPickAndPlace(ip=ip, port=port)
            
            self._drop_pos = [
                self.get_parameter('drop_x').value,
                self.get_parameter('drop_y').value,
                self.get_parameter('drop_z').value,
            ]
            
            # Subscribers
            self.create_subscription(
                String, '/gripper/command', self._gripper_cb, 10)
            self.create_subscription(
                PointStamped, '/pick_place/target', self._target_cb, 10)
            self.create_subscription(
                String, '/pick_place/command', self._command_cb, 10)
            
            # Connexion
            self.get_logger().info(f'Connexion CRI → {ip}:{port}...')
            try:
                self.robot.connect()
                self.get_logger().info('✅ Robot + Pince connectés !')
            except Exception as e:
                self.get_logger().error(f'❌ {e}')

        def _gripper_cb(self, msg):
            cmd = msg.data.strip().lower()
            if cmd == 'open':
                self.robot.ouvrir()
                self.get_logger().info('🔓 Pince ouverte')
            elif cmd == 'close':
                self.robot.fermer()
                self.get_logger().info('🔒 Pince fermée')
            elif cmd == 'release':
                self.robot.relacher()

        def _target_cb(self, msg):
            """Reçoit une cible XYZ et lance le cycle pick & place."""
            # Convertir mètres → millimètres (CRI travaille en mm)
            x_mm = msg.point.x * 1000.0
            y_mm = msg.point.y * 1000.0
            z_mm = msg.point.z * 1000.0
            
            self.get_logger().info(
                f'📦 Cible reçue : [{x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}] mm')
            
            self.robot.pick_and_place(
                pick_xyz=[x_mm, y_mm, z_mm],
                drop_xyz=self._drop_pos,
                velocity=30
            )

        def _command_cb(self, msg):
            cmd = msg.data.strip().lower()
            if cmd == 'home':
                self.robot.move_home()
            elif cmd == 'zero':
                self.robot.move_joint([0, 0, 0, 0, 0, 0])

        def destroy_node(self):
            self.robot.disconnect()
            super().destroy_node()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  POINTS D'ENTRÉE                                               ║
# ╚══════════════════════════════════════════════════════════════════╝

def demo_classe():
    """Option 1 : Test simple de la classe pince."""
    print("═" * 50)
    print("  DEMO CLASSE PINCE — Option 1")
    print("═" * 50)
    
    pince = SchunkEGP25()
    pince.connect()
    
    print("\nTest ouverture/fermeture :")
    pince.ouvrir()
    print("  Pince ouverte — attente 2s")
    time.sleep(2)
    
    pince.fermer()
    print("  Pince fermée — attente 2s")
    time.sleep(2)
    
    pince.ouvrir()
    print("  Pince ouverte")
    
    pince.disconnect()
    print("\nDéconnecté. ✅")


def demo_pick_and_place():
    """Option 2 : Pick & Place pur Python."""
    print("═" * 50)
    print("  DEMO PICK & PLACE — Option 2")
    print("═" * 50)
    
    robot = RebelPickAndPlace()
    robot.connect()
    
    # Coordonnées en MILLIMÈTRES (repère base robot)
    # Adapte selon ton setup !
    PICK = [300, 100, 150]    # Position de l'objet
    DROP = [250, -150, 150]   # Position de dépôt
    
    input("\n⚠️  Le robot va bouger ! ENTRÉE pour commencer...")
    
    robot.pick_and_place(
        pick_xyz=PICK,
        drop_xyz=DROP,
        approach_h=100,   # 100mm au-dessus
        velocity=25       # 25% de vitesse
    )
    
    robot.move_home(velocity=20)
    robot.wait_motion(8)
    
    robot.disconnect()


def demo_ros2_gripper():
    """Option 3a : Nœud ROS2 pince seule."""
    if not ROS2_AVAILABLE:
        print("❌ ROS2 non disponible (rclpy non importable)")
        return
    
    rclpy.init()
    node = RebelGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def demo_ros2_pick_and_place():
    """Option 3b : Nœud ROS2 pick & place complet."""
    if not ROS2_AVAILABLE:
        print("❌ ROS2 non disponible")
        return
    
    rclpy.init()
    node = RebelPickAndPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def mode_interactif():
    """Mode interactif : contrôle manuel depuis le terminal."""
    print("═" * 50)
    print("  MODE INTERACTIF — Contrôle manuel")
    print("═" * 50)
    
    robot = RebelPickAndPlace()
    robot.connect()
    
    print("\nCommandes disponibles :")
    print("  o        → ouvrir pince")
    print("  f        → fermer pince")
    print("  r        → relâcher pince")
    print("  h        → retour home")
    print("  j a b c d e f  → mouvement articulaire (6 angles en degrés)")
    print("  c x y z        → mouvement cartésien (en mm)")
    print("  p              → cycle pick & place (positions par défaut)")
    print("  q              → quitter")
    
    while True:
        try:
            cmd = input("\n>>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not cmd:
            continue
        
        if cmd == 'q':
            break
        elif cmd == 'o':
            robot.ouvrir()
        elif cmd == 'f':
            robot.fermer()
        elif cmd == 'r':
            robot.relacher()
        elif cmd == 'h':
            robot.move_home()
            robot.wait_motion(8)
        elif cmd.startswith('j '):
            parts = cmd[2:].split()
            if len(parts) == 6:
                angles = [float(x) for x in parts]
                robot.move_joint(angles)
                robot.wait_motion(5)
            else:
                print("❌ 6 angles requis : j 0 0 0 0 0 0")
        elif cmd.startswith('c '):
            parts = cmd[2:].split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                robot.move_cartesian(x, y, z)
                robot.wait_motion(5)
            else:
                print("❌ 3 coordonnées min : c 300 0 200")
        elif cmd == 'p':
            robot.pick_and_place([300, 100, 150], [250, -150, 150])
        else:
            print(f"❌ Commande inconnue : {cmd}")
    
    robot.disconnect()
    print("Déconnecté. ✅")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'classe':
            demo_classe()
        elif mode == 'pick':
            demo_pick_and_place()
        elif mode == 'ros_gripper':
            demo_ros2_gripper()
        elif mode == 'ros_pick':
            demo_ros2_pick_and_place()
        elif mode == 'interactif':
            mode_interactif()
        else:
            print(f"Mode inconnu : {mode}")
            print("Usage : python3 igus_rebel_control.py [classe|pick|interactif|ros_gripper|ros_pick]")
    else:
        mode_interactif()
