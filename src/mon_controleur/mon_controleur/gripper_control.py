import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class SchunkGripper:
    """
    Contrôle la pince Schunk EGP25 via le service ROS2 /gripper/command
    exposé par le driver igus_rebel_hw_controller.
    Le driver réutilise sa propre connexion CRI — pas de second socket TCP.

    API :
        connect()      — attend que le service soit disponible
        ouvrir()       — envoie DOUT 30 true  (ouvre)
        fermer()       — envoie DOUT 31 true  (ferme)
        relacher()     — coupe les deux DOUT  (relâche)
        disconnect()   — relâche et nettoie
    """

    SERVICE = '/gripper/command'

    def __init__(self, node: Node):
        self._node = node
        self._client = node.create_client(SetBool, self.SERVICE)

    def connect(self, timeout_sec: float = 10.0):
        print(f"Connexion au service {self.SERVICE}...")
        if not self._client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f"Service {self.SERVICE} non disponible après {timeout_sec}s.\n"
                "Vérifier que le driver igus_rebel_hw_controller est bien lancé."
            )
        print("Pince prête !")

    def _call(self, close: bool):
        req = SetBool.Request()
        req.data = close
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        if future.result() is None or not future.result().success:
            raise RuntimeError("Commande pince échouée (service sans réponse)")

    def ouvrir(self):
        print("Ouverture de la pince...")
        self._call(close=False)

    def fermer(self):
        print("Fermeture de la pince...")
        self._call(close=True)

    def relacher(self):
        # Pas de service dédié pour le relâchement — on ouvre simplement
        self.ouvrir()

    def disconnect(self):
        print("Relâchement pince...")
        try:
            self.relacher()
        except Exception:
            pass


# ── Test standalone ──────────────────────────────────────────────
if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node('gripper_test')
    pince = SchunkGripper(node)

    try:
        pince.connect()
        time.sleep(1)
        pince.fermer()
        time.sleep(1)
        pince.ouvrir()
        time.sleep(1)
        pince.fermer()
        time.sleep(1)
        pince.ouvrir()
    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        pince.disconnect()
        node.destroy_node()
        rclpy.shutdown()
