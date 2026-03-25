#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import threading
import time


class RebelGripperController(Node):
    def __init__(self):
        super().__init__('rebel_gripper_controller')
        self.robot_ip = '192.168.3.11'
        self.robot_port = 3920
        self.sock = None
        self.msg_counter = 1
        self._lock = threading.Lock()
        self._alive_running = False

        self.subscription = self.create_subscription(
            String, '/gripper/command', self.gripper_callback, 10)

        self.get_logger().info("Connexion CRI au robot...")
        if self._connect():
            self.get_logger().info("✅ Gripper prêt !")
            self.get_logger().info("Envoie 'open', 'close' ou 'release' sur /gripper/command")
        else:
            self.get_logger().error("❌ Connexion échouée")

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.robot_ip, self.robot_port))
            self._alive_running = True
            threading.Thread(target=self._alive_loop, daemon=True).start()
            time.sleep(1)
            self._send("CMD SetActive true")
            time.sleep(0.5)
            return True
        except Exception as e:
            self.get_logger().error(f"Connexion: {e}")
            return False

    def _send(self, command):
        with self._lock:
            msg = f"CRISTART {self.msg_counter} {command} CRIEND\n"
            self.msg_counter = (self.msg_counter % 9999) + 1
            try:
                self.sock.sendall(msg.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f"Send: {e}")

    def _alive_loop(self):
        while self._alive_running:
            try:
                self._send("ALIVEJOG 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0")
            except:
                break
            time.sleep(0.2)

    def _set_output(self, pin, state):
        val = "true" if state else "false"
        self._send(f"CMD SetOutput {pin} {val}")
        time.sleep(0.1)

    def gripper_callback(self, msg):
        cmd = msg.data.lower()
        self._send("CMD SetActive true")
        time.sleep(0.3)

        if cmd == "open":
            self.get_logger().info("🔓 OUVRIR (DOut31=true, DOut32=false)")
            self._set_output(32, False)
            self._set_output(31, True)
        elif cmd == "close":
            self.get_logger().info("🔒 FERMER (DOut31=false, DOut32=true)")
            self._set_output(31, False)
            self._set_output(32, True)
        elif cmd == "release":
            self.get_logger().info("⏹️ RELÂCHER (tout à false)")
            self._set_output(31, False)
            self._set_output(32, False)
        else:
            self.get_logger().warning(f"Commande inconnue: {cmd}")

    def destroy_node(self):
        self._alive_running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RebelGripperController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
