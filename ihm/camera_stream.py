#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_stream.py — Streamer ROS → fichiers pour la zone Visualisation de l'IHM.

Tourne dans un sous-processus AVEC l'environnement ROS sourcé (lancé par l'IHM via
``bash -lc "<SIM_ENV> && python3 ihm/camera_stream.py …"``). L'IHM elle-même reste
un processus SANS ROS (leçon segfault Tk/ROS — voir VLA_PROGRESS.md) : le
découplage se fait par fichiers écrits de façon ATOMIQUE (tmp + os.replace),
que l'IHM lit par simple polling mtime.

Sorties (dans --out, idéalement sur /dev/shm = RAM, zéro usure SSD) :
  - <nom>.ppm  par flux caméra — PPM binaire : encodage/décodage quasi gratuit
    (Tk lit le PPM nativement) → fluide à 15 Hz, contrairement au PNG.
  - joints.json — positions /joint_states (pour la vue Robot de l'IHM).

Usage :
    python3 ihm/camera_stream.py --out /dev/shm/ihm_cam_cache \
        --topic front=/front_camera/image --topic wrist=/wrist_camera/image --hz 15
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


class CameraStream(Node):
    """Souscrit N topics image + /joint_states, écrit <out>/… à cadence bornée."""

    def __init__(self, out_dir: str, topics: dict, hz: float, ext: str,
                 max_width: int = 0) -> None:
        super().__init__("ihm_camera_stream")
        self._out = out_dir
        self._ext = ext.lstrip(".")
        self._max_w = max(0, int(max_width))
        os.makedirs(out_dir, exist_ok=True)
        self._latest: dict = {}
        self._dirty: set = set()
        self._joints: dict | None = None
        for name, topic in topics.items():
            self.create_subscription(
                Image, topic,
                lambda msg, n=name: self._on_image(n, msg),
                qos_profile_sensor_data)
            self.get_logger().info(f"flux '{name}' ← {topic}")
        self.create_subscription(JointState, "/joint_states",
                                 self._on_joints, qos_profile_sensor_data)
        self.create_timer(1.0 / max(hz, 0.5), self._flush)

    def _on_image(self, name: str, msg: Image) -> None:
        self._latest[name] = msg
        self._dirty.add(name)

    def _on_joints(self, msg: JointState) -> None:
        self._joints = dict(zip(msg.name, [float(p) for p in msg.position]))
        self._dirty.add("__joints__")

    def _flush(self) -> None:
        # kill_all.sh peut purger /dev/shm entre deux runs → on recrée le dossier.
        os.makedirs(self._out, exist_ok=True)
        for name in list(self._dirty):
            self._dirty.discard(name)
            try:
                if name == "__joints__":
                    if self._joints is not None:
                        # tmp garde l'extension .json (lisibilité) ; os.replace = atomique
                        tmp = os.path.join(self._out, ".joints.tmp.json")
                        with open(tmp, "w") as f:
                            f.write(json.dumps(self._joints))
                        os.replace(tmp, os.path.join(self._out, "joints.json"))
                    continue
                msg = self._latest.get(name)
                if msg is None:
                    continue
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                # step ≥ width*3 (padding de ligne possible) → on rogne proprement.
                arr = arr.reshape(msg.height, msg.step)[:, : msg.width * 3]
                arr = arr.reshape(msg.height, msg.width, 3)
                bgr = arr[:, :, ::-1] if msg.encoding == "rgb8" else arr
                # --max-width : réduit AVANT écriture (moins d'IO et plus de
                # subsample côté Tk). 0 = comportement historique, inchangé.
                if self._max_w and bgr.shape[1] > self._max_w:
                    ratio = self._max_w / bgr.shape[1]
                    bgr = cv2.resize(
                        bgr, (self._max_w, max(1, int(bgr.shape[0] * ratio))),
                        interpolation=cv2.INTER_AREA)
                final = os.path.join(self._out, f"{name}.{self._ext}")
                # tmp DOIT garder l'extension image : cv2.imwrite en déduit le format.
                tmp = os.path.join(self._out, f".{name}.tmp.{self._ext}")
                cv2.imwrite(tmp, bgr)
                os.replace(tmp, final)   # l'IHM ne lit jamais une demi-image
            except Exception as exc:  # noqa: BLE001 — un flux cassé ≠ streamer mort
                self.get_logger().warn(f"écriture {name} : {exc}",
                                       throttle_duration_sec=5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamer ROS → fichiers (IHM).")
    parser.add_argument("--out", default="/dev/shm/ihm_cam_cache",
                        help="Dossier de sortie (défaut : /dev/shm/ihm_cam_cache)")
    parser.add_argument("--topic", action="append", default=[],
                        help="Flux 'nom=/topic/image' (répétable)")
    parser.add_argument("--hz", type=float, default=15.0,
                        help="Cadence max d'écriture par flux (défaut : 15 Hz)")
    parser.add_argument("--ext", default="ppm", choices=("ppm", "png"),
                        help="Format image (ppm = rapide/RAM ; png = compact)")
    parser.add_argument("--max-width", type=int, default=0,
                        help="Largeur max écrite en px (0 = pleine résolution)")
    args = parser.parse_args()

    topics = {}
    for spec in (args.topic or ["front=/front_camera/image",
                                "wrist=/wrist_camera/image"]):
        name, _, topic = spec.partition("=")
        if name and topic:
            topics[name.strip()] = topic.strip()
    if not topics:
        topics = {"front": "/front_camera/image", "wrist": "/wrist_camera/image"}

    rclpy.init()
    node = CameraStream(args.out, topics, args.hz, args.ext, args.max_width)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
