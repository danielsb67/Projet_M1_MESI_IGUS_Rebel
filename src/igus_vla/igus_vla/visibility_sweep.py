#!/usr/bin/env python3
"""
visibility_sweep.py — Mesure EMPIRIQUE de la zone visible par la caméra front.

But : décider, données à l'appui, quelles positions de pick exclure parce que la
roulette n'y est PAS visible (cachée par le bras / hors-cadre). Plutôt que de se
fier à un modèle géométrique théorique, on MESURE :

  Pour chaque position (x, y) d'une grille couvrant l'anneau accessible :
    1. téléporte la roulette là (publie /object_position_in_world → gripper_shim) ;
    2. attend une image fraîche de /front_camera/image ;
    3. détecte les pixels ORANGE de la roulette (RGB≈217,51,26) ;
    4. note visible (assez de pixels) ou non, + le nb de pixels.

Sortie :
  - CSV  : datasets/visibility/visibility_sweep_<stamp>.csv  (x,y,dist,visible,pixels)
  - carte ASCII dans les logs (V=visible, .=caché)
  - proposition de contrainte (secteur angulaire / rayons) à reporter dans
    record_orchestrator._sample_object_xy.

Ne lance NI expert NI recorder. Lancé via visibility_sweep.launch.py.
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image


class VisibilitySweep(Node):
    def __init__(self) -> None:
        super().__init__("visibility_sweep")
        # Grille de balayage (mêmes bornes que l'anneau v3 + un peu de marge)
        self.step = float(self.declare_parameter("step", 0.06).value)
        self.r_min = float(self.declare_parameter("r_min", 0.15).value)
        self.r_max = float(self.declare_parameter("r_max", 0.54).value)
        self.y_min = float(self.declare_parameter("y_min", -0.52).value)
        self.place_x = float(self.declare_parameter("place_x", 0.0).value)
        self.place_y = float(self.declare_parameter("place_y", 0.25).value)
        self.bin_keepout = float(self.declare_parameter("bin_keepout", 0.222).value)
        self.object_z = float(self.declare_parameter("object_z", 0.018).value)
        # Délais — on ne « regarde » pas à l'œil : on attend juste assez de frames
        # FRAÎCHES après la téléportation pour une détection fiable, puis on enchaîne.
        self.settle = float(self.declare_parameter("settle", 12.0).value)
        self.fresh_frames = int(self.declare_parameter("fresh_frames", 2).value)  # frames post-téléport
        self.frame_timeout = float(self.declare_parameter("frame_timeout", 0.4).value)  # garde-fou/pos
        self.poll = float(self.declare_parameter("poll", 0.01).value)  # granularité d'attente
        # Détection : seuil pixels orange pour déclarer "visible"
        self.pixel_thresh = int(self.declare_parameter("pixel_thresh", 60).value)
        self.raw_root = str(self.declare_parameter("raw_root", "datasets/visibility").value)

        self._latest = None  # dernière image RGB (np.uint8 HxWx3)
        self._frame_id = 0   # incrémenté à chaque image reçue (attente de fraîcheur)
        self._bridge = None
        try:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        except ImportError:
            self.get_logger().error("cv_bridge indisponible — impossible de lire la caméra.")

        self.create_subscription(Image, "/front_camera/image",
                                 self._on_image, qos_profile_sensor_data)
        self._obj_pub = self.create_publisher(PointStamped, "/object_position_in_world", 10)

        import threading
        threading.Thread(target=self._run, daemon=True).start()

    def _on_image(self, msg: Image) -> None:
        if self._bridge is None:
            return
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._latest = bgr[:, :, ::-1].copy()  # → RGB
            self._frame_id += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"décodage image: {exc}")

    def _publish_object(self, x: float, y: float) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x, msg.point.y, msg.point.z = x, y, self.object_z
        self._obj_pub.publish(msg)

    @staticmethod
    def _count_orange(rgb: np.ndarray) -> int:
        """Compte les pixels orange-rouge de la roulette (RGB≈217,51,26)."""
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        mask = (r > 120) & (g < 95) & (b < 85) & (r - g > 70) & (r - b > 90)
        return int(mask.sum())

    def _grid_positions(self):
        """Génère les positions valides (anneau, hors bac, pas derrière caméra)."""
        n = int(math.ceil(self.r_max / self.step))
        pts = []
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                x, y = ix * self.step, iy * self.step
                d = math.hypot(x, y)
                if d < self.r_min or d > self.r_max:
                    continue
                if y < self.y_min:
                    continue
                if (abs(x - self.place_x) < self.bin_keepout
                        and abs(y - self.place_y) < self.bin_keepout):
                    continue
                pts.append((x, y))
        return pts

    def _run(self) -> None:
        log = self.get_logger()
        log.info(f"=== BALAYAGE VISIBILITÉ === settle {self.settle:.0f}s puis "
                 f"grille pas={self.step} m. Détection roulette orange, "
                 f"seuil {self.pixel_thresh} px.")
        time.sleep(self.settle)
        if self._latest is None:
            log.warn("Aucune image reçue après settle — j'attends encore 5 s…")
            time.sleep(5.0)

        positions = self._grid_positions()
        log.info(f"{len(positions)} positions à tester (cadence = vitesse caméra, "
                 f"{self.fresh_frames} frames/pos, timeout {self.frame_timeout:.2f}s).")
        t0 = time.time()
        results = []  # (x, y, dist, visible, pixels)
        for i, (x, y) in enumerate(positions):
            if not rclpy.ok():
                break
            # Téléporte puis attend des frames FRAÎCHES (le rendu reflète la new pose).
            start_id = self._frame_id
            self._publish_object(x, y)
            deadline = time.time() + self.frame_timeout
            while (self._frame_id - start_id) < self.fresh_frames and time.time() < deadline:
                time.sleep(self.poll)
            img = self._latest
            px = self._count_orange(img) if img is not None else 0
            visible = px >= self.pixel_thresh
            results.append((x, y, math.hypot(x, y), visible, px))
            if (i + 1) % 25 == 0:
                vis_so_far = sum(1 for r in results if r[3])
                rate = (i + 1) / max(1e-3, time.time() - t0)
                log.info(f"  {i+1}/{len(positions)} — visibles {vis_so_far}/{i+1} "
                         f"({rate:.1f} pos/s)")
        log.info(f"Balayage de {len(results)} positions en {time.time()-t0:.0f}s.")

        self._report(results)
        log.info("Balayage terminé. Tu peux arrêter (⏹).")

    def _report(self, results) -> None:
        log = self.get_logger()
        if not results:
            log.warn("Aucun résultat.")
            return
        n = len(results)
        vis = sum(1 for r in results if r[3])
        log.info(f"\n=== RÉSULTAT : {vis}/{n} visibles ({100*vis/n:.0f}%) ===")

        # CSV
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.raw_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"visibility_sweep_{stamp}.csv"
        with open(csv_path, "w") as f:
            f.write("x,y,dist,visible,pixels\n")
            for x, y, d, v, px in results:
                f.write(f"{x:.4f},{y:.4f},{d:.4f},{int(v)},{px}\n")
        log.info(f"CSV → {csv_path}")

        # Carte ASCII
        cells = {(round(x / self.step), round(y / self.step)): v
                 for x, y, _, v, _ in results}
        gxs = [k[0] for k in cells]
        gys = [k[1] for k in cells]
        lines = ["", "=== CARTE VISIBILITÉ (V=visible, x=caché) ==="]
        for gy in range(max(gys), min(gys) - 1, -1):
            row = f"{gy*self.step:+.2f} "
            for gx in range(min(gxs), max(gxs) + 1):
                if (gx, gy) in cells:
                    row += "V" if cells[(gx, gy)] else "x"
                else:
                    row += " "
            lines.append(row)
        log.info("\n".join(lines))

        # Proposition : secteur angulaire visible (min/max angle des points visibles)
        vis_pts = [(x, y) for x, y, _, v, _ in results if v]
        if vis_pts:
            angles = sorted(math.degrees(math.atan2(y, x)) for x, y in vis_pts)
            dists = sorted(math.hypot(x, y) for x, y in vis_pts)
            log.info(f"\n=== ZONE VISIBLE (à reporter dans _sample_object_xy) ===")
            log.info(f"  Rayon visible   : [{dists[0]:.3f}, {dists[-1]:.3f}] m")
            log.info(f"  Angle visible   : [{angles[0]:.0f}°, {angles[-1]:.0f}°] "
                     f"(0°=+x, 90°=+y, ±180°=-x)")
            # Détecte un éventuel trou angulaire (occlusion colonne au milieu)
            gaps = []
            for a, b in zip(angles, angles[1:]):
                if b - a > 15:
                    gaps.append((a, b))
            if gaps:
                log.info(f"  TROU(s) angulaire(s) (occlusion ?) : "
                         + ", ".join(f"[{a:.0f}°,{b:.0f}°]" for a, b in gaps))

        # VERDICT : visibilité DANS la zone gardée par _sample_object_xy (v3).
        # Doit valoir 100% si la règle d'exclusion couvre bien les zones cachées.
        def kept(x, y, d):
            a = math.degrees(math.atan2(y, x))
            if a <= -140.0 or a >= 162.0:      # ombre colonne
                return False
            if 12.0 <= a <= 48.0 and d > 0.47:  # bord avant-droit
                return False
            return True
        kept_res = [(x, y, v) for x, y, d, v, _ in results if kept(x, y, d)]
        if kept_res:
            kv = sum(1 for _, _, v in kept_res if v)
            kn = len(kept_res)
            log.info(f"\n=== VERDICT ZONE GARDÉE (règle _sample_object_xy v3) ===")
            log.info(f"  Visibilité dans la zone gardée : {kv}/{kn} ({100*kv/kn:.0f}%)")
            misses = [(x, y) for x, y, v in kept_res if not v]
            if misses:
                log.warn(f"  ⚠ {len(misses)} position(s) CACHÉE(S) dans la zone gardée "
                         f"→ resserrer les exclusions :")
                for x, y in misses:
                    log.warn(f"     ({x:+.2f}, {y:+.2f}) angle={math.degrees(math.atan2(y,x)):+.0f}°")
            else:
                log.info("  ✅ 100% visible dans la zone gardée — règle respectée.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisibilitySweep()
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
