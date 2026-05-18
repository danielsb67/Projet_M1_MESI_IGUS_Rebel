#!/usr/bin/env python3
"""
Test standalone YOLO + RealSense D435 (sans ROS2).
IMPORTANT : arrête le node realsense2_camera ROS2 avant de lancer ce script
             (il monopolise la caméra).

Lance : python3 ~/projet_igus/test_yolo_camera.py
Appuie sur 'q' pour quitter, 's' pour sauvegarder une frame.
"""
import sys
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/dbal/projet_igus/src/yolo/yolo/best.pt'
CONF = 0.15

# Mapping des devices connus sur ce PC
DEVICE_NAMES = {
    0: 'Webcam laptop (HD User Facing)',
    1: 'Webcam laptop (miroir)',
    2: 'RealSense D435 — flux profondeur',
    3: 'RealSense D435 — flux IR',
    4: 'RealSense D435 — flux profondeur 2',
    5: 'RealSense D435 — flux IR 2',
    6: 'RealSense D435 — COULEUR RGB  ← celui-ci',
    7: 'RealSense D435 — métadonnées',
}

REALSENSE_COLOR_DEVICE = 6


def check_ros_node_running():
    try:
        result = subprocess.run(['pgrep', '-f', 'realsense2_camera_node'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("\n⚠️  ATTENTION : le node realsense2_camera ROS2 tourne encore !")
            print("   Il monopolise la caméra. Arrête-le avec Ctrl+C dans son terminal.")
            print("   Puis relance ce script.\n")
            return True
    except Exception:
        pass
    return False


def main():
    print("=" * 60)
    print("  Test YOLO + RealSense D435")
    print("=" * 60)

    if check_ros_node_running():
        sys.exit(1)

    print(f"\n[1] Chargement modèle YOLO...")
    model = YOLO(MODEL_PATH)
    print(f"    Classes : {model.names}")
    print(f"    Seuil de confiance : {CONF}")

    print(f"\n[2] Ouverture RealSense D435 (couleur = /dev/video{REALSENSE_COLOR_DEVICE})...")
    cap = cv2.VideoCapture(REALSENSE_COLOR_DEVICE)

    if not cap.isOpened():
        print(f"    ❌ /dev/video{REALSENSE_COLOR_DEVICE} inaccessible")
        print("       Essai des autres devices...")
        for idx in range(8):
            if idx == REALSENSE_COLOR_DEVICE:
                continue
            test = cv2.VideoCapture(idx)
            if test.isOpened():
                ret, f = test.read()
                if ret and f is not None and f.mean() > 5:
                    name = DEVICE_NAMES.get(idx, f'/dev/video{idx}')
                    print(f"    → Utilise /dev/video{idx} : {name}")
                    cap = test
                    break
            test.release()

    if not cap.isOpened():
        print("\n❌ Aucune caméra accessible. Vérifie la connexion USB.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Skip les premières frames (exposition automatique)
    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    print(f"    ✓ Résolution : {int(cap.get(3))}x{int(cap.get(4))},  "
          f"luminosité moyenne : {frame.mean():.1f}")

    print(f"\n[3] Inférence YOLO en direct — mets la roue devant la caméra")
    print(    "    q = quitter   s = sauvegarder la frame courante\n")

    frame_count = 0
    detection_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Perte du flux caméra.")
            break

        frame_count += 1
        results = model.predict(source=frame, conf=CONF, verbose=False)
        r = results[0]
        n = len(r.boxes) if r.boxes is not None else 0

        if n > 0:
            detection_count += 1
            for box in r.boxes:
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                print(f"  ✓ [{frame_count}] {cls}  conf={conf:.2f}  "
                      f"bbox=({x1},{y1})-({x2},{y2})  centre=({cx},{cy})")

        annotated = r.plot()
        label = (f"Frame:{frame_count}  Det:{detection_count}  conf>={CONF} "
                 f"({'DETECTE' if n > 0 else 'rien'})")
        color = (0, 255, 0) if n > 0 else (0, 100, 255)
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("YOLO Test RealSense D435  —  q=quit  s=save", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = f"/tmp/yolo_realsense_{frame_count}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"  → Sauvegardé : {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[FIN] {frame_count} frames, {detection_count} avec détection Roue.")


if __name__ == '__main__':
    main()
