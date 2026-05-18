from ultralytics import YOLO
import cv2, time

model = YOLO(r"D:\AI\best.pt")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=320, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()
    annotated_frame = annotated_frame.copy()

    # Récupération des boîtes détectées
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Centre de la roue
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # afficher sur l'image
        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(annotated_frame, f"({cx},{cy})", (cx+10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
