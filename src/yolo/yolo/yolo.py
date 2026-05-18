from ultralytics import YOLO
import cv2, time


#model = YOLO(r"D:\AI\best.pt")
model = YOLO(r"D:\AI\yolo11n.pt")



cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection
    results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Rendre le tableau NumPy inscriptible
    annotated_frame = annotated_frame.copy()
    
    # Calcul FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Afficher le FPS sur l’image
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()