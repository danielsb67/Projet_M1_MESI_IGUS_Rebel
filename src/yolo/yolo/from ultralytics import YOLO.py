from ultralytics import YOLO
import cv2
import time


from ultralytics import YOLO
model = YOLO("D:/AI/best.pt")
model.export(format="onnx")

# Load ONNX model for high speed inference
model = YOLO(r"D:\AI\best.onnx")  # ⚠️ ضع هنا best.onnx بدل best.pt


# Open webcam
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fast inference
    results = model(frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot().copy()

    # Loop through detected boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Compute center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw center
        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(annotated_frame, f"({cx},{cy})",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    # FPS computation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    # Show video
    cv2.imshow("YOLO ONNX Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
