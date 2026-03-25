from ultralytics import YOLO

model = YOLO("yolo11s.pt")   
model.train(
    data="/home/Downloads/roue_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/detect",
    name="robot_objects"
)
