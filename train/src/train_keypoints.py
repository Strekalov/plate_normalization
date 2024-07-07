from ultralytics import YOLO


model = YOLO("yolov8s-pose.pt")

model.train(
    data="configs/plate_keypoints.yaml",
    imgsz=640,
    batch=128,
    deterministic=False,
    pose=32,
    lr0=0.001,
    lrf=0.01,
    optimizer="SGD",
    seed=777,
    scale=1,
    fliplr=0.5,
    mosaic=0,
    cache=False,
    workers=12,
    project="runs/license_plate_number_keypoints",
    name="yolov8s-pose_",
    device="0",
    epochs=20,
)
