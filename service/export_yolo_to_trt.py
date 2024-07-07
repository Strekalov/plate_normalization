from ultralytics import YOLO

model = YOLO("weights/yolov8_keypoints.pt")
model.fuse()
model.export(
    format="tensorrt",
    imgsz=640,
    half=True,
    simplify=True,
    device=0,
)


model = YOLO("weights/yolov10_letters_detector.pt")
model.fuse()
model.export(
    format="tensorrt",
    imgsz=640,
    half=True,
    simplify=True,
    device=0,
)


