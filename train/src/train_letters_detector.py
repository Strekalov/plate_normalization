from ultralytics import YOLO

model = YOLO("yolov10x.pt")

model.train(
    data="configs/letters_detector.yaml",
    imgsz=640,
    batch=12,
    deterministic=False,
    lr0=0.001,
    lrf=0.01,
    optimizer="SGD",
    seed=777,
    single_cls=True,
    scale=0.8,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1,
    close_mosaic=5,
    mixup=0.15,
    copy_paste=0.3,
    cache=False,
    workers=12,
    project="runs/letters_detector",
    name="yolov10x_letters_detector",
    device="0",
    epochs=40,
)
