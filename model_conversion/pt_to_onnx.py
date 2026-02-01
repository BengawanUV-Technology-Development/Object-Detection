from ultralytics import YOLO

model = YOLO("model.pt")
model.export(format="engine")