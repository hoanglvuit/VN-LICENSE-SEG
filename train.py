from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

model = YOLO(args.model)
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=640,
    batch=32,
    device=0,
    close_mosaic=20
)
