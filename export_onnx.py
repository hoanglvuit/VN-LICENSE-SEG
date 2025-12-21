from ultralytics import YOLO
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, required=True)
args = parser.parse_args()

model = YOLO(args.weights)

model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False
)

print("Export done")
