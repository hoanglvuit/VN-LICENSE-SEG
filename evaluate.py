from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

model = YOLO(args.model)

metrics = model.val(
    data=args.data,
    imgsz=640,
    device=0
)

print("==== Segmentation metrics ====")
print(f"mAP50-95: {metrics.seg.map:.4f}")
print(f"mAP50:    {metrics.seg.map50:.4f}")
print(f"mAP75:    {metrics.seg.map75:.4f}")
