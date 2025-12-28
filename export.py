from ultralytics import YOLO
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="runs/segment/train/weights/best.pt")
parser.add_argument("--export_format", type=str, default="onnx")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--simplify", action='store_true')
parser.add_argument("--half", action='store_true')
parser.add_argument("--int8", action='store_true')

def generate_export_name(simplify, half, int8, export_format):
    return f"simplify_{str(simplify).lower()}_" \
           f"fp16_{str(half).lower()}_" \
           f"int8_{str(int8).lower()}.{export_format}"

if __name__ == "__main__":
    args = parser.parse_args()
    model = YOLO(args.model_path)

    export_out = model.export(
        format=args.export_format,
        simplify=args.simplify,
        opset=12,
        dynamic=False,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        device='cuda'
    )

    saved_path = Path(export_out)
    new_name = generate_export_name(args.simplify, args.half, args.int8, args.export_format)
    new_path = saved_path.with_name(new_name)

    if saved_path.resolve() != new_path.resolve():
        saved_path.rename(new_path)
        print(f"✔ Renamed → {new_name}")
    else:
        print(f"⚠ Already correct name → {new_name}")

    print("✔ Export done!")
