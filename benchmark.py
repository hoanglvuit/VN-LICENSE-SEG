import argparse
import time
import os
import csv
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data", type=str, default="data.yaml")
parser.add_argument("--image", type=str, default="dataset/images/val/carlong_0173.png")
parser.add_argument("--runs", type=int, default=1000)
parser.add_argument("--out_dir", type=str, default="benchmark")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
csv_path = os.path.join(args.out_dir, "results.csv")

# =====================
# Load model
# =====================
model = YOLO(args.model_path)

# =====================
# Evaluate (Validation)
# =====================
metrics = model.val(
    data=args.data,
    imgsz=640,
    device="cuda",
    save=False,
    plots=False
)

map_50_95 = metrics.seg.map
map_50 = metrics.seg.map50
map_75 = metrics.seg.map75

# =====================
# Benchmark + VRAM
# =====================
torch.cuda.reset_peak_memory_stats()

# warmup
for _ in range(10):
    model(args.image)

start = time.perf_counter()
for _ in range(args.runs):
    model(args.image)
end = time.perf_counter()

latency = (end - start) / args.runs
fps = 1.0 / latency

vram_mb = torch.cuda.max_memory_allocated() / 1024**2

# =====================
# Print
# =====================
print("==== Segmentation metrics ====")
print(f"mAP50-95: {map_50_95:.4f}")
print(f"mAP50:    {map_50:.4f}")
print(f"mAP75:    {map_75:.4f}")

print("==== Benchmark ====")
print(f"Latency: {latency*1000:.2f} ms")
print(f"FPS:     {fps:.2f}")
print(f"VRAM:    {vram_mb:.2f} MB")

# =====================
# Save to CSV
# =====================
header = [
    "model_path",
    "mAP50-95",
    "mAP50",
    "mAP75",
    "latency_ms",
    "FPS",
    "VRAM_MB"
]

row = [
    args.model_path,
    f"{map_50_95:.4f}",
    f"{map_50:.4f}",
    f"{map_75:.4f}",
    f"{latency*1000:.2f}",
    f"{fps:.2f}",
    f"{vram_mb:.2f}"
]

file_exists = os.path.isfile(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(row)

print(f"\n✅ Saved results to {csv_path}")
