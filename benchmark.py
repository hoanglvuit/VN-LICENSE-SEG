import time
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--runs", type=int, default=100)
args = parser.parse_args()

model = YOLO(args.model)

# warmup
for _ in range(10):
    model(args.image)

start = time.perf_counter()
for _ in range(args.runs):
    model(args.image)
end = time.perf_counter()

latency = (end - start) / args.runs
fps = 1.0 / latency

print("==== Benchmark ====")
print(f"Model:   {args.model}")
print(f"Latency: {latency*1000:.2f} ms")
print(f"FPS:     {fps:.2f}")
