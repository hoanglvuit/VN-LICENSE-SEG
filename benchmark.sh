python benchmark.py --model_path runs/segment/train/weights/best.pt

python benchmark.py --model_path runs/segment/train/weights/simplify_false_fp16_false_int8_false.onnx 
python benchmark.py --model_path runs/segment/train/weights/simplify_true_fp16_false_int8_false.onnx 
python benchmark.py --model_path runs/segment/train/weights/simplify_true_fp16_true_int8_false.onnx 

python benchmark.py --model_path runs/segment/train/weights/simplify_false_fp16_false_int8_false.engine
python benchmark.py --model_path runs/segment/train/weights/simplify_true_fp16_false_int8_false.engine
python benchmark.py --model_path runs/segment/train/weights/simplify_true_fp16_true_int8_false.engine

