python export.py --export_format onnx 
python export.py --export_format onnx --simplify 
python export.py --export_format onnx --simplify --half 
python export.py --export_format engine 
python export.py --export_format engine --simplify 
python export.py --export_format engine --simplify --half 

# not fixed issues with int8 export yet
# python export.py --export_format engine --simplify --int8 
