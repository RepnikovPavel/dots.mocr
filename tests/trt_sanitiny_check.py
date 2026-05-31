import sys

try:
    import tensorrt as trt
    print(f"SUCCESS: TensorRT imported successfully. Version: {trt.__version__}")
    sys.exit(0)
except ImportError as e:
    print(f"ERROR: Failed to import TensorRT. Details: {e}")
    sys.exit(1)