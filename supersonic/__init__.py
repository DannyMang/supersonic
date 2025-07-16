"""
SuperSonic: Ultra-fast LLM fine-tuning with quantization
Built on TinyGrad! By Daniel Ung, @danielung19 on X!
"""

__version__ = "0.1.0"
__author__ = "Daniel Ung"

from tinygrad import Device
def _check_gpu():
    try:
        if Device.DEFAULT == "CPU":
            print("⚠️  Warning: No GPU detected. UltraLM will be significantly slower on CPU.")
            print("   For best performance, use NVIDIA GPU with CUDA or AMD GPU with ROCm")
        else:
            print(f"✅ GPU detected: {Device.DEFAULT}")
    except Exception as e:
        print(f"⚠️  Warning: Could not detect GPU: {e}")

def _check_dependencies():
    missing = []

    try:
        import triton
    except ImportError:
        missing.append("triton")

    try:
        import bitsandbytes
    except ImportError:
        missing.append("bitsandbytes")

    if missing:
        print(f"⚠️  Warning: Missing optional dependencies: {', '.join(missing)}")
        print("   Install with: pip install triton bitsandbytes")

_check_gpu()
_check_dependencies()

# Main API exports (not yet implemented)
#from .qlora import QLoRAModel, QLoRALinear
#from .training import UltraTrainer
#from .export import export_to_vllm, export_to_gguf
