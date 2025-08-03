# supersonic/quantize/__init__.py
from .quantization import quantize_4bit, dequantize_4bit, QuantState
from .modules import SuperSonicParams4bit, SuperSonicLinear4bit, replace_linear_with_supersonic
from ..utils import prepare_model_for_supersonic_training, load_quantized_model
from .qlora import SuperSonicLinear4bit as QLoRALinear  # Keep existing alias
from .lora import Linear as LoRALinear
from .autograd import SuperSonicMatMul4bit

__all__ = [
    # Core quantization
    'quantize_4bit', 'dequantize_4bit', 'QuantState',
    # Quantized modules
    'SuperSonicParams4bit', 'SuperSonicLinear4bit',
    # Model utilities
    'prepare_model_for_supersonic_training', 'load_quantized_model',
    # Legacy/compatibility
    'QLoRALinear', 'LoRALinear',
    # Autograd
    'SuperSonicMatMul4bit'
]
