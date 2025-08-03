# supersonic/quantize/__init__.py

# Core quantization functions and classes
from .quantization import quantize_4bit, dequantize_4bit, QuantState

# 4-bit quantized modules and parameters
from .modules import (
    SuperSonicParams4bit,
    SuperSonicLinear4bit, 
    LinearFP4,
    LinearNF4
)

# Model utilities for loading and preparation  
from .model_utils import (
    prepare_model_for_supersonic_training,
    load_model_with_supersonic,
    replace_linear_with_supersonic
)

# Autograd functions for 4-bit operations
from .autograd import (
    SuperSonicMatMul4bit,
    supersonic_matmul_4bit,
    supersonic_linear_forward_backward,
    matmul_4bit
)

# LoRA and QLoRA compatibility (if available)
try:
    from .lora import Linear as LoRALinear
    _HAS_LORA = True
except ImportError:
    LoRALinear = None
    _HAS_LORA = False
    
try:
    from .qlora import SuperSonicLinear4bit as QLoRALinear
    _HAS_QLORA = True  
except ImportError:
    QLoRALinear = SuperSonicLinear4bit  # Fallback to base implementation
    _HAS_QLORA = False

# Public API
__all__ = [
    # Core quantization
    'quantize_4bit', 'dequantize_4bit', 'QuantState',
    
    # Quantized modules and parameters
    'SuperSonicParams4bit', 'SuperSonicLinear4bit', 'LinearFP4', 'LinearNF4',
    
    # Model utilities
    'prepare_model_for_supersonic_training', 'load_model_with_supersonic', 'replace_linear_with_supersonic',
    
    # Autograd functions
    'SuperSonicMatMul4bit', 'supersonic_matmul_4bit', 'supersonic_linear_forward_backward', 'matmul_4bit',
    
    # LoRA/QLoRA (if available)
    'LoRALinear', 'QLoRALinear',
]

# Version info
__version__ = '0.1.0'

# Convenience imports for common use cases
# Users can do: from supersonic.quantize import SuperSonicLinear4bit, prepare_model_for_supersonic_training
