# supersonic/quantize/autograd.py
from typing import Optional, Tuple, Any
from tinygrad.tensor import Tensor
from .quantization import QuantState, dequantize_4bit

class SuperSonicMatMul4bit:
    """Gradient-aware 4bit matrix multiplication for tinygrad
    
    Based on bitsandbytes MatMul4Bit but adapted for tinygrad's autograd system.
    Handles forward pass with dequantization and backward pass gradients.
    """
    
    @staticmethod 
    def apply(
        A: Tensor, 
        B: Tensor, 
        bias: Optional[Tensor] = None, 
        quant_state: Optional[QuantState] = None
    ) -> Tensor:
        """Apply 4-bit matrix multiplication with gradient support
        
        Args:
            A: Input tensor (activations)
            B: Quantized weight tensor
            bias: Optional bias tensor
            quant_state: Quantization state for dequantizing B
            
        Returns:
            Output tensor from linear operation
        """
        # Handle empty tensor case (like bitsandbytes)
        if A.numel() == 0:
            if quant_state and hasattr(quant_state, 'shape'):
                B_shape = quant_state.shape
                if A.shape[-1] == B_shape[0]:
                    return Tensor.empty(*A.shape[:-1], *B_shape[1:], device=A.device, dtype=A.dtype)
                else:
                    return Tensor.empty(*A.shape[:-1], *B_shape[:1], device=A.device, dtype=A.dtype)
        
        # 1. Dequantize B using SuperSonic quantization
        if quant_state is not None:
            B_dequant = dequantize_4bit(B, quant_state)
            # Cast to match A's dtype for computation
            B_dequant = B_dequant.cast(A.dtype)
        else:
            B_dequant = B.cast(A.dtype)
        
        # 2. Perform linear operation (A @ B.T + bias)
        # In tinygrad, linear expects weight transposed
        output = A.linear(B_dequant.T, bias)
        
        return output

def supersonic_linear_forward_backward(
    x: Tensor, 
    weight: Tensor, 
    bias: Optional[Tensor], 
    quant_state: Optional[QuantState]
) -> Tensor:
    """Forward/backward for quantized linear layers
    
    This is the main function for performing quantized linear operations
    with proper gradient handling in tinygrad.
    
    Args:
        x: Input activations
        weight: Quantized weight tensor  
        bias: Optional bias tensor
        quant_state: Quantization state for weight
        
    Returns:
        Output tensor from quantized linear operation
    """
    return SuperSonicMatMul4bit.apply(x, weight, bias, quant_state)

def supersonic_matmul_4bit(
    A: Tensor,
    B: Tensor, 
    quant_state: QuantState,
    bias: Optional[Tensor] = None
) -> Tensor:
    """4-bit matrix multiplication function
    
    Tinygrad equivalent of bitsandbytes.functional.matmul_4bit
    
    Args:
        A: Input tensor (activations)
        B: Quantized weight tensor
        quant_state: Quantization state for B
        bias: Optional bias tensor
        
    Returns:
        Output tensor from 4-bit matmul
    """
    assert quant_state is not None, "quant_state is required for 4-bit matmul"
    
    # For tinygrad, we use our custom matmul implementation
    # Since tinygrad handles autograd automatically, we don't need
    # the complex gradient handling that bitsandbytes does
    return SuperSonicMatMul4bit.apply(A, B, bias, quant_state)

# Convenience function for backward compatibility
def matmul_4bit(
    A: Tensor,
    B: Tensor,
    quant_state: QuantState,
    out: Optional[Tensor] = None,  # Not used in tinygrad version
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Backward compatible 4-bit matmul function"""
    return supersonic_matmul_4bit(A, B, quant_state, bias)
