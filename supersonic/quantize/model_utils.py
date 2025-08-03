from typing import Any, Dict, Optional
from pathlib import Path
from tinygrad import nn
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from tinygrad.dtype import dtypes
from .modules import SuperSonicLinear4bit


def prepare_model_for_supersonic_training(model: Any, config: Dict[str, Any]) -> Any:
    """Replace standard Linear layers with SuperSonicLinear4bit layers.

    Based on bitsandbytes replace_8bit_linear functionality but adapted for tinygrad.
    This function recursively traverses a tinygrad model and replaces nn.Linear layers
    with SuperSonicLinear4bit quantized layers.

    Args:
        model: The original tinygrad model with standard linear layers.
        config: Configuration dict for quantization containing:
            - compute_dtype: Compute dtype for operations
            - compress_statistics: Whether to compress quantization statistics
            - quant_type: "fp4" or "nf4"
            - quant_storage: Storage dtype for quantized weights
            - device: Target device
            - blocksize: Block size for quantization (default: 64)

    Returns:
        Modified model with SuperSonicLinear4bit layers.
    """
    def replace_linear_recursive(module, prefix=""):
        """Recursively replace Linear layers in the module"""
        # Get all attributes of the module (tinygrad models store layers as attributes)
        for name in list(vars(module).keys()):
            if name.startswith('_'):  # Skip private attributes
                continue

            submodule = getattr(module, name)
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this is a tinygrad Linear layer
            if isinstance(submodule, nn.Linear):
                # Extract dimensions from the Linear layer
                in_features = submodule.weight.shape[1]
                out_features = submodule.weight.shape[0]
                has_bias = submodule.bias is not None

                print(f"Found Linear layer {full_name}: {in_features} -> {out_features}, bias={has_bias}")

                # Create replacement SuperSonicLinear4bit layer
                new_layer = SuperSonicLinear4bit(
                    input_features=in_features,
                    output_features=out_features,
                    bias=has_bias,
                    compute_dtype=config.get("compute_dtype", None),
                    compress_statistics=config.get("compress_statistics", True),
                    quant_type=config.get("quant_type", "fp4"),
                    quant_storage=config.get("quant_storage", dtypes.uint8),
                    device=config.get("device", None)
                )

                # Copy weights and bias from original layer
                new_layer.weight._tensor = submodule.weight.detach()  # Copy the tensor data
                if has_bias:
                    assert submodule.bias is not None
                    new_layer.bias = submodule.bias.detach()

                # Replace the layer in the model
                setattr(module, name, new_layer)
                print(f"✓ Replaced {full_name} with SuperSonicLinear4bit")

            elif hasattr(submodule, '__dict__') and len(vars(submodule)) > 0:
                # Recursively process submodules that have attributes
                replace_linear_recursive(submodule, full_name)

    print(f"Starting linear layer replacement with config: {config}")
    replace_linear_recursive(model)
    print("Linear layer replacement completed")
    return model


def load_model_with_supersonic(model_path: str, config: Dict[str, Any], model_class: Any, **model_kwargs) -> Any:
    """Load a model and prepare it for SuperSonic quantization.

    This is the main function for loading models with SuperSonic quantization support.
    It follows tinygrad's typical model loading pattern.

    Args:
        model_path: Path to model weights (.safetensors, .pth, etc.)
        config: SuperSonic quantization configuration
        model_class: The model class to instantiate (e.g., Transformer, etc.)
        **model_kwargs: Additional arguments for model initialization

    Returns:
        Model instance with SuperSonic quantized layers and weights loaded.
    """
    print(f"Loading model from {model_path} with SuperSonic quantization")

    # 1. Create the model instance
    model = model_class(**model_kwargs)
    # 2. Replace Linear layers with SuperSonicLinear4bit
    model = prepare_model_for_supersonic_training(model, config)
    # 3. Load weights using tinygrad's loading functions
    if isinstance(model_path, str):
        path = Path(model_path)

    if path.suffix == '.safetensors':
        weights = safe_load(str(model_path))
    elif path.suffix in ['.pth', '.bin']:
        weights = torch_load(str(model_path))
    else:
        raise ValueError(f"Unsupported model file format: {path.suffix}")

    # 4. Load the state dict into the model
    print(f"Loading {len(weights)} weight tensors...")
    load_state_dict(model, weights, strict=False, verbose=True)

    print("Model loaded successfully with SuperSonic quantization!")
    return model


# Utility function for the stub in modules.py
def replace_linear_with_supersonic(model: Any, config: Dict[str, Any]) -> Any:
    """Convenience function - alias for prepare_model_for_supersonic_training"""
    return prepare_model_for_supersonic_training(model, config)
