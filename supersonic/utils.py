from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from typing import cast
import json

def dropout(x : Tensor, p: float=0.5, training:bool=True):
    r"""During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if not training or p == 0.0:
        return x

    mask = Tensor.rand(*x.shape) > p
    return x * mask / (1-p)


def unpack_tensor_to_dict(tensor_data:Tensor):
    """
    Unpack a torch tensor into a Python dictionary.

    Parameters:
    - tensor_data: The torch tensor containing the packed data.

    Returns:
    A Python dictionary containing the unpacked data.
    """
    json_bytes = bytes(tensor_data.numpy())
    json_str = json_bytes.decode("utf-8")
    unpacked_dict = json.loads(json_str)

    return unpacked_dict

def pack_dict_to_tensor(source_dict):
    """
    Pack a dictionary into a torch tensor for storing quant_state items in state_dict.

    Parameters:
    - source_dict: The dictionary to be packed.

    Returns:
    A torch tensor containing the packed data.
    """
    json_str = json.dumps(source_dict)
    json_bytes = json_str.encode("utf-8")
    tensor_data = Tensor(list(json_bytes), dtype=dtypes.uint8)

    return tensor_data

def quantize_to_indices(normalized: Tensor, code: Tensor, quant_type: str) -> Tensor:
    """Convert normalized values to quantization indices."""
    # Broadcast for distance computation
    # normalized: [num_blocks, blocksize] -> [num_blocks, blocksize, 1]
    # code: [16] -> [1, 1, 16]
    normalized_expanded = normalized.unsqueeze(-1)
    code_expanded = code.reshape(1, 1, -1)

    # Compute distances to all quantization values
    diff = cast(Tensor, normalized_expanded - code_expanded)
    distances = diff.abs()

    # Find nearest neighbor indices
    indices = distances.argmin(axis=-1)

    return indices.cast(dtypes.uint8)
