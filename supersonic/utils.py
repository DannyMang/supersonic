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


def pack_4bit_pairs(indices: Tensor) -> Tensor:
    """Pack pairs of 4-bit indices into uint8 storage"""
    # [5, 10, 3, 15, 7] =>  [5, 10, 3, 15, 7, 0]
    if indices.shape[0] % 2 != 0:
        indices = indices.cat(Tensor([0], device=indices.device, dtype=indices.dtype))
    # Result: [[5, 10], [3, 15], [7, 0]]
    pairs = indices.reshape(-1, 2)
    # pairs[:, 0] = [5, 3, 7]     ← First value of each pair
    # pairs[:, 1] = [10, 15, 0]   ← Second value of each pair

    # Left shift first values by 4 bits (move to high nibble):
    # 5 << 4  = 0101 << 4 = 0101 0000 = 80
    # 3 << 4  = 0011 << 4 = 0011 0000 = 48
    # 7 << 4  = 0111 << 4 = 0111 0000 = 112

    # OR with second values (place in low nibble):
    # 80 | 10  = 0101 0000 | 0000 1010 = 0101 1010 = 90
    # 48 | 15  = 0011 0000 | 0000 1111 = 0011 1111 = 63
    # 112 | 0  = 0111 0000 | 0000 0000 = 0111 0000 = 112
    packed = ((pairs[:, 0] << 4) | pairs[:, 1]).cast(dtypes.uint8)

    #Final result: [90, 63, 112] (3 bytes storing 6 values)
    return packed

def unpack_4bit_pairs(packed: Tensor, target_length: int) -> Tensor:
    """Unpack uint8 storage back to 4-bit indices"""
    high_nibble = ((packed >> 4) & 0xF).cast(dtypes.uint8)
    low_nibble = (packed & 0xF).cast(dtypes.uint8)
    # Interleave high and low nibbles
    unpacked = high_nibble.unsqueeze(1).cat(low_nibble.unsqueeze(1), dim=1).flatten()
    return unpacked[:target_length]
