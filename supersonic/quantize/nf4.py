from __future__ import annotations
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from utils import unpack_tensor_to_dict, quantize_to_indices, pack_dict_to_tensor, pack_4bit_pairs
from tinygrad.uop.mathtraits import MathTrait
from typing import Any, Optional, Union, cast

class QuantState:
    """container for quantization state components to work with Params4bit and similar classes"""

    valid_quant_types = ("fp4", "nf4")
    valid_qs_type_keys = [f"bitsandbytes__{x}" for x in valid_quant_types]
    valid_qs_keys = [
        "absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state",
        "quant_type", "blocksize", "dtype", "shape", "nested_blocksize",
        "nested_dtype", "nested_offset",
    ]

    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        quant_type=None,
        dtype=None,
        offset=None,
        state2=None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.dtype = dtype
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None

    def __getitem__(self, idx):
        """ensures compatibility with older quant state scheme with nested lists."""
        if self.nested:
            list_repr = [
                self.absmax, self.shape, self.dtype, self.blocksize,
                [self.offset, self.state2], self.quant_type,
            ]
        else:
            list_repr = [self.absmax, self.shape, self.dtype, self.blocksize, None, self.quant_type]
        return list_repr[idx]

    @classmethod
    def from_dict(cls, qs_dict: dict[str, Any], device: str) -> "QuantState":
        """unpacks components of state_dict into QuantState"""
        device = Device.canonicalize(device)

        # Find quantization state key
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, Tensor)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # Unpack if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            offset = Tensor([float(qs_dict["nested_offset"])], device=device)
            state2 = cls(
                absmax=qs_dict["nested_absmax"].to(device),
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"].to(device),
                dtype=qs_dict["nested_dtype"],
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"].to(device),
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"].to(device),
            dtype=qs_dict["dtype"],
            shape=tuple(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state

    def as_dict(self, packed=False):
        """returns dict of tensors and strings for serialization"""

        if self.quant_type is None:
            raise ValueError("quant_type cannot be None for serialization")
        if self.absmax is None:
            raise ValueError("absmax cannot be None for serialization")
        if self.blocksize is None:
            raise ValueError("blocksize cannot be None for serialization")
        if self.code is None:
            raise ValueError("code cannot be None for serialization")
        if self.dtype is None:
            raise ValueError("dtype cannot be None for serialization")

        qs_dict = {
            "quant_type": self.quant_type,
            "absmax": self.absmax,
            "blocksize": self.blocksize,
            "quant_map": self.code,
            "dtype": str(self.dtype),
            "shape": tuple(self.shape) if self.shape is not None else None,
        }

        if self.nested:
            if self.state2 is None:
                raise ValueError("state2 cannot be None when nested=True")
            if self.offset is None:
                raise ValueError("offset cannot be None when nested=True")
            qs_dict.update({
                "nested_absmax": self.state2.absmax,
                "nested_blocksize": self.state2.blocksize,
                "nested_quant_map": self.state2.code,
                "nested_dtype": str(self.state2.dtype),
                "nested_offset": self.offset.item() if hasattr(self.offset, 'item') else float(self.offset),
            })

        if not packed:
            return qs_dict

        # Packed format for safetensors
        qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, Tensor)}
        non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, Tensor)}
        qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = pack_dict_to_tensor(non_tensor_dict)
        return qs_packed_dict

    def to(self, device):
        """Move quantization state to specified device"""
        device = Device.canonicalize(device)

        if self.code is not None:
            self.code = self.code.to(device)
        if self.absmax is not None:
            self.absmax = self.absmax.to(device)

        if self.nested:
            if self.offset is not None:
                self.offset = self.offset.to(device)
            if self.state2 is not None:
                if self.state2.absmax is not None:
                    self.state2.absmax = self.state2.absmax.to(device)
                if self.state2.code is not None:
                    self.state2.code = self.state2.code.to(device)

    def __eq__(self, other):
        if not isinstance(other, QuantState):
            return False

        if self.absmax is not None and other.absmax is not None:
            absmax_diff = (self.absmax - other.absmax).abs().max()
            absmax_equal = absmax_diff.item() < 1e-6
        else:
            absmax_equal = self.absmax is other.absmax

        if self.code is not None and other.code is not None:
            code_diff = (self.code - other.code).abs().max()
            code_equal = code_diff.item() < 1e-6
        else:
            code_equal = self.code is other.code

        return (
            absmax_equal
            and self.shape == other.shape
            and code_equal
            and self.dtype == other.dtype
            and self.blocksize == other.blocksize
            and self.quant_type == other.quant_type
            and (
                self.offset == other.offset
                if self.offset is not None and other.offset is not None
                else self.offset is other.offset
            )
            and (
                self.state2 == other.state2
                if self.state2 is not None and other.state2 is not None
                else self.state2 is other.state2
            )
        )


def get_4bit_type(typename:str, device=None, blocksize = 64):
    if device is None:
        device = Device.DEFAULT
    else:
        device = Device.canonicalize(device)
    data = None
    if typename == "nf4":
        """
        NF4 Implementation data type
        from https://arxiv.org/pdf/2305.14314
        """
        data = Tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ])
    elif typename == "fp4":
        data = Tensor([0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0])
    elif typename == "int4":
        data = Tensor([7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7])
    elif typename == "af4":
        # Taken from: NF4 Isn't Information Theoretically Optimal (and that's Good)
        # https://arxiv.org/abs/2306.06965
        data = Tensor([
            -1.0,
            -0.69441008,
            -0.51243739,
            -0.3736951,
            -0.25607552,
            -0.14982478,
            -0.04934812,
            0.0,
            0.04273164,
            0.12934483,
            0.21961274,
            0.31675666,
            0.42563882,
            0.55496234,
            0.72424863,
            1.0,
        ])[::-1]
    else:
        raise NotImplementedError("4-bit AbnormalFloats currently only support blocksize 64.")

    if data is None:
        raise NotImplementedError(f"Typename {typename} not supported")

    data = data/data.abs().max()
    return data


def quantize_fp4(
    A: Tensor,
    absmax : Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_storage=dtypes.uint8
):
    if blocksize is None:
        blocksize = 64
    return quantize_4bit(A,absmax,out,blocksize,compress_statistics,"fp4",quant_storage)


def quantize_nf4(
    A: Tensor,
    absmax : Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_storage=dtypes.uint8
):
    if blocksize is None:
        blocksize = 64
    return quantize_4bit(A,absmax,out,blocksize,compress_statistics,"nf4",quant_storage)

def quantize_4bit(
    A: Tensor,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize=None,
    compress_statistics=False,
    quant_type="fp4",
    quant_storage=dtypes.uint8,
) -> tuple[Tensor, QuantState]:

    """
    Quantize tensor A in blocks of 4-bit values

    => Divides tensor A into blocks which are then individually quantized

    Args:
        A(Tensor): Input Tensor. Supports 'float16' or 'float32'
        absmax (optional): Tensor to store absolute max value
        out (optional): Tensor to store result
        blocksize: Size of blocks, Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        compress_statistics: bool to additionally quantize absmax values
        quant_type: The data type to use: 'nf4' or 'fp4", default is fp4
        quant_storage: the dtype of the tensor used to store the result, defaults to dtypes.uint8

    Raises:
        ValueError: Raised when the input data type is not supported.

    Returns:
        Tuple[`Tensor`, `QuantState`]: A tuple containing the quantization results.
        - `Tensor`: The quantized tensor with packed 4-bit values.
        - [`QuantState`]: The state object used to undo the quantization.
    """
    if blocksize is None:
        #blocksize = 64 if not HIP_ENVIRONMENT else 128
        blocksize = 64

    input_shape = A.shape
    code = cast(Tensor, get_4bit_type(quant_type))

    # _out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(
    #     A,
    #     blocksize,
    #     quant_type,
    #     quant_storage,
    # )

    A_flat  = A.reshape(-1)
    n = A_flat.shape[0]

    remainder = n % blocksize
    if remainder != 0:
        padding = blocksize - remainder
        A_flat = A_flat.cat(Tensor.zeros(padding, dtype=A.dtype, device=A.device))
        n = A_flat.shape[0]

    num_blocks = n // blocksize
    A_blocks = A_flat.reshape(num_blocks,blocksize)
    _absmax = A_blocks.abs().max(axis=1,keepdim=False)
    #Normalize each block to [-1, 1]
    normalized = cast(Tensor, A_blocks / (_absmax.unsqueeze(1) + 1e-8))
    normalized = normalized.clamp(-1.0, 1.0)
    assert isinstance(normalized, Tensor), f"Expected Tensor, got {type(normalized)}"
    indices= quantize_to_indices(normalized, code, quant_type)
    packed_out = pack_4bit_pairs(indices)
    if compress_statistics:
        offset = _absmax.mean()
        absmax_centered = cast(Tensor,_absmax - offset)
        qabsmax, state2 = quantize_blockwise(absmax_centered, blocksize=256)
        state = QuantState(
            absmax=qabsmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
            offset=offset,
            state2=state2,
        )
    else:
        state = QuantState(
            absmax=_absmax,
            shape=input_shape,
            dtype=A.dtype,
            blocksize=blocksize,
            code=code,
            quant_type=quant_type,
        )
    if absmax is not None:
        absmax.assign(state.absmax)

    return packed_out , state


def quantize_blockwise(
    A: Tensor,
    code: Optional[Tensor] = None,
    blocksize=4096,
    nested=False,
) -> tuple[Tensor, QuantState]:
    """Simple blockwise quantization for compress_statistics."""

    if code is None:
        code = Tensor.linspace(-1.0, 1.0, 256)

    A_flat = A.reshape(-1)
    n = A_flat.shape[0]

    remainder = n % blocksize
    if remainder != 0:
        padding = blocksize - remainder
        A_flat = A_flat.cat(Tensor.zeros(padding, dtype=A.dtype, device=A.device))
        n = A_flat.shape[0]

    num_blocks = n // blocksize
    A_blocks = A_flat.reshape(num_blocks, blocksize)

    _absmax = A_blocks.abs().max(axis=1, keepdim=False)
    normalized = cast(Tensor, A_blocks / (_absmax.unsqueeze(1) + 1e-8))
    normalized = normalized.clamp(-1.0, 1.0)
    _out = quantize_to_8bit_indices(normalized, code)

    state = QuantState(
        absmax=_absmax,
        code=code,
        blocksize=blocksize,
        dtype=A.dtype
    )

    return _out, state

def quantize_to_8bit_indices(normalized: Tensor, code: Tensor) -> Tensor:
    """Quantize to 8-bit indices using 256-value lookup table."""
    normalized_expanded = normalized.unsqueeze(-1)
    code_expanded = code.reshape(1, 1, -1)
    distances = cast(Tensor, normalized_expanded - code_expanded).abs()
    indices = distances.argmin(axis=-1)
    return indices.cast(dtypes.uint8)


def dequantize_fp4(
    A:Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: Optional[int] = None,
) -> Tensor:
    if blocksize is None:
        blocksize = 64
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


def dequantize_nf4(
    A:Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: Optional[int] = None,
) -> Tensor:
    if blocksize is None:
        blocksize = 64
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")

def dequantize_4bit(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type="fp4",
) -> Tensor:
    """Dequantizes a packed 4-bit quantized tensor.

    The input tensor is dequantized by dividing it into blocks of `blocksize` values.
    The the absolute maximum value within these blocks is used for scaling
    the non-linear dequantization.

    Args:
        A (`Tensor`): The quantized input tensor.
        quant_state ([`QuantState`], *optional*):
            The quantization state as returned by [`quantize_4bit`].
            Required if `absmax` is not provided.
        absmax (`Tensor`, *optional*):
            A tensor containing the scaling values.
            Required if `quant_state` is not provided and ignored otherwise.
        out (`Tensor`, *optional*): A tensor to use to store the result.
        blocksize (`int`, *optional*):
            The size of the blocks. Defaults to 128 on ROCm and 64 otherwise.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
        quant_type (`str`, *optional*): The data type to use: `nf4` or `fp4`. Defaults to `fp4`.

    Raises:
        ValueError: Raised when the input data type or blocksize is not supported.

    Returns:
        `Tensor`: The dequantized tensor.
    """
    if blocksize is None:
        blocksize = 64

    if quant_state is None:
        assert absmax is not None and out is not None
        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=blocksize,
            quant_type=quant_type,
        )
    else:
        absmax = quant_state.absmax
        blocksize = quant_state.blocksize
        quant_type = quant_state.quant_type

    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        if quant_state.offset is not None:
            absmax = cast(Tensor, absmax + quant_state.offset)
        if absmax.dtype != dtypes.float32:
            absmax = absmax.float()
    assert quant_type is not None, "quant_type is null"
    code = cast(Tensor, get_4bit_type(quant_type))

    A_flat = A.flatten()

    dequantized_values = code[A_flat.int()]

    flat_size = int(A_flat.shape[0])
    assert blocksize is not None, "blocksize cannot be None"
    num_blocks = (flat_size + blocksize - 1) // blocksize

    remainder = A_flat.shape[0] % blocksize
    if remainder != 0:
        padding = blocksize - remainder
        dequantized_values = dequantized_values.cat(
            Tensor.zeros(padding, dtype=dequantized_values.dtype, device=dequantized_values.device)
        )

    value_blocks = dequantized_values.reshape(num_blocks, blocksize)

    assert absmax is not None, "absmax cannot be None"
    absmax_expanded = absmax.unsqueeze(1)
    scaled_blocks = cast(Tensor, value_blocks * absmax_expanded)

    result = scaled_blocks.flatten()

    if remainder != 0:
        result = result[:A_flat.shape[0]]

    if quant_state.shape is not None:
        result = result.reshape(quant_state.shape)

    if A.shape[0] == 1:
        result = result.T

    if out is not None:
        out.assign(result)
        return out

    return result


def dequantize_blockwise(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    code: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 4096,
    nested=False,
) -> Tensor:
    """Dequantize a tensor in blocks of values.

    The input tensor is dequantized by dividing it into blocks of blocksize values.
    The absolute maximum value within these blocks is used for scaling
    the non-linear dequantization.

    Args:
        A (Tensor): The quantized input tensor.
        quant_state (QuantState, optional): The quantization state as returned by quantize_blockwise.
            Required if absmax is not provided.
        absmax (Tensor, optional): A tensor containing the scaling values.
            Required if quant_state is not provided and ignored otherwise.
        code (Tensor, optional): A mapping describing the low-bit data type.
            Defaults to a signed 8-bit dynamic type.
            Ignored when quant_state is provided.
        out (Tensor, optional): A tensor to use to store the result.
        blocksize (int, optional): The size of the blocks. Defaults to 4096.
            Valid values are 64, 128, 256, 512, 1024, 2048, and 4096.
            Ignored when quant_state is provided.
        nested (bool, optional): Whether this is a nested quantization call.

    Returns:
        Tensor: The dequantized tensor. The datatype defaults to float32.
    """
    assert quant_state is not None or absmax is not None, "Either quant_state or absmax must be provided"

    if code is None and quant_state is None:
        code = Tensor.linspace(-1.0, 1.0, 256)

    if quant_state is None:
        quant_state = QuantState(
            absmax=absmax,
            code=code,
            blocksize=blocksize,
            dtype=dtypes.float32
        )

    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        if quant_state.offset is not None:
            absmax = cast(Tensor, absmax + quant_state.offset)
        if absmax.dtype != dtypes.float32:
            absmax = absmax.float()

    A_flat = A.flatten()

    assert quant_state.code is not None, "quant_state.code cannot be None"
    assert quant_state.blocksize is not None, "quant_state.blocksize cannot be None"

    dequantized_values = quant_state.code[A_flat.int()]
    blocksize = quant_state.blocksize
    num_blocks = (A_flat.shape[0] + blocksize - 1) // blocksize

    remainder = A_flat.shape[0] % blocksize
    if remainder != 0:
        padding = blocksize - remainder
        dequantized_values = dequantized_values.cat(
            Tensor.zeros(padding, dtype=dequantized_values.dtype, device=dequantized_values.device)
        )
    value_blocks = dequantized_values.reshape(num_blocks, blocksize)
    assert absmax is not None, "absmax cannot be None"
    absmax_expanded = absmax.unsqueeze(1)  # Shape: [num_blocks, 1]
    scaled_blocks = cast(Tensor, value_blocks * absmax_expanded)
    result = scaled_blocks.flatten()

    if remainder != 0:
        result = result[:A_flat.shape[0]]

    if quant_state.dtype and result.dtype != quant_state.dtype:
        result = result.cast(quant_state.dtype)

    if out is not None:
        out.assign(result)
        return out

    return result
