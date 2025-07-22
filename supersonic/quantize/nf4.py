
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from utils import unpack_tensor_to_dict

from typing import Any, Optional, Union

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
        data = Tensor([[7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]])
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

    pass
   # TO-DO JULY 22

def dequantize_4bit():
    pass


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
