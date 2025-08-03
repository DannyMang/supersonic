from tinygrad.tensor import Tensor
from typing import Optional, Any, Union
from tinygrad.dtype import dtypes
from .quantization import QuantState, quantize_4bit, dequantize_4bit

class SuperSonicParams4bit:
    """Tinygrad equivalent of bitsandbytes Params4bit - using composition instead of inheritance"""
    def __init__(
        self,
        data: Optional[Tensor] = None,
        requires_grad=False,  # quantized weights should be frozen by default
        quant_state=None,
        blocksize: Optional[int] = None,
        compress_statistics: bool = True,
        quant_type: str = "fp4",
        quant_storage=dtypes.uint8,
        module: Optional["SuperSonicLinear4bit"] = None,
        is_quantized: bool = False,
    ):
        if data is None:
            data = Tensor.zeros(0)
        if blocksize is None:
            blocksize = 64

        self._tensor = data
        if hasattr(data, 'requires_grad'):
            self._tensor.requires_grad = requires_grad

        # SuperSonic specific attributes
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.is_quantized = is_quantized
        self.module = module

    @property
    def requires_grad(self):
        return getattr(self._tensor, 'requires_grad', False)

    @requires_grad.setter
    def requires_grad(self, value):
        self._tensor.requires_grad = value

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def device(self):
        return self._tensor.device

    def numpy(self):
        return self._tensor.numpy()

    def contiguous(self):
        return SuperSonicParams4bit(
            data=self._tensor.contiguous(),
            requires_grad=self.requires_grad,
            quant_state=self.quant_state,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            is_quantized=self.is_quantized,
            module=self.module
        )

    @property
    def T(self):
        return SuperSonicParams4bit(
            data=self._tensor.T,
            requires_grad=self.requires_grad,
            quant_state=self.quant_state,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            is_quantized=self.is_quantized,
            module=self.module
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tensor_data"] = self._tensor
        return state

    def __setstate__(self, state):
        self._tensor = state["tensor_data"]
        self.blocksize = state["blocksize"]
        self.compress_statistics = state["compress_statistics"]
        self.quant_type = state["quant_type"]
        self.quant_state = state["quant_state"]
        self.quant_storage = state["quant_storage"]
        self.is_quantized = state["is_quantized"]
        self.module = state["module"]

    def __deepcopy__(self, memo):
        # Create a deep copy of the tensor data
        tensor_copy = Tensor(self._tensor.numpy())  # Deep copy via numpy
        tensor_copy.requires_grad = self.requires_grad

        # Deep copy quantization state if it exists
        quant_state_copy = None
        if self.quant_state is not None:
            # Create a new QuantState with copied tensors
            quant_state_copy = QuantState(
                absmax=Tensor(self.quant_state.absmax.numpy()) if self.quant_state.absmax is not None else None,
                shape=self.quant_state.shape,
                code=Tensor(self.quant_state.code.numpy()) if self.quant_state.code is not None else None,
                blocksize=self.quant_state.blocksize,
                quant_type=self.quant_state.quant_type,
                dtype=self.quant_state.dtype,
                offset=Tensor(self.quant_state.offset.numpy()) if self.quant_state.offset is not None else None,
                state2=self.quant_state.state2  # This might need deeper copying too
            )

        return SuperSonicParams4bit(
            data=tensor_copy,
            requires_grad=self.requires_grad,
            quant_state=quant_state_copy,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            is_quantized=self.is_quantized,
            module=self.module
        )

    def __copy__(self):
        return SuperSonicParams4bit(
            data=self._tensor,  # Shallow copy
            requires_grad=self.requires_grad,
            quant_state=self.quant_state,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            is_quantized=self.is_quantized,
            module=self.module
        )

    @classmethod
    def from_prequantized(
        cls,
        data: Tensor,
        quantized_stats: dict[str, Any],
        requires_grad: bool = False,
        device="cuda",
        module: Optional["SuperSonicLinear4bit"] = None,
        **kwargs,
    ) -> "SuperSonicParams4bit":
        self = cls(data.to(device), requires_grad=requires_grad)
        self.requires_grad = requires_grad
        self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.is_quantized = True

        self.quant_storage = data.dtype
        self.module = module

        if self.module is not None:
            self.module.quant_state = self.quant_state

        return self

    def _quantize(self, device):
        """Quantize the weights using SuperSonic quantization"""
        w = self.contiguous().to(device)
        # Pass the underlying tensor to quantize_4bit
        w_4bit, quant_state = quantize_4bit(
            w._tensor,  # Pass the underlying tensor
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
        )
        new_param = SuperSonicParams4bit(
            data=w_4bit,
            requires_grad=self.requires_grad,
            quant_state=quant_state,
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            is_quantized=True,
            module=self.module
        )

        if self.module is not None:
            self.module.quant_state = quant_state

        return new_param

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device: Optional[Union[int, str]] = None):
        device_str = "cuda" if device is None else f"cuda:{device}" if isinstance(device, int) else device
        return self.to(device=device_str)

    def xpu(self, device: Optional[Union[int, str]] = None):
        device_str = "xpu" if device is None else f"xpu:{device}" if isinstance(device, int) else device
        return self.to(device=device_str)

    def to(self, device=None):
        """Move tensor to device, triggering quantization if needed"""
        if device is not None and device != "meta" and not self.is_quantized:
            return self._quantize(device)
        else:
            # Move existing tensor using tinygrad's to() method
            new_tensor = self._tensor.to(device) if device is not None else self._tensor

            new_param = SuperSonicParams4bit(
                data=new_tensor,
                requires_grad=self.requires_grad,
                quant_state=self.quant_state,
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                is_quantized=self.is_quantized,
                module=self.module
            )

            if self.quant_state is not None and device is not None:
                assert new_param.quant_state is not None #type check to remove annoying zed type checker bruh
                new_param.quant_state.to(device)

            return new_param


class SuperSonicLinear4bit:
    """4-bit quantized linear layer with LoRA support"""
    """
    This class is the base module for the 4-bit quantization algorithm presented in [QLoRA](https://arxiv.org/abs/2305.14314).
    QLoRA 4-bit linear layers uses blockwise k-bit quantization under the hood, with the possibility of selecting various
    compute datatypes such as FP4 and NF4.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear4bit module, then call `quantized_module.to("cuda")` to quantize the fp16 / bf16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bnb.nn import Linear4bit

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    quantized_model = nn.Sequential(
        Linear4bit(64, 64),
        Linear4bit(64, 64)
    )

    quantized_model.load_state_dict(fp16_model.state_dict())
    quantized_model = quantized_model.to(0) # Quantization happens here
    ```
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="fp4",
        quant_storage=dtypes.uint8,
        device=None,
    ):
        """
        Initialize Linear4bit class.

        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        self.in_features = input_features
        self.out_features = output_features

        weight_data = Tensor.uniform(output_features, input_features, low=-0.1, high=0.1)
        self.weight = SuperSonicParams4bit(
            data=weight_data,
            requires_grad=False,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=quant_storage,
            module=self,
        )

        if bias:
            self.bias = Tensor.zeros(output_features, requires_grad=True)
        else:
            self.bias = None

        self.compute_dtype = compute_dtype
        self.quant_state: Optional[QuantState] = None
        self.quant_storage = quant_storage

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with 4-bit quantized weights"""
        if self.compute_dtype is None:
            self.compute_dtype = x.dtype
        if x.dtype != self.compute_dtype:
            x = x.cast(self.compute_dtype)

        bias = None
        if self.bias is not None:
            bias = self.bias.cast(self.compute_dtype) if self.bias.dtype != self.compute_dtype else self.bias

        if self.weight.is_quantized and self.weight.quant_state is not None:
            dequantized_weight = dequantize_4bit(
                self.weight._tensor,
                self.weight.quant_state,
                blocksize=self.weight.blocksize,
                quant_type=self.weight.quant_type
            )
            output = x.linear(dequantized_weight.T, bias)
        else:
            # Use the underlying tensor for linear operation
            output = x.linear(self.weight._tensor.T, bias)

        return output

    def to(self, device=None):
        """Move layer to device, triggering quantization if needed"""
        # Move weight (this will trigger quantization)
        self.weight = self.weight.to(device=device)
        if self.bias is not None:
            self.bias = self.bias.to(device)

        return self

class LinearFP4(SuperSonicLinear4bit):
    """
    Implements the FP4 data type.
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_storage=dtypes.uint8,
        device=None,
    ):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            "fp4",
            quant_storage,
            device,
        )


class LinearNF4(SuperSonicLinear4bit):
    """Implements the NF4 data type.

    Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
    is normalized into the range [-1, 1].

    For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

    Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
    the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
    """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_storage=dtypes.uint8,
        device=None,
    ):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            "nf4",
            quant_storage,
            device,
        )

"""
to-do for whenever I need to do this lol
NO INT8 SUPPORT!!

class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad=True,
        has_fp16_weights=False,
        CB: Optional[torch.Tensor] = None,
        SCB: Optional[torch.Tensor] = None,
    ):
        if data is None:
            data = torch.empty(0)
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)
        obj.CB = CB
        obj.SCB = SCB
        obj.has_fp16_weights = has_fp16_weights
        return obj

    def _quantize(self, device):
        if self.has_fp16_weights:
            return super().to(device)

        # We quantize the weight and store in 8bit row-major
        B = self.data.contiguous().to(device=device, dtype=torch.float16)
        CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B)
        self.data = CB
        self.CB = CB
        self.SCB = SCB

        return self

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device: Optional[Union[int, device, str]] = None, non_blocking: bool = False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def xpu(self, device: Optional[Union[int, device, str]] = None, non_blocking: bool = False):
        return self.to(device="xpu" if device is None else device, non_blocking=non_blocking)

    def __deepcopy__(self, memo):
        # adjust this if new arguments are added to the constructor
        new_instance = type(self).__new__(
            type(self),
            data=copy.deepcopy(self.data, memo),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
            CB=copy.deepcopy(self.CB, memo),
            SCB=copy.deepcopy(self.SCB, memo),
        )
        return new_instance

    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[dtype, str]] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None and device.type != "meta" and self.data.device.type == "cpu":
            if device.type != "cpu" or self.data.dtype != torch.int8:
                return self._quantize(device)
            elif self.data.dtype == torch.int8 and device.type in ("cpu", "xpu") and (ipex_cpu or ipex_xpu):
                self.CB = self.data

        new_param = Int8Params(
            super().to(device=device, dtype=dtype, non_blocking=non_blocking),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
        )
        new_param.CB = self.CB
        new_param.SCB = self.SCB

        return new_param
"""
def replace_linear_with_supersonic(model, config):
    """Replace nn.Linear layers with SuperSonicLinear4bit"""
    pass
