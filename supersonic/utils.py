from tinygrad.tensor import Tensor

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
