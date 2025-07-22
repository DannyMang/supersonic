
from tinygrad.tensor import Tensor
from tinygrad.device import device

def get_4bit_type(typename:str. device=None, blocksize = 64):
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


    # data = torch.tensor(data, device=device)
    # data.div_(data.abs().max())

    # assert data.numel() == 16


    return data




def quantize_4bit():
    pass


def dequantize_4bit():
    pass
