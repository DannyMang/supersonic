from tinygrad import dtype
from tinygrad.device import device
from tinygrad.tensor  import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
from typing import cast, Optional, List
from utils import dropout
#import torch.nn as nn

class LoRALayer():
    def __init__(
        self,
        r:int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.training = True

        if lora_dropout > 0.:
            self.dropout_probability = lora_dropout
            self.lora_dropout = self._apply_dropout
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights


    def _apply_dropout(self, x:Tensor):
        return dropout(x,self.dropout_probability, training=self.training)

    def train(self, mode:bool=True):
        #Setter for setting the training mode attr for dropout
        self.training = mode

class Embedding(LoRALayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.glorot_uniform(num_embeddings, embedding_dim)

        if r > 0:
            self.lora_A = Tensor.zeros(r, num_embeddings, requires_grad=True)
            self.lora_B = Tensor.zeros(embedding_dim, r, requires_grad=True)
            self.scaling = self.lora_alpha / self.r

            # Freeze base weights, we will not train these
            self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            self.lora_A = Tensor.zeros(*self.lora_A.shape, requires_grad=True)
            self.lora_B = Tensor.kaiming_uniform(*self.lora_B.shape, requires_grad=True)

    def _base_embedding(self, x: Tensor) -> Tensor:
            """Equivalent to nn.Embedding.forward(self, x)"""
            return self.weight[x]

    def train(self, mode:bool=True):
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight -= (self.lora_B.matmul(self.lora_A)).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight += (self.lora_B.matmul(self.lora_A)).transpose(0, 1) * self.scaling
                self.merged = True

    def __call__(self, x:Tensor) -> Tensor:
        #Forward pass
        if self.r > 0 and not self.merged:
            base_result = self.weight[x]
            after_A = self.lora_A.T[x]
            lora_result = after_A.matmul(self.lora_B.T) * self.scaling

            return cast(Tensor, base_result + lora_result)
        else:
            return self.weight[x]

class Linear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        self.linear = nn.Linear(in_features,out_features,bias=True)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out

         # Initialize weight and bias
        bound = 1 / (in_features ** 0.5)
        self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        self.bias = Tensor.uniform(out_features, low=-bound, high=bound)

        #Trainable params
        if r > 0:
            # For basic Linear: only 1 projection, so:
            # num_lora_projections = 1 (instead of sum(enable_lora))
            # projection_size = out_features (instead of out_features // len(enable_lora))

            # LoRA A: (r, in_features)
            self.lora_A = Tensor.zeros(
                r,                    # Just r, not r * num_projections
                in_features,
                requires_grad=True
            )

            # LoRA B: (out_features, r)
            self.lora_B = Tensor.zeros(
                out_features,         # Just out_features
                r,
                requires_grad=True
            )

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight = self.weight.T

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            self.lora_A = Tensor.kaiming_uniform(*self.lora_A.shape, requires_grad=True)
            self.lora_B = Tensor.zeros(*self.lora_B.shape, requires_grad=True)


    def train(self, mode:bool=True):
        def T(w: Tensor):
            return w.T if self.fan_in_fan_out else w
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    #to ensure weights are not MergedLinear
                    self.weight -= (self.lora_B.matmul(self.lora_A)).T * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight += (self.lora_B.matmul(self.lora_A)).T * self.scaling
                self.merged = True

    def forward(self, x: Tensor) -> Tensor:
            def T(w: Tensor) -> Tensor:
                return cast(Tensor, w.transpose(0, 1) if self.fan_in_fan_out else w)

            # Cast self.weight and self.bias to Tensor to help type checker
            weight = cast(Tensor, self.weight)
            bias = cast(Tensor, self.bias)

            if self.r > 0 and not self.merged:
                # Base linear transformation
                result = x.linear(cast(Tensor, T(weight).transpose()), bias=bias)

                # LoRA path: x -> A^T -> B^T, using dot() to avoid type issues
                lora_x = cast(Tensor, self.lora_dropout(x))
                lora_A = cast(Tensor, self.lora_A)
                lora_B = cast(Tensor, self.lora_B)

                # Chain the multiplications: x @ A^T @ B^T
                after_A = cast(Tensor, lora_x.dot(lora_A.T))
                lora_result = cast(Tensor, after_A.dot(lora_B.T) * self.scaling)

                return cast(Tensor, result + lora_result)
            else:
                return x.linear(cast(Tensor, T(weight).transpose()), bias=bias)



class MergedLinear(LoRALayer):
    #LoRA implemented in a base LoRALayer
    def __init__(
        self,
        in_features:int,
        out_features:int,
        r:int=0,
        lora_alpha:int=1,
        lora_dropout:float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights:bool = True,
        **kwargs
    ):
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        merge_weights=merge_weights)
        self.linear = nn.Linear(in_features,out_features,bias=True)
        assert out_features % len(enable_lora) == 0, \
                   'The length of enable_lora must divide out_features'
        self.in_features = in_features
        self.out_features = out_features
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.weight = Tensor.kaiming_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features)

        #Trainable params
        if r > 0 and any(enable_lora):
            num_lora_projections = sum(enable_lora)
            projection_size = out_features // len(enable_lora)
            #LoRA A: (r * num_lora_projections, in_features)
            self.lora_A = Tensor.zeros(
                r * num_lora_projections,
                in_features,
                requires_grad=True
            )

            # LoRA B: (projection_size * num_lora_projections, r)
            self.lora_B = Tensor.zeros(
                projection_size * num_lora_projections,
                r,
                requires_grad=True
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self._create_lora_indices()

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight = self.weight.T

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            self.lora_A = Tensor.kaiming_uniform(*self.lora_A.shape, requires_grad=True)
            self.lora_B = Tensor.zeros(*self.lora_B.shape, requires_grad=True)

    def _create_lora_indices(self):
        #TO-DO review this
        proj_size = self.out_features // len(self.enable_lora)
        indices = Tensor.zeros(len(self.enable_lora), proj_size, dtype='bool')
        for i, enabled in enumerate(self.enable_lora):
            if enabled:
                indices[i] = True

        return indices.reshape(-1)

    def zero_pad(self, x: Tensor):
        if not hasattr(self.lora_ind, 'shape'):
            self.lora_ind = Tensor(self.lora_ind)

        result_shape = (self.lora_ind.shape[0], *x.shape[1:])
        result = Tensor.zeros(*result_shape, dtype=x.dtype, device=x.device)
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w: Tensor):
            return w.T if self.fan_in_fan_out else w
        delta_w = self.lora_A.unsqueeze(0).conv2d(
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode:bool=True):
        def T(w: Tensor):
            return w.T if self.fan_in_fan_out else w
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0 and any(self.enable_lora):
                    #to ensure weights are not MergedLinear
                    self.weight -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: Tensor) -> Tensor:
        def T(w: Tensor) -> Tensor:
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return x.linear(T(self.weight).transpose(), self.bias)
        else:
            result = x.linear(T(self.weight).transpose(), self.bias)
            if self.r > 0:
                lora_input = cast(Tensor, self.lora_dropout(x))
                lora_result = lora_input.matmul(T(self.merge_AB().T)) * self.scaling
                result += lora_result
            return result
