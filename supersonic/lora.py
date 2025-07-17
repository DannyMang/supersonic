from tinygrad.tensor  import  Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
from typing import cast
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
        return self

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

        return self

    def __call__(self, x:Tensor) -> Tensor:
        #Forward pass
        if self.r > 0 and not self.merged:
            base_result = self.weight[x]
            after_A = self.lora_A.T[x]
            lora_result = after_A.matmul(self.lora_B.T) * self.scaling

            return cast(Tensor, base_result + lora_result)
        else:
            return self.weight[x]
