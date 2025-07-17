from tinygrad.tensor  import  Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
#from tinygrad import nn
#import torch.nn as nn
#
from utils import dropout



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

        if lora_dropout > 0.:
            self.dropout_probability = lora_dropout
            self.lora_dropout = self._apply_dropout
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights


    def _apply_dropout(self, x:Tensor):
        return dropout(x,self.dropout_probability, training=True)

class Embedding(nn.Embedding, LoRALayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)

        if r > 0:
            self.lora_A = Tensor.zeros(r, num_embeddings)
            self.lora_B = Tensor.zeros(embedding_dim, r)
            self.scaling = self.lora_alpha / self.r

            # "Freeze" base weights
            self.weight.requires_grad = False

            # LoRA weights are trainable
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            self.lora_A = Tensor.zeros(*self.lora_A.shape)
            self.lora_B = Tensor.kaiming_uniform(*self.lora_B.shape)
