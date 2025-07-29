from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW,Adam,LAMB, SGD
from tinygrad.helpers import trange
from transformers import PreTrainedModel

class SuperSonicTrainer:
    def __init__(self, model, args, data_args=None, config=None):
        self.model = model
        self.data_args= data_args
        self.args = args
        self.config = config

        if config is None:
            assert isinstance(self.model, PreTrainedModel), (
                "If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is"
                f" {self.model.__class__}"
            )
            self.config = self.model.config
        else:
            self.config = config

        self.data_args = data_args

    def create_optimizer_and_scheduler(self, num_train_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        """
        if self.optimizer is None:
            no_decay = "no decay"
