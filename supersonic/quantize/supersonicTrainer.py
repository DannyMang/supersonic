from tinygrad.tensor import Tensor
from tinygrad.nn.optim import AdamW, Adam, SGD
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import trange
from extra.lr_scheduler import MultiStepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from examples.mlperf.lr_schedulers import CosineAnnealingLRWithWarmup, PolynomialDecayWithWarmup
import numpy as np
from typing import Optional, Dict, Any, Union
import os
import json
from tqdm import tqdm
from dataclasses import dataclass, field
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingArguments:
    # Core training arguments
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before an optimizer step.'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take.'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm.'})

    # Scheduler and warmup
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule.'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for.'})

    # Logging and saving
    logging_steps: int = field(default=10, metadata={"help": 'Frequency of update steps to log the loss.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints.'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model.'})
    save_total_limit: int = field(default=40, metadata={"help": 'Max number of checkpoints to save.'})

    # Evaluation
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation on the dev set."})
    eval_steps: int = field(default=250, metadata={"help": "How often to run evaluation."})
    evaluation_strategy: str = field(default="steps", metadata={"help": "The evaluation strategy during training."})

    # Training control
    do_train: bool = field(default=True, metadata={"help": 'Whether to run training.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing.'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length.'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Remove unused columns from the dataset.'})

    # QLoRA specific arguments
    full_finetune: bool = field(default=False, metadata={"help": "Finetune the entire model without adapters."})
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(default=True, metadata={"help": "Compress quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. ('fp4' or 'nf4')."})
    bits: int = field(default=4, metadata={"help": "How many bits to use for quantization."})

    # LoRA arguments
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})

    # Hardware and optimization
    optim: str = field(default='adamw', metadata={"help": 'The optimizer to be used.'})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})

    # Data processing
    train_on_source: Optional[bool] = field(default=False, metadata={"help": "Whether to train on the input in addition to the target text."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to cache downloaded models and datasets."})

    # MMLU evaluation (if needed)
    do_mmlu_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to run the MMLU evaluation."})
    mmlu_split: Optional[str] = field(default='eval', metadata={"help": "The MMLU split to run on."})
    mmlu_dataset: Optional[str] = field(default='mmlu-fs', metadata={"help": "MMLU dataset to use."})
    max_mmlu_samples: Optional[int] = field(default=None, metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMLU dataset."})
    mmlu_source_max_len: int = field(default=2048, metadata={"help": "Maximum source sequence length for MMLU."})

    # Reporting
    report_to: str = field(default='none', metadata={"help": "The list of integrations to report results to."})


"""
original for reference

# @dataclass
# class TrainingArguments(transformers.Seq2SeqTrainingArguments):
#     cache_dir: Optional[str] = field(
#         default=None
#     )
#     train_on_source: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Whether to train on the input in addition to the target text."}
#     )
#     mmlu_split: Optional[str] = field(
#         default='eval',
#         metadata={"help": "The MMLU split to run on"}
#     )
#     mmlu_dataset: Optional[str] = field(
#         default='mmlu-fs',
#         metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
#     )
#     do_mmlu_eval: Optional[bool] = field(
#         default=False,
#         metadata={"help": "Whether to run the MMLU evaluation."}
#     )
#     max_mmlu_samples: Optional[int] = field(
#         default=None,
#         metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
#     )
#     mmlu_source_max_len: int = field(
#         default=2048,
#         metadata={"help": "Maximum source sequence length for mmlu."}
#     )
#     full_finetune: bool = field(
#         default=False,
#         metadata={"help": "Finetune the entire model without adapters."}
#     )
#     adam8bit: bool = field(
#         default=False,
#         metadata={"help": "Use 8-bit adam."}
#     )
#     double_quant: bool = field(
#         default=True,
#         metadata={"help": "Compress the quantization statistics through double quantization."}
#     )
#     quant_type: str = field(
#         default="nf4",
#         metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
#     )
#     bits: int = field(
#         default=4,
#         metadata={"help": "How many bits to use."}
#     )
#     lora_r: int = field(
#         default=64,
#         metadata={"help": "Lora R dimension."}
#     )
#     lora_alpha: float = field(
#         default=16,
#         metadata={"help": " Lora alpha."}
#     )
#     lora_dropout: float = field(
#         default=0.0,
#         metadata={"help":"Lora dropout."}
#     )
#     max_memory_MB: int = field(
#         default=80000,
#         metadata={"help": "Free memory per gpu."}
#     )
#     report_to: str = field(
#         default='none',
#         metadata={"help": "To use wandb or something else for reporting."}
#     )
#     output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
#     optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
#     per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
#     gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
#     max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
#     weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
#     learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
#     remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
#     max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
#     gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
#     do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
#     lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
#     warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
#     logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
#     group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
#     save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
#     save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
#     save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

"""

class SuperSonicTrainer:
    def __init__(self, model, tokenizer, args, train_dataset=None, eval_dataset=None, data_collator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.log_history = []
        self.state = {
            'epoch': 0,
            'global_step': 0,
            'max_steps': args.max_steps,
            'log_history': [],
            'best_metric': None,
            'best_model_checkpoint': None
        }

        self.create_optimizer_and_scheduler()
        self.setup_dataloaders()

    def create_optimizer_and_scheduler(self):
        """Setup the optimizer and the learning rate scheduler."""
        trainable_params = [p for p in get_parameters(self.model) if p.requires_grad]

        print(f"Found {len(trainable_params)} trainable parameters")
        if not trainable_params:
            raise ValueError("No trainable parameters found! Check QLoRA setup.")

        if self.args.optim.lower() in ("adamw", "paged_adamw_32bit"):
            self.optimizer = AdamW(trainable_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay, eps=1e-8)
        elif self.args.optim.lower() == "adam":
            self.optimizer = Adam(trainable_params, lr=self.args.learning_rate, eps=1e-8)
        elif self.args.optim.lower() == "sgd":
            self.optimizer = SGD(trainable_params, lr=self.args.learning_rate, momentum=0.9,
                                           weight_decay=self.args.weight_decay, nesterov=True)
        else:
            print(f"Optimizer '{self.args.optim}' not recognized. Defaulting to AdamW.")
            self.optimizer = AdamW(trainable_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        num_training_steps = self.args.max_steps
        warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        if self.args.lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps, eta_min=0)
        elif self.args.lr_scheduler_type == "cosine_with_warmup":
            self.scheduler = CosineAnnealingLRWithWarmup(self.optimizer, base_lr=self.args.learning_rate, end_lr=1e-7, warmup_steps=warmup_steps, decay_steps=num_training_steps - warmup_steps)
        elif self.args.lr_scheduler_type == "polynomial":
            self.scheduler = PolynomialDecayWithWarmup(self.optimizer, initial_lr=self.args.learning_rate, end_lr=1e-7, train_steps=num_training_steps, warmup=warmup_steps)
        elif self.args.lr_scheduler_type == "linear":
            self.scheduler = PolynomialDecayWithWarmup(self.optimizer, initial_lr=self.args.learning_rate, end_lr=1e-7, train_steps=num_training_steps, warmup=warmup_steps, power=1)
        else:
            self.scheduler = None

    def setup_dataloaders(self):
        """Setup data loaders with proper batching and collation."""
        if self.train_dataset is not None:
            self.train_dataloader = self.create_dataloader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
        if self.eval_dataset is not None:
            self.eval_dataloader = self.create_dataloader(self.eval_dataset, batch_size=self.args.per_device_eval_batch_size if hasattr(self.args, 'per_device_eval_batch_size') else self.args.per_device_train_batch_size, shuffle=False)

    def create_dataloader(self, dataset, batch_size, shuffle=False):
        """Create a simple dataloader that yields batches."""
        if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
            return dataset
        def dataloader_generator():
            indices = list(range(len(dataset)))
            if shuffle: np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_items = [dataset[idx] for idx in batch_indices]
                if self.data_collator:
                    yield self.data_collator(batch_items)
                else:
                    yield self.default_collate(batch_items)
        return dataloader_generator()

    def default_collate(self, batch):
        """Default collation function."""
        if not batch: return {}
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], Tensor):
                collated[key] = Tensor.stack(*values)
            else:
                # Handle padding for lists of numbers (like input_ids)
                if isinstance(values[0], (list, np.ndarray)) and all(isinstance(i, (int, np.integer)) for i in values[0]):
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0
                    max_len = max(len(v) for v in values)
                    padded_values = [v + [pad_token_id] * (max_len - len(v)) for v in values]
                    collated[key] = Tensor(padded_values)
                else:
                    collated[key] = values
        return collated

    def compute_loss(self, batch):
        """Compute loss for a batch."""
        if hasattr(self.model, 'forward'):
            outputs = self.model(**batch)
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch.get('labels')
            )
        if hasattr(outputs, 'loss'):
            return outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        else:
            raise ValueError("Could not extract loss from model outputs")

    def train_step(self, batch):
        """Performs a single training step with gradient accumulation support."""
        assert self.optimizer is not None
        accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        if self.global_step % accumulation_steps == 0:
            self.optimizer.zero_grad()

        loss = self.compute_loss(batch)
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
        loss.backward()

        if (self.global_step + 1) % accumulation_steps == 0:
            if self.args.max_grad_norm > 0:
                self.clip_gradients()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        return loss.item() * accumulation_steps


    def clip_gradients(self):
        """Clip gradients to prevent exploding gradients."""
        assert self.optimizer is not None
        grad_norms = [p.grad.square().sum() for p in self.optimizer.params if p.grad is not None]
        if grad_norms:
            total_norm = Tensor.stack(*grad_norms).sum().sqrt()
            clip_factor = self.args.max_grad_norm / (total_norm + 1e-6)

            if clip_factor < 1:
                for param in self.optimizer.params:
                    if param.grad is not None:
                        param.grad.assign(param.grad * clip_factor).realize()

    def save_checkpoint(self, checkpoint_dir):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        if hasattr(self.model, 'save_pretrained'): self.model.save_pretrained(checkpoint_dir)
        state_path = os.path.join(checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"Checkpoint saved to {checkpoint_dir}")

    def log_metrics(self, metrics, step=None):
        """Log training metrics."""
        step = step if step is not None else self.global_step
        metrics['step'] = step
        metrics['epoch'] = self.epoch
        self.log_history.append(metrics)
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        print(f"Step {step} | {metric_str}")

    def train(self):
        """Main training loop."""
        print(f"Starting training... Total steps: {self.args.max_steps}")
        if hasattr(self.model, 'train'): self.model.train()

        pbar = trange(self.args.max_steps, initial=self.global_step)
        while self.global_step < self.args.max_steps:
            for batch in self.train_dataloader:
                if self.global_step >= self.args.max_steps: break
                step_loss = self.train_step(batch)
                self.global_step += 1

                if self.global_step % self.args.logging_steps == 0:
                    metrics = {'loss': step_loss, 'lr': self.get_current_lr()}
                    self.log_metrics(metrics)
                    pbar.set_description(f"Loss: {step_loss:.4f}")

                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}"))
                pbar.update(1)

            self.epoch += 1

        print("Training completed!")
        self.save_checkpoint(os.path.join(self.args.output_dir, "final_checkpoint"))

    def get_current_lr(self):
        """Get current learning rate."""
        assert self.optimizer is not None
        if hasattr(self.optimizer.lr, 'numpy'):
            return float(self.optimizer.lr.numpy()[0])
        elif hasattr(self.optimizer.lr, 'item'):
            return float(self.optimizer.lr.item())
        else:
            return float(self.optimizer.lr.numpy())

    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataset is None: return {}
        print("Running evaluation...")

        with Tensor.train(False):
            total_loss = 0.0
            num_steps = 0
            for batch in self.eval_dataloader:
                try:
                    loss = self.compute_loss(batch)
                    total_loss += loss.item()
                    num_steps += 1
                except Exception as e:
                    print(f"Error in evaluation step: {e}")

        if num_steps > 0:
            avg_loss = total_loss / num_steps
            print(f"Evaluation Loss: {avg_loss:.4f}")
            return {"eval_loss": avg_loss}
        return {}

    def predict(self, test_dataset):
        """Prediction loop."""
        print("Running prediction...")
        test_dataloader = self.create_dataloader(test_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=False)
        predictions = []

        with Tensor.train(False):
            for batch in test_dataloader:
                input_batch = {k: v for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**input_batch)
                preds = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions.append(preds.numpy())
        return {"predictions": np.concatenate(predictions, axis=0)}
