"""
- Combine your existing LoRA with NF4 quantization
- 4-bit weight storage + BFloat16 compute
- Memory-efficient gradient computation
- Integration with your existing Linear layers


https://arxiv.org/pdf/2305.14314
"""
import os
import argparse
import json
import sys
import logging
import pandas as pd
import copy
import json
from os.path import exists, join, isdir
from tinygrad.tensor import Tensor
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
import numpy as np
from .lora import Linear as LoRALinear
from .quantization import quantize_4bit, dequantize_4bit, QuantState
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .supersonicTrainer import SuperSonicTrainer, TrainingArguments
from tinygrad.nn.state import torch_load, safe_load, load_state_dict
import torch  # For type annotations and pad_sequence
from torch.nn.utils.rnn import pad_sequence

# Constants
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)



class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def create_qlora_config(
    quant_type: str = 'nf4',
    compute_dtype: str = 'float16',
    blocksize: int = 64,
    double_quant: bool = True
) -> Dict:
    """
    Create configuration for SuperSonicLinear4bit layer

    Args:
        quant_type: Quantization type - 'nf4' (recommended for QLoRA) or 'fp4'
        compute_dtype: Computation dtype - 'float16', 'bfloat16', or 'float32'
        blocksize: Block size for quantization (64, 128, 256, etc.)
        double_quant: Whether to use double quantization for statistics compression

    Returns:
        Configuration dictionary for SuperSonicLinear4bit
    """
    if quant_type not in ['nf4', 'fp4']:
        raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type}")
    if compute_dtype not in ['float16', 'bfloat16', 'float32']:
        raise ValueError(f"compute_dtype must be 'float16', 'bfloat16', or 'float32', got {compute_dtype}")
    valid_blocksizes = [64, 128, 256, 512, 1024, 2048, 4096]
    if blocksize not in valid_blocksizes:
        raise ValueError(f"blocksize must be one of {valid_blocksizes}, got {blocksize}")

    return {
        'quant_type': quant_type,
        'compute_dtype': compute_dtype,
        'blocksize': blocksize,
        'double_quant': double_quant
    }

class SuperSonicLinear4bit(LoRALinear):
    """
    The core QLoRA layer - inherits LoRA capabilities + adds quantization

    This combines 4-bit quantized weights with LoRA adapters for memory-efficient training.
    Key features:
    - 4-bit weight storage (NF4/FP4)
    - BFloat16 compute for LoRA adapters
    - Memory-efficient gradient computation
    """

    def __init__(self, in_features, out_features, config, lora_config):
        super().__init__(in_features, out_features, **lora_config)

        # Validate and set quantization parameters
        self.quant_config = config
        self.compute_dtype = config.get('compute_dtype', 'float16')
        self.quant_type = config.get('quant_type', 'nf4')  # Default to NF4 as per QLoRA paper
        self.blocksize = config.get('blocksize', 64)
        self.double_quant = config.get('double_quant', True)  # Compress statistics

        # Validate quantization type
        if self.quant_type not in ['nf4', 'fp4']:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}. Must be 'nf4' or 'fp4'")

        # Validate compute dtype
        if self.compute_dtype not in ['float16', 'bfloat16', 'float32']:
            raise ValueError(f"Unsupported compute_dtype: {self.compute_dtype}")

        # Print configuration for debugging
        print(f"SuperSonicLinear4bit initialized with:")
        print(f"  - Quantization: {self.quant_type}")
        print(f"  - Compute dtype: {self.compute_dtype}")
        print(f"  - Block size: {self.blocksize}")
        print(f"  - Double quantization: {self.double_quant}")

        if hasattr(self, 'weight'):
            self._quantize_weight()

    def _quantize_weight(self):
        """Quantize the base weight to 4-bit and store quantization state"""
        from .quantization import quantize_4bit

        # Quantize the weight tensor
        quantized_weight, quant_state = quantize_4bit(
            self.weight,
            blocksize=self.blocksize,
            quant_type=self.quant_type
        )

        self.quantized_weight = quantized_weight
        self.quant_state = quant_state

        # Remove original weight to save memory
        self.weight.requires_grad = False
        del self.weight

    def _dequantize_weight(self):
        """Dequantize weight for computation"""
        from .quantization import dequantize_4bit

        return dequantize_4bit(
            self.quantized_weight,
            self.quant_state
        )

    def forward(self, x):
        """
        Forward pass for QLoRA layer:
        1. Dequantize base weights to compute dtype (BF16/FP16)
        2. Compute base linear transformation
        3. Compute LoRA adaptation (A @ B)
        4. Combine base + LoRA results
        """
        # Convert input to compute dtype if needed
        if self.compute_dtype == 'bfloat16':
            x = x.cast('bfloat16')
        elif self.compute_dtype == 'float16':
            x = x.cast('float16')

        # Base computation with dequantized weights
        dequantized_weight = self._dequantize_weight()
        if self.compute_dtype == 'bfloat16':
            dequantized_weight = dequantized_weight.cast('bfloat16')
        elif self.compute_dtype == 'float16':
            dequantized_weight = dequantized_weight.cast('float16')

        # Base linear transformation: x @ W.T
        base_result = x @ dequantized_weight.T
        if self.bias is not None:
            base_result = base_result + self.bias

        # LoRA computation (if r > 0 and not merged)
        if self.r > 0 and not self.merged:
            # Apply dropout to input
            lora_input = self.lora_dropout(x)

            # LoRA forward: input @ A.T @ B.T * scaling
            lora_result = (
                lora_input @ self.lora_A.T @ self.lora_B.T
            ) * self.scaling

            result = base_result + lora_result
        else:
            result = base_result

        return result

def get_accelerate_model(args, checkpoint_dir):
    # TO-DO :function will  INIT MODEL ARCHITECTURE
    #
    # 1) Hardware:
    #device = setup_tinygrad_devices(args)
    #
    # 2) model
    #model = create_tinygrad_model(args.model_name_or_path)

    # or Load pretrained weights from HuggingFace/PyTorch
    if args.model_name_or_path.endswith('.safetensors'):
        weights = safe_load(args.model_name_or_path)
    else:
        model_path = download_hf_model(args.model_name_or_path)
        weights = torch_load(model_path)

    #TO-DO Handle HuggingFace -> tinygrad key mapping if needed TO_DO
   # weights = convert_hf_to_tinygrad_keys(weights)

    # Load weights into model
    load_state_dict(model, weights, strict=False)

    """

    just some references on how some helpful functions that can implemnet this part

    basically get_accelerate_model will
    1.setup hardware/ detect GPUs/ config device mapping for multi-gpu or perhaps multi-training
    2. model loading such as AutoModelForCausalLM.from_pretrained()
    3. model configurations // configuring compute dtype based on hardware
    4. tokenizer setup
    5. qLoRA prep => prepare_model_for_kbit_training()




    from tinygrad.nn.state import torch_load, load_state_dict

    # Load PyTorch weights directly
    weights = torch_load("model.pth")  # or from HuggingFace hub
    load_state_dict(model, weights)
        ________________________________________________________
    from tinygrad.nn.state import safe_load, load_state_dict

    # Load safetensors (preferred format)
    weights = safe_load("model.safetensors")
    load_state_dict(model, weights)

    ________________________________________________________


    weights = load("model.gguf")  # Handles GGUF automatically
    load_state_dict(model, weights)

    """

    # TO_DO Apply quantization and LoRA
    #model = prepare_model_for_qlora(model, args)

    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(Tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        Tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(Tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(Tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    # To _DO REPLACE THIS WITH TINYGRAD TENSORS INSTEAD OF PYTORCH
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print('loaded model')
    #set_seed(args.seed) TO-DO

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    """
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    """
     # Create training arguments for SuperSonicTrainer with ALL original fields
    training_args = TrainingArguments(
        # Core training
        output_dir=getattr(args, 'output_dir', './output'),
        learning_rate=getattr(args, 'learning_rate', 0.0002),
        per_device_train_batch_size=getattr(args, 'per_device_train_batch_size', 1),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 16),
        max_steps=getattr(args, 'max_steps', 10000),
        weight_decay=getattr(args, 'weight_decay', 0.0),
        max_grad_norm=getattr(args, 'max_grad_norm', 0.3),

        # Scheduler
        lr_scheduler_type=getattr(args, 'lr_scheduler_type', 'constant'),
        warmup_ratio=getattr(args, 'warmup_ratio', 0.03),

        # Logging and saving
        logging_steps=getattr(args, 'logging_steps', 10),
        save_strategy=getattr(args, 'save_strategy', 'steps'),
        save_steps=getattr(args, 'save_steps', 250),
        save_total_limit=getattr(args, 'save_total_limit', 40),

        # Evaluation
        do_eval=getattr(args, 'do_eval', True),
        eval_steps=getattr(args, 'eval_steps', 250),
        evaluation_strategy='steps',

        # Training control
        do_train=getattr(args, 'do_train', True),
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', True),
        group_by_length=getattr(args, 'group_by_length', True),
        remove_unused_columns=getattr(args, 'remove_unused_columns', False),

        # QLoRA specific
        full_finetune=getattr(args, 'full_finetune', False),
        adam8bit=getattr(args, 'adam8bit', False),
        double_quant=getattr(args, 'double_quant', True),
        quant_type=getattr(args, 'quant_type', 'nf4'),
        bits=getattr(args, 'bits', 4),

        # LoRA
        lora_r=getattr(args, 'lora_r', 64),
        lora_alpha=getattr(args, 'lora_alpha', 16),
        lora_dropout=getattr(args, 'lora_dropout', 0.0),

        # Hardware and optimization
        optim=getattr(args, 'optim', 'adamw'),  # Simplified from paged_adamw_32bit
        max_memory_MB=getattr(args, 'max_memory_MB', 80000),

        # Data processing
        train_on_source=getattr(args, 'train_on_source', False),
        cache_dir=getattr(args, 'cache_dir', None),

        # MMLU evaluation
        do_mmlu_eval=getattr(args, 'do_mmlu_eval', False),
        mmlu_split=getattr(args, 'mmlu_split', 'eval'),
        mmlu_dataset=getattr(args, 'mmlu_dataset', 'mmlu-fs'),
        max_mmlu_samples=getattr(args, 'max_mmlu_samples', None),
        mmlu_source_max_len=getattr(args, 'mmlu_source_max_len', 2048),

        # Reporting
        report_to=getattr(args, 'report_to', 'none'),
    )
    trainer = SuperSonicTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module['train_dataset'],
        eval_dataset=data_module['eval_dataset'],
        data_collator=data_module['data_collator'],
        tokenizer=tokenizer,
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        pass
        """
            to-do if needed
            You'll need to implement MMLUEvalCallback for SuperSonicTrainer
            if args.mmlu_dataset == 'mmlu-zs':
                mmlu_dataset = load_dataset("json", data_files={
                    'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                    'test': 'data/mmlu/zero_shot_mmlu_test.json',
                })
                mmlu_dataset = mmlu_dataset.remove_columns('subject')
            # MMLU Five-shot (Eval/Test only)
            elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
                mmlu_dataset = load_dataset("json", data_files={
                    'eval': 'data/mmlu/five_shot_mmlu_val.json',
                    'test': 'data/mmlu/five_shot_mmlu_test.json',
                })
                # mmlu_dataset = mmlu_dataset.remove_columns('subject')
            mmlu_dataset = mmlu_dataset[args.mmlu_split]
            if args.max_mmlu_samples is not None:
                mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
            abcd_idx = [
                tokenizer("A", add_special_tokens=False).input_ids[0],
                tokenizer("B", add_special_tokens=False).input_ids[0],
                tokenizer("C", add_special_tokens=False).input_ids[0],
                tokenizer("D", add_special_tokens=False).input_ids[0],
            ]
            accuracy = evaluate.load("accuracy")
            class MMLUEvalCallback(transformers.TrainerCallback):
                def on_evaluate(self, args, state, control, model, **kwargs):
                    data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                    source_max_len = trainer.data_collator.source_max_len
                    trainer.data_collator.source_max_len = args.mmlu_source_max_len
                    trainer.model.eval()
                    preds, refs = [], []
                    loss_mmlu = 0
                    for batch in tqdm(data_loader, total=len(data_loader)):
                        (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                        # There are two tokens, the output, and eos token.
                        for i, logit in enumerate(logits):
                            label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                            logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                            preds.append(torch.argmax(logit_abcd).item())
                        labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                        refs += [abcd_idx.index(label) for label in labels.tolist()]
                        loss_mmlu += loss.item()
                    # Extract results by subject.
                    results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                    subject = mmlu_dataset['subject']
                    subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                    for s,p,r in zip(subject, preds, refs):
                        subjects[s]['preds'].append(p)
                        subjects[s]['refs'].append(r)
                    subject_scores = []
                    for subject in subjects:
                        subject_score = accuracy.compute(
                            references=subjects[subject]['refs'],
                            predictions=subjects[subject]['preds']
                        )['accuracy']
                        results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                        subject_scores.append(subject_score)
                    results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                    trainer.log(results)
                    trainer.data_collator.source_max_len = source_max_len

            trainer.add_callback(MMLUEvalCallback)
            """
    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)


    all_metrics = {"run_name": args.run_name}
    # Training
    if training_args.do_train:
        print("*** Train ***")
        train_result = trainer.train()
        if hasattr(train_result, 'metrics'):
            all_metrics.update(train_result.metrics)

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")
        eval_result = trainer.evaluate()
        if hasattr(eval_result, 'metrics'):
            all_metrics.update(eval_result.metrics)

    # Save metrics
    if (training_args.do_train or training_args.do_eval):
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
