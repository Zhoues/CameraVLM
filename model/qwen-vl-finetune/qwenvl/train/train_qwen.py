# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import json
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
import qwenvl.data.data_processor as data_processor_module
from qwenvl.active import (
    add_active_tokens_to_tokenizer,
    configure_activeqwen_config,
    get_active_special_tokens,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.model import ActiveQwen3VLForConditionalGeneration
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def is_rank0():
    return local_rank in (None, -1, 0)


def rank0_print(*args):
    if is_rank0():
        print(*args)


def rank0_print_json(title, payload):
    rank0_print(f"=== {title} (Rank0) ===")
    rank0_print(json.dumps(payload, ensure_ascii=False, indent=2))


def summarize_parameters(module):
    if module is None:
        return {
            "total_params": 0,
            "trainable_params": 0,
            "total_bytes": 0,
            "trainable_bytes": 0,
            "trainable_ratio": 0.0,
        }

    total_params = 0
    trainable_params = 0
    total_bytes = 0
    trainable_bytes = 0

    for param in module.parameters():
        param_count = getattr(param, "ds_numel", None)
        if param_count is None:
            param_count = param.numel()
        param_bytes = param_count * param.element_size()
        total_params += param_count
        total_bytes += param_bytes
        if param.requires_grad:
            trainable_params += param_count
            trainable_bytes += param_bytes

    trainable_ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_bytes": total_bytes,
        "trainable_bytes": trainable_bytes,
        "trainable_ratio": trainable_ratio,
    }


def format_param_count(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def format_size_bytes(value):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0


def print_module_trainability(module_name, module):
    summary = summarize_parameters(module)
    if summary["trainable_params"] == 0:
        status = "frozen"
    elif summary["trainable_params"] == summary["total_params"]:
        status = "trainable"
    else:
        status = "partially_trainable"

    rank0_print(
        f"[Module] {module_name}: status={status}, "
        f"trainable={format_param_count(summary['trainable_params'])}/"
        f"{format_param_count(summary['total_params'])} "
        f"({summary['trainable_ratio']:.2f}%), "
        f"trainable_size={format_size_bytes(summary['trainable_bytes'])}"
    )


def print_dataset_summary(train_dataset):
    dataset_statistics = getattr(train_dataset, "dataset_statistics", [])
    if not dataset_statistics:
        rank0_print("=== Dataset Summary (Rank0) ===")
        rank0_print("No dataset statistics available.")
        return

    rank0_print_json(
        "Dataset Summary",
        {
            "datasets": [
                {
                    "dataset_name": stat["dataset_name"],
                    "annotation_path": stat["annotation_path"],
                    "sampling_rate": stat["sampling_rate"],
                    "provide_latent": stat["provide_latent"],
                    "record_count": stat["record_count"],
                    "qa_count": stat["qa_count"],
                }
                for stat in dataset_statistics
            ],
            "total_qa_count": getattr(train_dataset, "total_qa_count", "N/A"),
            "total_record_count": getattr(train_dataset, "total_record_count", "N/A"),
        },
    )


def print_token_summary(tokenizer, model_args, num_active_tokens_added):
    if not model_args.activeqwen_enable:
        rank0_print("=== Special Tokens (Rank0) ===")
        rank0_print("ActiveQwen special tokens disabled for this run.")
        return

    active_tokens = get_active_special_tokens(model_args.activeqwen_latent_token_count)
    rank0_print_json(
        "Special Tokens",
        {
            "count": len(active_tokens),
            "newly_added": num_active_tokens_added,
            "tokens": [
                {"token": token, "id": tokenizer.convert_tokens_to_ids(token)}
                for token in active_tokens
            ],
        },
    )


def print_training_summary(model, tokenizer, model_args, data_args, training_args, train_dataset, num_active_tokens_added):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )
    summary = summarize_parameters(model)
    training_summary = {
        "model_path": model_args.model_name_or_path,
        "model_class": model.__class__.__name__,
        "datasets": data_args.dataset_use,
        "data_flatten": data_args.data_flatten,
        "data_packing": data_args.data_packing,
        "batch": {
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "world_size": world_size,
            "effective_global_batch_size": global_batch_size,
        },
        "learning_rates": {
            "base": training_args.learning_rate,
            "vision_tower": training_args.vision_tower_lr if training_args.vision_tower_lr is not None else training_args.learning_rate,
            "mm_projector": training_args.mm_projector_lr if training_args.mm_projector_lr is not None else training_args.learning_rate,
            "active_projector": training_args.active_projector_lr if training_args.active_projector_lr is not None else training_args.learning_rate,
        },
        "loss_weights": {
            "text_ce": training_args.active_ce_loss_weight,
            "active_3d": training_args.active_3d_loss_weight,
        },
        "parameters": {
            "total": {
                "count": summary["total_params"],
                "formatted": format_param_count(summary["total_params"]),
                "size": format_size_bytes(summary["total_bytes"]),
            },
            "trainable": {
                "count": summary["trainable_params"],
                "formatted": format_param_count(summary["trainable_params"]),
                "size": format_size_bytes(summary["trainable_bytes"]),
                "ratio_percent": round(summary["trainable_ratio"], 2),
            },
        },
    }
    if model_args.activeqwen_enable:
        training_summary["activeqwen"] = {
            "tune_active_projector": model_args.tune_active_projector,
            "latent_token_count": model_args.activeqwen_latent_token_count,
            "projector_prompt_length": model_args.activeqwen_projector_prompt_length,
            "projector_depth": model_args.activeqwen_projector_depth,
            "target_dim": model_args.activeqwen_target_dim,
        }
    rank0_print_json("Training Summary", training_summary)

    print_dataset_summary(train_dataset)
    print_token_summary(tokenizer, model_args, num_active_tokens_added)

    rank0_print("=== Module Trainability (Rank0) ===")
    print_module_trainability("visual", getattr(model, "visual", None))
    visual_module = getattr(model, "visual", None)
    print_module_trainability(
        "visual.merger",
        getattr(visual_module, "merger", None) if visual_module is not None else None,
    )
    print_module_trainability("language_model", getattr(model, "language_model", None))
    print_module_trainability("lm_head", getattr(model, "lm_head", None))
    print_module_trainability("active_projector", getattr(model, "active_projector", None))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def prepare_activeqwen(model, processor, tokenizer, model_args, training_args):
    num_added_tokens = add_active_tokens_to_tokenizer(
        tokenizer,
        model_args.activeqwen_latent_token_count,
    )
    processor.tokenizer = tokenizer
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    model.config.activeqwen_projector_prompt_length = (
        model_args.activeqwen_projector_prompt_length
    )
    model.config.activeqwen_target_dim = model_args.activeqwen_target_dim
    model.config.activeqwen_projector_depth = model_args.activeqwen_projector_depth
    model.config.active_ce_loss_weight = training_args.active_ce_loss_weight
    model.config.active_3d_loss_weight = training_args.active_3d_loss_weight
    configure_activeqwen_config(
        model.config,
        tokenizer,
        model_args.activeqwen_latent_token_count,
    )
    if hasattr(model, "reset_active_projector"):
        model.reset_active_projector()
    return num_added_tokens


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

    if hasattr(model, "active_projector"):
        for n, p in model.active_projector.named_parameters():
            p.requires_grad = model_args.activeqwen_enable and model_args.tune_active_projector


def train(attn_implementation="flash_attention_2"):
    global local_rank
    startup_debug = os.environ.get("ACTIVEQWEN_VERBOSE_STARTUP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    data_processor_module.local_rank = local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        if model_args.activeqwen_enable:
            raise ValueError("ActiveQwen currently supports Qwen3-VL dense checkpoints only.")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model_cls = (
            ActiveQwen3VLForConditionalGeneration
            if model_args.activeqwen_enable
            else Qwen3VLForConditionalGeneration
        )
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    rank0_print(
        f"Initialized model from {model_args.model_name_or_path} with class {model.__class__.__name__}"
    )

    num_active_tokens_added = 0
    if model_args.activeqwen_enable:
        num_active_tokens_added = prepare_activeqwen(model, processor, tokenizer, model_args, training_args)
        data_args.activeqwen_enable = True
        data_args.activeqwen_latent_token_count = model_args.activeqwen_latent_token_count
        data_args.activeqwen_latent_token_ids = model.config.activeqwen_latent_token_ids
    else:
        data_args.activeqwen_enable = False
        data_args.activeqwen_latent_token_count = 0
        data_args.activeqwen_latent_token_ids = []

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if (
            startup_debug
            and (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            )
        ):
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    print_training_summary(
        model=model,
        tokenizer=tokenizer,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        train_dataset=data_module["train_dataset"],
        num_active_tokens_added=num_active_tokens_added,
    )
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
