#!/usr/bin/env python3
"""
Full fine-tune training script for Qwen2.5 models.
Supports single-GPU and multi-GPU (DDP via torchrun).
Designed for high-VRAM GPUs (H100 80GB, A100 80GB) — NO LoRA, NO quantization.

All 7.6B parameters are trained in bf16 with AdamW + gradient checkpointing.

Usage (single GPU):
    python scripts/train_qwen_fullft.py --config configs/qwen_7b_fullft_a100_1gpu_fast.json

Usage (multi-GPU with DDP):
    torchrun --nproc_per_node=8 scripts/train_qwen_fullft.py --config configs/qwen_7b_fullft_h100_8gpu.json
"""

import os
import json
import logging
import argparse
import gc
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import load_from_disk, Dataset

# Optional: wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class MemoryManagementCallback(TrainerCallback):
    """Clear CUDA cache before saves and evals to reduce OOM risk."""

    def on_save(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class ProgressCallback(TrainerCallback):
    """Log GPU memory periodically during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and state.global_step % 100 == 0:
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"Step {state.global_step}: GPU mem {alloc:.1f} GB allocated, "
                f"{reserved:.1f} GB reserved"
            )


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class FullFineTuneTrainer:
    """Full fine-tune trainer for Qwen2.5 models (single or multi-GPU)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main_process = self.local_rank == 0

    # ---- tokenizer --------------------------------------------------------

    def setup_tokenizer(self) -> None:
        model_name = self.config["model_name"]
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        logger.info(f"Loading tokenizer: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")

    # ---- model ------------------------------------------------------------

    def setup_model(self) -> None:
        model_name = self.config["model_name"]
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        use_bf16 = self.config.get("bf16", True)
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        logger.info(f"Loading model: {model_name} (full params, dtype={dtype})")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )

        # Move to the correct device (respects LOCAL_RANK for DDP)
        device = torch.device(f"cuda:{self.local_rank}")
        self.model = self.model.to(device)

        # Gradient checkpointing
        if self.config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.config.use_cache = False
            logger.info("Gradient checkpointing enabled")

        self.model.train()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
            f"({100 * trainable_params / total_params:.1f}%)"
        )

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                f"GPU memory after model load: {alloc:.1f} GB allocated, "
                f"{reserved:.1f} GB reserved, {total_mem:.1f} GB total"
            )

    # ---- data -------------------------------------------------------------

    def load_dataset(self) -> Dataset:
        dataset_path = self.config["dataset_path"]
        if not os.path.isabs(dataset_path):
            project_root = Path(__file__).parent.parent
            dataset_path = str(project_root / dataset_path)

        logger.info(f"Loading dataset: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        logger.info(
            f"Dataset: {len(dataset['train']):,} train, "
            f"{len(dataset['validation']):,} validation"
        )
        return dataset

    def format_prompt(self, example: Dict[str, Any]) -> str:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = len(examples["instruction"])
        prompts = []
        for i in range(batch_size):
            ex = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i],
            }
            prompts.append(self.format_prompt(ex))

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=self.config.get("max_length", 1024),
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # ---- training args ----------------------------------------------------

    def setup_training_arguments(self) -> TrainingArguments:
        cfg = self.config

        # Build kwargs, including optional DDP settings
        kwargs = dict(
            output_dir=cfg["output_dir"],
            num_train_epochs=cfg.get("num_epochs", 3),
            per_device_train_batch_size=cfg.get("batch_size", 4),
            per_device_eval_batch_size=cfg.get("eval_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            learning_rate=cfg.get("learning_rate", 2e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            logging_steps=cfg.get("logging_steps", 25),
            eval_steps=cfg.get("eval_steps", 500),
            save_steps=cfg.get("save_steps", 1000),
            eval_strategy=cfg.get("eval_strategy", "steps"),
            save_strategy=cfg.get("save_strategy", "steps"),
            save_total_limit=cfg.get("save_total_limit", 3),
            load_best_model_at_end=cfg.get("load_best_model_at_end", True),
            metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=False,
            fp16=cfg.get("fp16", False),
            bf16=cfg.get("bf16", True),
            dataloader_pin_memory=cfg.get("dataloader_pin_memory", True),
            dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            remove_unused_columns=False,
            optim=cfg.get("optim", "adamw_torch_fused"),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            report_to="wandb" if cfg.get("use_wandb", False) and HAS_WANDB else "none",
            seed=cfg.get("seed", 42),
        )

        # Optional max_steps to cap training time
        if "max_steps" in cfg:
            kwargs["max_steps"] = cfg["max_steps"]

        # DDP-specific options (applied when running with torchrun)
        if "ddp_find_unused_parameters" in cfg:
            kwargs["ddp_find_unused_parameters"] = cfg["ddp_find_unused_parameters"]
        if "ddp_broadcast_buffers" in cfg:
            kwargs["ddp_broadcast_buffers"] = cfg["ddp_broadcast_buffers"]
        if "ddp_bucket_cap_mb" in cfg:
            kwargs["ddp_bucket_cap_mb"] = cfg["ddp_bucket_cap_mb"]

        return TrainingArguments(**kwargs)

    # ---- trainer ----------------------------------------------------------

    def setup_trainer(self, dataset: Dataset) -> Trainer:
        dataset_path = self.config["dataset_path"]
        if not os.path.isabs(dataset_path):
            project_root = Path(__file__).parent.parent
            dataset_path = str(project_root / dataset_path)

        tokenized_path = dataset_path + "_tokenized"

        # Try pre-tokenized first
        tokenized_dataset = None
        if os.path.exists(tokenized_path):
            try:
                logger.info(f"Loading pre-tokenized dataset: {tokenized_path}")
                tokenized_dataset = load_from_disk(tokenized_path)
                logger.info("Pre-tokenized dataset loaded.")
            except Exception as e:
                logger.warning(f"Failed to load pre-tokenized dataset: {e}")

        if tokenized_dataset is None:
            logger.info("Tokenizing dataset (this may take a few minutes)...")
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=True,
                num_proc=4,
            )
            try:
                tokenized_dataset.save_to_disk(tokenized_path)
                logger.info(f"Saved tokenized dataset to {tokenized_path}")
            except Exception as e:
                logger.warning(f"Could not save tokenized dataset: {e}")

        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding=True, return_tensors="pt"
        )
        training_args = self.setup_training_arguments()

        callbacks = [MemoryManagementCallback(), ProgressCallback()]
        if self.config.get("early_stopping", True):
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=collator,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
        return trainer

    # ---- main entry -------------------------------------------------------

    def run(self) -> None:
        if self.is_main_process:
            logger.info("=" * 60)
            logger.info(f"Qwen 7B Full Fine-Tune — {self.world_size} GPU(s)")
            logger.info("=" * 60)

        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — full fine-tune requires a GPU.")
        gpu = torch.cuda.get_device_properties(self.local_rank)
        logger.info(f"[Rank {self.local_rank}] GPU: {gpu.name}, {gpu.total_memory / 1024**3:.0f} GB VRAM")

        # Setup
        self.setup_tokenizer()
        self.setup_model()
        dataset = self.load_dataset()
        self.trainer = self.setup_trainer(dataset)

        # Check for checkpoints to resume from
        resume_from = None
        output_dir = self.config["output_dir"]
        if os.path.exists(output_dir):
            checkpoints = sorted(
                [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0,
            )
            if checkpoints:
                resume_from = os.path.join(output_dir, checkpoints[-1])
                logger.info(f"Resuming from checkpoint: {resume_from}")

        # Train
        steps_per_epoch = len(self.trainer.get_train_dataloader())
        total_steps = steps_per_epoch * self.config.get("num_epochs", 3)
        logger.info(f"Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"Total training steps: {total_steps:,}")

        try:
            self.trainer.train(resume_from_checkpoint=resume_from)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            try:
                emergency = os.path.join(output_dir, "emergency_checkpoint")
                self.trainer.save_model(emergency)
                logger.info(f"Emergency checkpoint saved: {emergency}")
            except Exception:
                pass
            raise

        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        logger.info("Training completed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Full fine-tune Qwen2.5 on a single high-VRAM GPU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_7b_fullft_h100.json",
        help="Path to config JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Save config copy
    with open(os.path.join(config["output_dir"], "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    trainer = FullFineTuneTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
