#!/usr/bin/env python3
"""
Train a single skill-specific LoRA adapter on top of the frozen full-FT base model.

Adapted from train_qwen_counsel_multi_gpu.py. Key differences:
  - --skill argument selects which skill to train
  - Base model is the full-FT checkpoint (not original Qwen HF)
  - Saves ONLY adapter weights to adapters/{skill_name}/
  - Supports single-GPU and multi-GPU (DDP via torchrun)

Usage:
    # Single GPU
    python scripts/train_skill_lora.py --skill psychoeducation

    # With config file
    python scripts/train_skill_lora.py --skill cbt-therapy --config configs/skills/cbt-therapy.json

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=2 scripts/train_skill_lora.py --skill crisis-intervention
"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_from_disk
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# transformers v5+ renamed torch_dtype -> dtype
_TF_MAJOR = int(transformers.__version__.split(".")[0])
_DTYPE_KEY = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

SKILLS = [
    "crisis-intervention",
    "general-support",
    "cbt-therapy",
    "empathetic-listening",
    "psychoeducation",
    "professional-counseling",
]

DEFAULT_BASE_MODEL = "models/qwen2.5-7b-mental-health-fullft-a100"


# ---------------------------------------------------------------------------
# Callbacks (reused from train_qwen_counsel_multi_gpu.py)
# ---------------------------------------------------------------------------
class MemoryManagementCallback(TrainerCallback):
    """Callback to manage GPU memory during training and checkpointing."""

    def on_save(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared CUDA cache before checkpoint save")

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class ProgressCallback(TrainerCallback):
    """Log training progress at regular intervals."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % args.logging_steps == 0 and logs:
            loss = logs.get("loss", logs.get("train_loss", "?"))
            lr = logs.get("learning_rate", "?")
            logger.info(f"Step {state.global_step}: loss={loss}, lr={lr}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SkillLoRATrainer:
    """Train a single LoRA adapter for a specific counseling skill."""

    def __init__(self, config: Dict[str, Any], skill_name: str):
        self.config = config
        self.skill_name = skill_name
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup_logging_for_rank(self):
        """Configure logging based on DDP rank."""
        import warnings
        warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*")

        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            logging.getLogger().setLevel(logging.INFO)
            logging.getLogger("transformers").setLevel(logging.INFO)
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            logging.getLogger("datasets").setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.WARNING)
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)

    def setup_tokenizer(self):
        """Load tokenizer from the base model."""
        model_name = self.config["model_name"]
        logger.info(f"Loading tokenizer from {model_name}")

        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def setup_model(self):
        """Load base model (frozen) and apply fresh LoRA adapter."""
        model_name = self.config["model_name"]
        logger.info(f"Loading base model from {model_name}")

        # Resolve relative path
        if not os.path.isabs(model_name) and not model_name.startswith("Qwen/"):
            resolved = PROJECT_ROOT / model_name
            if resolved.exists():
                model_name = str(resolved)

        # Clean memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        cuda_available = torch.cuda.is_available()
        use_4bit = self.config.get("use_4bit", False)

        # BF16 / FP16 auto-detection
        if cuda_available:
            # Use device 0 for capability check — all GPUs on the node are
            # the same type, and LOCAL_RANK may exceed device_count when
            # SLURM allocates fewer GPUs than torchrun processes.
            capability = torch.cuda.get_device_capability(0)
            bf16_supported = capability[0] >= 8
            if self.config.get("bf16", True) and not bf16_supported:
                logger.warning("bf16 not supported on this GPU, falling back to fp16")
                self.config["bf16"] = False
                self.config["fp16"] = True

        # Quantization config (optional)
        bnb_config = None
        # Use fp16 compute dtype for GPUs without native bf16 (e.g. RTX 2080 Ti sm_75)
        bnb_compute_dtype = torch.bfloat16 if self.config.get("bf16", True) else torch.float16
        model_dtype = bnb_compute_dtype

        if use_4bit and cuda_available:
            try:
                import bitsandbytes as _bnb  # noqa: F401 — verify installed
            except ImportError:
                raise ImportError(
                    "4-bit quantization requires bitsandbytes: "
                    "pip install -U bitsandbytes>=0.46.1"
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )
            logger.info(f"Using 4-bit quantization (compute_dtype={bnb_compute_dtype})")

        cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)

        # Load base model
        if bnb_config:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            # Serialize loading across DDP ranks to avoid simultaneous OOM
            for loading_rank in range(world_size):
                if local_rank == loading_rank:
                    logger.info(f"Rank {local_rank}: loading model...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        device_map={"": device},
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir,
                        max_memory={local_rank: "10GiB"},
                        **{_DTYPE_KEY: model_dtype},
                    )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

            self.model = prepare_model_for_kbit_training(self.model)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            # Disable use_cache for generation_config too
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.use_cache = False
        else:
            # Non-quantized loading (requires >=24GB VRAM per GPU)
            if cuda_available:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
                world_size = int(os.environ.get("WORLD_SIZE", 1))

                for loading_rank in range(world_size):
                    if local_rank == loading_rank:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map={"": device},
                            cache_dir=cache_dir,
                            **{_DTYPE_KEY: model_dtype},
                        )
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    **{_DTYPE_KEY: torch.float32},
                )

        # Apply LoRA adapter
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_r", 32),
            lora_alpha=self.config.get("lora_alpha", 64),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=self.config.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Gradient checkpointing
        if self.config.get("gradient_checkpointing", True):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        self.model.train()
        logger.info(f"Model loaded with LoRA for skill: {self.skill_name}")

    def load_dataset(self):
        """Load the skill-specific dataset."""
        dataset_path = self.config["dataset_path"]
        if not os.path.isabs(dataset_path):
            dataset_path = str(PROJECT_ROOT / dataset_path)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        dataset = load_from_disk(dataset_path)
        logger.info(f"Dataset loaded: train={len(dataset['train'])}, val={len(dataset.get('validation', []))}")
        return dataset

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format a training example using the Qwen chat template."""
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a batch of examples."""
        batch_size = len(examples["instruction"])
        prompts = []
        for i in range(batch_size):
            example = {
                "instruction": examples["instruction"][i],
                "output": examples["output"][i],
            }
            prompts.append(self.format_prompt(example))

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=self.config.get("max_length", 768),
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def train(self):
        """Run the full training pipeline."""
        self.setup_logging_for_rank()

        # Setup HF cache
        base_dir = os.environ.get("HF_BASE_DIR", str(PROJECT_ROOT))
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = str(Path(base_dir) / ".cache" / "huggingface")
        if "TRANSFORMERS_CACHE" not in os.environ:
            os.environ["TRANSFORMERS_CACHE"] = str(Path(os.environ["HF_HOME"]) / "hub")
        for d in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"]]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Check for checkpoint to resume
        resume_from_checkpoint = None
        output_dir = self.config["output_dir"]
        if not os.path.isabs(output_dir):
            output_dir = str(PROJECT_ROOT / output_dir)
        self.config["output_dir"] = output_dir

        if os.path.exists(output_dir):
            checkpoints = [
                d for d in os.listdir(output_dir)
                if d.startswith("checkpoint-") and not d.endswith(".backup")
            ]
            if checkpoints:
                steps = []
                for cp in checkpoints:
                    try:
                        steps.append(int(cp.split("-")[1]))
                    except (ValueError, IndexError):
                        continue
                if steps:
                    resume_from_checkpoint = os.path.join(output_dir, f"checkpoint-{max(steps)}")
                    logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

        # Guard: 4-bit + multi-GPU requires torchrun (not DataParallel)
        if (
            self.config.get("use_4bit", False)
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and int(os.environ.get("WORLD_SIZE", 1)) == 1
        ):
            raise RuntimeError(
                "Detected multiple GPUs but WORLD_SIZE=1. "
                "4-bit models are incompatible with DataParallel. "
                "Launch via torchrun: torchrun --nproc_per_node=N scripts/train_skill_lora.py ..."
            )

        # Setup components
        self.setup_tokenizer()
        self.setup_model()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load and tokenize dataset
        dataset = self.load_dataset()

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc=f"Tokenize {self.skill_name}",
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("eval_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            logging_steps=self.config.get("logging_steps", 50),
            eval_steps=self.config.get("eval_steps", 500),
            save_steps=self.config.get("save_steps", 1000),
            eval_strategy=self.config.get("eval_strategy", "steps"),
            save_strategy=self.config.get("save_strategy", "steps"),
            save_total_limit=self.config.get("save_total_limit", 2),
            load_best_model_at_end=self.config.get("load_best_model_at_end", True),
            metric_for_best_model=self.config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=False,
            fp16=self.config.get("fp16", False),
            bf16=self.config.get("bf16", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            remove_unused_columns=False,
            optim=self.config.get("optim", "adamw_torch_fused"),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            report_to="wandb" if self.config.get("use_wandb", False) else "none",
            run_name=self.config.get("run_name", f"skill-lora-{self.skill_name}"),
            seed=self.config.get("seed", 42),
            ddp_find_unused_parameters=False,
            ddp_broadcast_buffers=False,
            ddp_bucket_cap_mb=5,
            dataloader_pin_memory=True,
            dataloader_num_workers=self.config.get("dataloader_num_workers", 0),
        )

        # Callbacks
        callbacks = [MemoryManagementCallback(), ProgressCallback()]
        if self.config.get("early_stopping", False):
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            data_collator=data_collator,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )

        # Train
        rank = int(os.environ.get("RANK", 0))
        logger.info(f"Starting LoRA training for skill: {self.skill_name}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if rank == 0:
                try:
                    emergency_path = os.path.join(output_dir, "emergency_checkpoint")
                    trainer.save_model(emergency_path)
                    logger.info(f"Emergency checkpoint saved to {emergency_path}")
                except Exception:
                    pass
            raise

        # Save final adapter
        if rank == 0:
            logger.info(f"Saving adapter to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Save training config alongside adapter
            config_save_path = os.path.join(output_dir, "training_config.json")
            with open(config_save_path, "w") as f:
                json.dump(self.config, f, indent=2)

        logger.info(f"Training completed for skill: {self.skill_name}")


def main():
    parser = argparse.ArgumentParser(description="Train a skill-specific LoRA adapter")
    parser.add_argument(
        "--skill",
        type=str,
        required=True,
        choices=SKILLS,
        help="Skill to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to skill config JSON (default: configs/skills/{skill}.json)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override dataset path (default: datasets/skills/{skill})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: adapters/{skill})",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Enable 4-bit quantization (required for GPUs with <24GB VRAM)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override max sequence length (default: 768, use 512 for 11GB GPUs)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="Override LoRA rank (default: 32, use 16 for 11GB GPUs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size (default: 4, use 1 for 11GB GPUs)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps (default: 8, use 32 for batch_size=1)",
    )
    args = parser.parse_args()

    # Build default config
    config = {
        "model_name": args.base_model,
        "dataset_path": args.dataset_path or f"datasets/skills/{args.skill}",
        "output_dir": args.output_dir or f"adapters/{args.skill}",
        "use_4bit": False,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "max_length": 768,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "eval_steps": 500,
        "save_steps": 1000,
        "save_total_limit": 2,
        "seed": 42,
        "use_wandb": False,
        "run_name": f"skill-lora-{args.skill}",
    }

    # Load config file if specified or if default exists
    config_path = args.config
    if config_path is None:
        default_config = f"configs/skills/{args.skill}.json"
        if os.path.exists(default_config):
            config_path = default_config
        else:
            # Try relative to project root
            alt = str(PROJECT_ROOT / default_config)
            if os.path.exists(alt):
                config_path = alt

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            file_config = json.load(f)
        config.update(file_config)

    # CLI overrides
    if args.base_model != DEFAULT_BASE_MODEL:
        config["model_name"] = args.base_model
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.use_4bit:
        config["use_4bit"] = True
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.lora_r is not None:
        config["lora_r"] = args.lora_r
        config["lora_alpha"] = args.lora_r * 2  # keep alpha = 2 * r
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["gradient_accumulation_steps"] = args.grad_accum

    # Ensure output dir exists
    out = config["output_dir"]
    if not os.path.isabs(out):
        out = str(PROJECT_ROOT / out)
        config["output_dir"] = out
    Path(out).mkdir(parents=True, exist_ok=True)

    # Train
    trainer = SkillLoRATrainer(config, args.skill)
    trainer.train()


if __name__ == "__main__":
    main()
