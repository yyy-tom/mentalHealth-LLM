#!/usr/bin/env python3
"""
Telegram bot interface for the fine-tuned Qwen2.5 mental health counselor model.

Usage:
    TELEGRAM_BOT_TOKEN="xxx" python3 scripts/telegram_bot.py \
        --model_path models/qwen2.5-7b-mental-health-fullft-a100
"""

import os
from pathlib import Path

_LARGE_DISK_PATH = Path(os.environ.get("HF_LARGE_DISK_PATH", "/research/d7/fyp25/yyyu2"))


def _configure_large_disk_cache() -> None:
    """Relocate HuggingFace cache directories onto a large disk (if available)."""
    if not _LARGE_DISK_PATH.exists():
        return

    cache_base = _LARGE_DISK_PATH / ".cache" / "huggingface"
    tmp_dir = _LARGE_DISK_PATH / ".cache" / "tmp"
    env_dirs = {
        "HF_HOME": cache_base,
        "TRANSFORMERS_CACHE": cache_base / "transformers",
        "HF_DATASETS_CACHE": cache_base / "datasets",
        "HF_HUB_CACHE": cache_base / "hub",
        "XET_CACHE": cache_base / "xet",
        "TMPDIR": tmp_dir,
        "TMP": tmp_dir,
        "TEMP": tmp_dir,
    }

    for directory in env_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for env_var, directory in env_dirs.items():
        os.environ[env_var] = str(directory)

    print(f"Using large disk cache: {cache_base}")
    print(f"Temporary files directory: {tmp_dir}")


_configure_large_disk_cache()

import argparse
import logging

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

_TF_MAJOR = int(transformers.__version__.split(".")[0])
_DTYPE_KEY = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HISTORY_LIMIT = 6

# Global model/tokenizer — loaded once at startup
model = None
tokenizer = None

# Per-user conversation history: {user_id: [(user_msg, bot_msg), ...]}
user_histories: dict[int, list[tuple[str, str]]] = {}


def load_model_and_tokenizer(model_path: str, base_model: str) -> None:
    """Load the fine-tuned model with 4-bit quantization for 11GB GPUs."""
    global model, tokenizer

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_available = torch.cuda.is_available()
    model_dtype = torch.float16 if cuda_available else torch.float32

    # Check available VRAM — use 4-bit if less than 16GB
    use_4bit = False
    if cuda_available:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU VRAM: {vram_gb:.1f} GB")
        if vram_gb < 16:
            use_4bit = True
            print("Using 4-bit quantization (VRAM < 16GB)")

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": 0} if cuda_available else "cpu",
        "low_cpu_mem_usage": True,
        _DTYPE_KEY: model_dtype,
    }

    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    try:
        model = PeftModel.from_pretrained(model, model_path)
        print(f"Loaded LoRA weights from {model_path}")
    except Exception as e:
        print(f"No LoRA weights found at {model_path}, using base model. Error: {e}")

    # Note: model.eval() sets the model to inference mode
    model.training = False
    print("Model loaded and ready.")


def generate_response(
    question: str,
    history: list[tuple[str, str]] | None = None,
    max_length: int = 1024,
) -> str:
    """Generate a counseling response for the given question."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a compassionate and professional mental health counselor. "
                "Please provide helpful, empathetic, and evidence-based advice. "
                "Your responses should:\n"
                "1. Acknowledge the person's feelings\n"
                "2. Offer practical advice\n"
                "3. Suggest professional resources if appropriate\n"
                "4. Maintain a warm, non-judgmental tone"
            ),
        }
    ]

    if history:
        for user_turn, counselor_turn in history:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": counselor_turn})

    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_length, 1024),
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if len(response) > 2000:
        sentences = response.split(". ")
        response = ". ".join(sentences[:10]) + "."

    return response


# ── Telegram handlers ──────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text(
        "Hello! I'm a mental health counseling assistant. "
        "You can share what's on your mind and I'll do my best to help.\n\n"
        "Commands:\n"
        "/clear - Reset our conversation history\n\n"
        "Feel free to start whenever you're ready."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command."""
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("Conversation history cleared. Let's start fresh.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages — generate a counselor response."""
    user_id = update.effective_user.id
    user_text = update.message.text

    history = user_histories.get(user_id, [])

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    response = generate_response(user_text, history)

    if not response or not response.strip():
        response = (
            "I'm not sure how to respond to that. "
            "Could you tell me more about what's on your mind?"
        )

    history.append((user_text, response))
    if len(history) > HISTORY_LIMIT:
        history.pop(0)
    user_histories[user_id] = history

    await update.message.reply_text(response)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram bot for the fine-tuned Qwen2.5 mental health counselor"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (LoRA weights directory)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set the TELEGRAM_BOT_TOKEN environment variable.")

    load_model_and_tokenizer(args.model_path, args.base_model)

    # Configure proxy if HTTPS_PROXY is set (e.g. CSE CUHK HPC cluster)
    builder = Application.builder().token(token)
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy_url:
        logger.info(f"Using proxy: {proxy_url}")
        builder = builder.proxy(proxy_url)
    app = builder.build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started — polling for updates.")
    app.run_polling()


if __name__ == "__main__":
    main()
