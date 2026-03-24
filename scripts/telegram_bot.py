#!/usr/bin/env python3
"""
Telegram bot interface for the fine-tuned Qwen2.5 mental health counselor model.

Supports skill-based routing: each user message is classified by the SkillRouter
and the corresponding LoRA adapter + system prompt is activated before generation.

Usage:
    TELEGRAM_BOT_TOKEN="xxx" python3 scripts/telegram_bot.py \
        --model_path models/qwen2.5-7b-mental-health-fullft-a100 \
        --adapters_dir adapters
"""

import os
import sys
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
import asyncio
import logging
import tempfile

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from faster_whisper import WhisperModel

# Add project root for mental_health_llm imports
_SCRIPT_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mental_health_llm.skill_router import SkillRouter

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
whisper_model = None
skill_router: SkillRouter | None = None
loaded_skills: list[str] = []

# Per-user conversation history: {user_id: [(user_msg, bot_msg), ...]}
user_histories: dict[int, list[tuple[str, str]]] = {}

# Per-user voice language preference: {user_id: language_code or None}
user_languages: dict[int, str | None] = {}

# Server-wide default language (set via --whisper_language CLI arg)
default_whisper_language: str | None = None

LANGUAGE_OPTIONS: dict[str, str] = {
    "auto": "Auto-detect",
    "cantonese": "Cantonese (廣東話)",
    "mandarin": "Mandarin (普通話)",
    "english": "English",
}

# Map user-facing names to Whisper language codes
_LANGUAGE_TO_WHISPER: dict[str, str] = {
    "cantonese": "yue",
    "mandarin": "zh",
    "english": "en",
}


def load_model_and_tokenizer(model_path: str, adapters_dir: str | None = None) -> None:
    """Load the base model and all available skill LoRA adapters."""
    global model, tokenizer, loaded_skills

    print(f"Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    # Load skill adapters
    if adapters_dir:
        adapters_path = Path(adapters_dir)
        if not adapters_path.is_absolute():
            adapters_path = _PROJECT_ROOT / adapters_path
        _load_adapters(str(adapters_path))

    model.training = False
    if loaded_skills:
        print(f"Model loaded with {len(loaded_skills)} adapters: {', '.join(loaded_skills)}")
    else:
        print("Model loaded (no adapters).")


SKILL_NAMES = [
    "crisis-intervention",
    "general-support",
    "cbt-therapy",
    "empathetic-listening",
    "psychoeducation",
    "professional-counseling",
]


def _load_adapters(adapters_dir: str) -> None:
    """Load all available skill LoRA adapters from a directory."""
    global model, loaded_skills
    first_adapter = True

    for skill_name in SKILL_NAMES:
        adapter_path = os.path.join(adapters_dir, skill_name)
        has_adapter = (
            os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
            or os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
            or os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
        )
        if not has_adapter:
            print(f"  Adapter not found: {adapter_path} — skipping")
            continue

        try:
            if first_adapter:
                model = PeftModel.from_pretrained(
                    model, adapter_path, adapter_name=skill_name
                )
                first_adapter = False
            else:
                model.load_adapter(adapter_path, adapter_name=skill_name)
            loaded_skills.append(skill_name)
            print(f"  Loaded adapter: {skill_name}")
        except Exception as e:
            print(f"  Failed to load {skill_name}: {e}")


def generate_response(
    question: str,
    system_prompt: str,
    history: list[tuple[str, str]] | None = None,
    max_length: int = 1024,
) -> str:
    """Generate a counseling response using the given system prompt."""
    messages = [{"role": "system", "content": system_prompt}]

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
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean up — stop at common continuation patterns
    stop_patterns = [
        "\n\nQuestion:", "\n\nHuman:", "\n\nUser:",
        "[End]", "\n\nBased on", "\n\nThis response",
    ]
    for pattern in stop_patterns:
        if pattern in response:
            response = response.split(pattern)[0].strip()
            break

    if len(response) > 2000:
        sentences = response.split(". ")
        response = ". ".join(sentences[:10]) + "."

    return response


def route_and_generate(
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Route to the best skill, activate its adapter, and generate a response.

    Returns (response_text, skill_name).
    """
    skill_name = skill_router.route(question)

    # Activate adapter if available
    if loaded_skills:
        if skill_name in loaded_skills:
            model.set_adapter(skill_name)
        else:
            fallback = (
                "general-support"
                if "general-support" in loaded_skills
                else loaded_skills[0]
            )
            model.set_adapter(fallback)

    system_prompt = skill_router.get_system_prompt(skill_name)
    response = generate_response(question, system_prompt, history)
    return response, skill_name


def load_whisper_model(model_size: str = "base") -> None:
    """Load faster-whisper on CPU with int8 quantization."""
    global whisper_model
    print(f"Loading Whisper model: {model_size}")
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("Whisper model loaded and ready.")


def transcribe_audio(file_path: str, language: str | None = None) -> str:
    """Transcribe an audio file using faster-whisper. Synchronous — use via asyncio.to_thread."""
    segments, _info = whisper_model.transcribe(
        file_path, beam_size=5, language=language
    )
    return " ".join(segment.text.strip() for segment in segments)


# ── Telegram handlers ──────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text(
        "Hello! I'm a mental health counseling assistant. "
        "You can share what's on your mind and I'll do my best to help.\n\n"
        "You can send text or voice messages — voice messages will be "
        "transcribed and I'll respond to the transcript.\n\n"
        "Supported voice languages: English, Cantonese (廣東話), "
        "Mandarin (普通話).\n\n"
        "Commands:\n"
        "/language - Set voice transcription language\n"
        "/clear - Reset our conversation history\n\n"
        "Feel free to start whenever you're ready."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command."""
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("Conversation history cleared. Let's start fresh.")


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /language command — set voice transcription language."""
    user_id = update.effective_user.id
    args = context.args

    if args:
        choice = args[0].lower()
        if choice not in LANGUAGE_OPTIONS:
            valid = ", ".join(f"{k} ({v})" for k, v in LANGUAGE_OPTIONS.items())
            await update.message.reply_text(
                f"Unknown language: {choice}\n\nValid options: {valid}"
            )
            return
        lang = None if choice == "auto" else _LANGUAGE_TO_WHISPER[choice]
        user_languages[user_id] = lang
        label = LANGUAGE_OPTIONS[choice]
        await update.message.reply_text(f"Voice language set to: {label}")
        return

    # No argument — show current setting and options
    current = user_languages.get(user_id, default_whisper_language)
    current_label = LANGUAGE_OPTIONS.get(current or "auto", "Auto-detect")
    lines = [f"Current voice language: {current_label}\n", "Usage: /language <code>\n"]
    for code, label in LANGUAGE_OPTIONS.items():
        lines.append(f"  /language {code}  —  {label}")
    await update.message.reply_text("\n".join(lines))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages — route to skill and generate response."""
    user_id = update.effective_user.id
    user_text = update.message.text

    history = user_histories.get(user_id, [])

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    response, skill_name = await asyncio.to_thread(
        route_and_generate, user_text, history
    )
    logger.info("User %s routed to [%s]", user_id, skill_name)

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


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming voice messages — transcribe and generate a counselor response."""
    voice = update.message.voice
    if voice.duration > 120:
        await update.message.reply_text(
            "Sorry, I can only process voice messages up to 2 minutes long."
        )
        return

    user_id = update.effective_user.id
    tmp_path = None
    try:
        # Download voice OGG to a temp file
        voice_file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        await voice_file.download_to_drive(tmp_path)

        # Transcribe on CPU in a worker thread
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        lang = user_languages.get(user_id, default_whisper_language)
        transcript = await asyncio.to_thread(transcribe_audio, tmp_path, lang)

        if not transcript or not transcript.strip():
            await update.message.reply_text(
                "I couldn't make out any words in that voice message. "
                "Could you try again or type your message instead?"
            )
            return

        history = user_histories.get(user_id, [])

        # Send transcript immediately while generating LLM response concurrently
        async def generate_and_reply() -> str:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            resp, skill_name = await asyncio.to_thread(
                route_and_generate, transcript, history
            )
            logger.info("User %s (voice) routed to [%s]", user_id, skill_name)
            if not resp or not resp.strip():
                resp = (
                    "I'm not sure how to respond to that. "
                    "Could you tell me more about what's on your mind?"
                )
            await update.message.reply_text(resp)
            return resp

        transcript_msg = f'I heard:\n"{transcript}"'
        _, response = await asyncio.gather(
            update.message.reply_text(transcript_msg),
            generate_and_reply(),
        )

        history.append((transcript, response))
        if len(history) > HISTORY_LIMIT:
            history.pop(0)
        user_histories[user_id] = history

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Telegram bot for the fine-tuned Qwen2.5 mental health counselor"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned base model",
    )
    parser.add_argument(
        "--adapters_dir",
        type=str,
        default=None,
        help="Directory containing skill adapter subdirectories (default: None)",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper model size for voice transcription (default: base)",
    )
    parser.add_argument(
        "--whisper_language",
        type=str,
        default=None,
        help="Force whisper language code, e.g. 'yue' for Cantonese (default: auto-detect)",
    )
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set the TELEGRAM_BOT_TOKEN environment variable.")

    global skill_router
    skill_router = SkillRouter()
    print(f"Skill router loaded with {len(skill_router.list_skills())} skills")

    load_model_and_tokenizer(args.model_path, args.adapters_dir)
    load_whisper_model(args.whisper_model)

    global default_whisper_language
    default_whisper_language = args.whisper_language

    # Configure proxy if HTTPS_PROXY is set (e.g. CSE CUHK HPC cluster)
    builder = Application.builder().token(token)
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy_url:
        logger.info(f"Using proxy: {proxy_url}")
        builder = builder.proxy(proxy_url)
    app = builder.build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Bot started — polling for updates.")
    app.run_polling()


if __name__ == "__main__":
    main()
