#!/usr/bin/env python3
"""
Telegram bot for mental health counseling with multi-model support.

Supports 6 model variants (3 fine-tuned + 3 base) with per-user model
selection via /model command. Skill-based routing selects system prompts;
LoRA adapters are activated only for Qwen fine-tuned models.

Usage:
    TELEGRAM_BOT_TOKEN="xxx" python3 scripts/telegram_bot.py \
        --models qwen-ft gemma-ft mistral-ft \
        --adapters-dir adapters \
        --preload
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
import threading

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

sys.path.insert(0, str(_SCRIPT_DIR))  # for evaluation subpackage
from evaluation.judge_scoring import (
    init_judges,
    score_exchange_async,
    format_score_report,
)

_TF_MAJOR = int(transformers.__version__.split(".")[0])
_DTYPE_KEY = "dtype" if _TF_MAJOR >= 5 else "torch_dtype"
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
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

# ── Model Registry ────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "qwen-ft": {
        "name": "Qwen 2.5 7B (fine-tuned)",
        "path": "models/qwen2.5-7b-mental-health-fullft-a100",
    },
    "qwen-base": {
        "name": "Qwen 2.5 7B (base)",
        "path": "Qwen/Qwen2.5-7B-Instruct",
    },
    "gemma-ft": {
        "name": "Gemma 2 9B (fine-tuned)",
        "path": "models/gemma2-9b-mental-health-fullft-a100",
    },
    "gemma-base": {
        "name": "Gemma 2 9B (base)",
        "path": "google/gemma-2-9b-it",
    },
    "mistral-ft": {
        "name": "Mistral 7B (fine-tuned)",
        "path": "models/mistral-7b-mental-health-fullft-a100",
    },
    "mistral-base": {
        "name": "Mistral 7B (base)",
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
    },
}

SKILL_NAMES = [
    "crisis-intervention",
    "general-support",
    "cbt-therapy",
    "empathetic-listening",
    "psychoeducation",
    "professional-counseling",
]


# ── ModelManager ──────────────────────────────────────────────────


class ModelManager:
    """Manages multiple models across GPUs with preload or on-demand loading."""

    def __init__(
        self,
        model_keys: list[str],
        preload: bool = False,
        adapters_dir: str | None = None,
    ):
        self._models: dict[str, tuple] = {}  # key -> (model, tokenizer, loaded_skills)
        self._lock = threading.Lock()
        self._model_keys = model_keys
        self._adapters_dir = adapters_dir

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"ModelManager: {len(model_keys)} models requested, {gpu_count} GPUs available")

        if preload:
            for i, key in enumerate(model_keys):
                gpu_id = i % gpu_count if gpu_count > 0 else None
                self._load_model(key, gpu_id)

    def get(self, model_key: str) -> tuple:
        """Return (model, tokenizer, loaded_skills) for the given key. Load if needed."""
        if model_key in self._models:
            return self._models[model_key]

        with self._lock:
            # Double-check after acquiring lock
            if model_key in self._models:
                return self._models[model_key]

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

            # On-demand: if only 1 GPU, unload all others first
            if gpu_count <= 1 and self._models:
                for old_key in list(self._models.keys()):
                    print(f"Unloading {old_key} to free GPU memory...")
                    del self._models[old_key]
                torch.cuda.empty_cache()

            gpu_id = 0 if gpu_count > 0 else None
            self._load_model(model_key, gpu_id)
            return self._models[model_key]

    def _load_model(self, model_key: str, gpu_id: int | None) -> None:
        """Load one model onto a specific GPU in 4-bit."""
        info = MODEL_REGISTRY[model_key]
        model_path = info["path"]

        # Resolve relative paths against project root
        if not model_path.startswith("/") and "/" not in model_path.split("/")[0].split("."):
            candidate = _PROJECT_ROOT / model_path
            if candidate.exists():
                model_path = str(candidate)

        print(f"Loading {model_key} ({info['name']}) from {model_path}"
              + (f" -> GPU {gpu_id}" if gpu_id is not None else " -> CPU"))

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        cuda_available = gpu_id is not None
        model_dtype = torch.float16 if cuda_available else torch.float32

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": {"": gpu_id} if cuda_available else "cpu",
            "low_cpu_mem_usage": True,
            _DTYPE_KEY: model_dtype,
        }

        if cuda_available:
            vram_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            if vram_gb < 16:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=model_dtype,
                )
                print(f"  Using 4-bit quantization (GPU {gpu_id} VRAM: {vram_gb:.1f} GB)")

        mdl = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # Load LoRA adapters only for qwen-ft
        loaded_skills: list[str] = []
        if model_key == "qwen-ft" and self._adapters_dir:
            adapters_path = Path(self._adapters_dir)
            if not adapters_path.is_absolute():
                adapters_path = _PROJECT_ROOT / adapters_path
            mdl, loaded_skills = _load_adapters(mdl, str(adapters_path))

        mdl.training = False

        if loaded_skills:
            print(f"  {model_key} loaded with {len(loaded_skills)} adapters: {', '.join(loaded_skills)}")
        else:
            print(f"  {model_key} loaded (no adapters).")

        self._models[model_key] = (mdl, tok, loaded_skills)

    def list_available(self) -> list[str]:
        """Return the list of model keys this manager can serve."""
        return list(self._model_keys)

    def display_name(self, model_key: str) -> str:
        """Return human-readable name for a model key."""
        return MODEL_REGISTRY[model_key]["name"]

    def is_loaded(self, model_key: str) -> bool:
        """Check whether a model is already loaded in memory."""
        return model_key in self._models


def _load_adapters(mdl, adapters_dir: str) -> tuple:
    """Load all available skill LoRA adapters from a directory.

    Returns (model_with_adapters, list_of_loaded_skill_names).
    """
    loaded_skills: list[str] = []
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
                mdl = PeftModel.from_pretrained(
                    mdl, adapter_path, adapter_name=skill_name
                )
                first_adapter = False
            else:
                mdl.load_adapter(adapter_path, adapter_name=skill_name)
            loaded_skills.append(skill_name)
            print(f"  Loaded adapter: {skill_name}")
        except Exception as e:
            print(f"  Failed to load {skill_name}: {e}")

    return mdl, loaded_skills


# ── Global state ──────────────────────────────────────────────────

model_manager: ModelManager | None = None
whisper_model = None
skill_router: SkillRouter | None = None
default_model_key: str = "qwen-ft"

# Per-user state
user_histories: dict[int, list[tuple[str, str]]] = {}
user_models: dict[int, str] = {}
user_languages: dict[int, str | None] = {}
user_adapters_enabled: dict[int, bool] = {}  # True = adapters on (default)

default_whisper_language: str | None = None

# Per-user scoring state
user_scores: dict[int, list[dict]] = {}
user_pending_scores: dict[int, int] = {}
_judges_active = False

LANGUAGE_OPTIONS: dict[str, str] = {
    "auto": "Auto-detect",
    "cantonese": "Cantonese (廣東話)",
    "mandarin": "Mandarin (普通話)",
    "english": "English",
}

_LANGUAGE_TO_WHISPER: dict[str, str] = {
    "cantonese": "yue",
    "mandarin": "zh",
    "english": "en",
}


# ── Generation ────────────────────────────────────────────────────


def _supports_system_role(tok) -> bool:
    """Check whether the tokenizer's chat template supports a system role."""
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def generate_response(
    question: str,
    system_prompt: str,
    mdl,
    tok,
    history: list[tuple[str, str]] | None = None,
    max_length: int = 1024,
) -> str:
    """Generate a counseling response using the given system prompt and model."""
    use_system_role = _supports_system_role(tok)

    if use_system_role:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        # Models like Gemma that don't support system role:
        # prepend the system prompt into the first user message.
        messages = []

    if history:
        for i, (user_turn, counselor_turn) in enumerate(history):
            content = user_turn
            if not use_system_role and i == 0:
                content = f"{system_prompt}\n\n{user_turn}"
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": counselor_turn})

    user_content = question
    if not use_system_role and not history:
        user_content = f"{system_prompt}\n\n{question}"
    messages.append({"role": "user", "content": user_content})

    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(mdl.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=min(max_length, 1024),
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            no_repeat_ngram_size=3,
        )

    new_tokens = outputs[0][input_length:]
    response = tok.decode(new_tokens, skip_special_tokens=True).strip()

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
    model_key: str,
    history: list[tuple[str, str]] | None = None,
    adapters_enabled: bool = True,
) -> tuple[str, str]:
    """Route to the best skill, activate adapter if applicable, and generate.

    Returns (response_text, skill_name).
    """
    skill_name = skill_router.route(question, history=history)

    mdl, tok, loaded_skills = model_manager.get(model_key)

    # Activate or disable LoRA adapters based on user preference
    if loaded_skills:
        if adapters_enabled:
            mdl.enable_adapter_layers()
            if skill_name in loaded_skills:
                mdl.set_adapter(skill_name)
            else:
                fallback = (
                    "general-support"
                    if "general-support" in loaded_skills
                    else loaded_skills[0]
                )
                mdl.set_adapter(fallback)
        else:
            mdl.disable_adapter_layers()

    system_prompt = skill_router.get_system_prompt(skill_name)
    try:
        response = generate_response(question, system_prompt, mdl, tok, history)
    except Exception:
        logger.exception("Generation failed for model=%s skill=%s", model_key, skill_name)
        raise
    finally:
        # Always re-enable adapters so other users aren't affected
        if loaded_skills and not adapters_enabled:
            mdl.enable_adapter_layers()
    return response, skill_name


# ── Background scoring ────────────────────────────────────────────


async def _score_in_background(
    user_id: int,
    user_text: str,
    response: str,
    history_before: list[tuple[str, str]],
) -> None:
    """Fire-and-forget: score one exchange via LLM judges."""
    user_pending_scores[user_id] = user_pending_scores.get(user_id, 0) + 1
    try:
        result = await score_exchange_async(user_text, response, history_before)
        if result:
            user_scores.setdefault(user_id, []).append(result)
    except Exception as e:
        logger.error("Scoring failed for user %s: %s", user_id, e)
    finally:
        user_pending_scores[user_id] = max(
            0, user_pending_scores.get(user_id, 1) - 1
        )


# ── Whisper ───────────────────────────────────────────────────────


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
    user_scores.pop(user_id, None)
    user_pending_scores.pop(user_id, None)

    current_key = user_models.get(user_id, default_model_key)
    current_name = model_manager.display_name(current_key)

    await update.message.reply_text(
        "Hello! I'm a mental health counseling assistant. "
        "You can share what's on your mind and I'll do my best to help.\n\n"
        f"Current model: {current_name}\n\n"
        "You can send text or voice messages — voice messages will be "
        "transcribed and I'll respond to the transcript.\n\n"
        "Supported voice languages: English, Cantonese (廣東話), "
        "Mandarin (普通話).\n\n"
        "Commands:\n"
        "/model - Switch between AI models\n"
        "/adapters - Toggle LoRA adapters on/off\n"
        "/language - Set voice transcription language\n"
        "/score - View conversation quality scores\n"
        "/clear - Reset our conversation history\n\n"
        "Feel free to start whenever you're ready."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command."""
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    user_scores.pop(user_id, None)
    user_pending_scores.pop(user_id, None)
    await update.message.reply_text("Conversation history cleared. Let's start fresh.")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model command — show inline keyboard for model selection."""
    user_id = update.effective_user.id
    current_key = user_models.get(user_id, default_model_key)
    available = model_manager.list_available()

    buttons = []
    for key in available:
        name = model_manager.display_name(key)
        label = f"{'✅ ' if key == current_key else ''}{name}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"model:{key}")])

    await update.message.reply_text(
        "Select a model:", reply_markup=InlineKeyboardMarkup(buttons)
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button press for model selection."""
    query = update.callback_query
    await query.answer()

    if not query.data.startswith("model:"):
        return

    model_key = query.data.removeprefix("model:")
    user_id = query.from_user.id
    current_key = user_models.get(user_id, default_model_key)

    if model_key == current_key:
        await query.edit_message_text(
            f"Already using {model_manager.display_name(model_key)}."
        )
        return

    if model_key not in model_manager.list_available():
        await query.edit_message_text("That model is not available.")
        return

    # Show loading message if model isn't preloaded
    if not model_manager.is_loaded(model_key):
        await query.edit_message_text(
            f"Loading {model_manager.display_name(model_key)}... this may take 20-30s."
        )
        # Trigger load in a thread so we don't block the event loop
        await asyncio.to_thread(model_manager.get, model_key)

    user_models[user_id] = model_key
    user_histories.pop(user_id, None)  # Clear history on model switch

    await query.edit_message_text(
        f"Switched to {model_manager.display_name(model_key)}.\n"
        "Conversation history cleared."
    )


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
    model_key = user_models.get(user_id, default_model_key)

    history = user_histories.get(user_id, [])
    history_snapshot = list(history)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        adapters_on = user_adapters_enabled.get(user_id, True)
        response, skill_name = await asyncio.to_thread(
            route_and_generate, user_text, model_key, history, adapters_on
        )
    except Exception:
        logger.exception("handle_message: generation error for user %s model %s", user_id, model_key)
        await update.message.reply_text(
            "Sorry, something went wrong generating a response. "
            "Please try again or switch models with /model."
        )
        return

    logger.info("User %s [%s] routed to [%s]", user_id, model_key, skill_name)

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

    if _judges_active:
        asyncio.create_task(
            _score_in_background(user_id, user_text, response, history_snapshot)
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming voice messages — transcribe and generate a counselor response."""
    voice = update.message.voice
    if voice.duration > 120:
        await update.message.reply_text(
            "Sorry, I can only process voice messages up to 2 minutes long."
        )
        return

    user_id = update.effective_user.id
    model_key = user_models.get(user_id, default_model_key)
    tmp_path = None
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        await voice_file.download_to_drive(tmp_path)

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
        history_snapshot = list(history)

        async def generate_and_reply() -> str:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            adapters_on = user_adapters_enabled.get(user_id, True)
            resp, skill_name = await asyncio.to_thread(
                route_and_generate, transcript, model_key, history, adapters_on
            )
            logger.info("User %s [%s] (voice) routed to [%s]", user_id, model_key, skill_name)
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

        if _judges_active:
            asyncio.create_task(
                _score_in_background(user_id, transcript, response, history_snapshot)
            )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def score_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /score command — show aggregated quality scores."""
    user_id = update.effective_user.id
    scores = user_scores.get(user_id, [])
    pending = user_pending_scores.get(user_id, 0)
    total = len(user_histories.get(user_id, []))

    if not _judges_active:
        await update.message.reply_text(
            "Scoring is not available — no judge API keys configured."
        )
        return

    report = format_score_report(scores, total, pending)
    await update.message.reply_text(report)


async def adapters_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /adapters command — toggle LoRA adapters on/off for the current user."""
    user_id = update.effective_user.id
    model_key = user_models.get(user_id, default_model_key)

    # Check if the current model even has adapters
    if model_manager.is_loaded(model_key):
        _, _, loaded_skills = model_manager.get(model_key)
    else:
        loaded_skills = []

    if not loaded_skills:
        await update.message.reply_text(
            f"No LoRA adapters are loaded for {model_manager.display_name(model_key)}.\n"
            "Adapters are only available for the Qwen fine-tuned model."
        )
        return

    current = user_adapters_enabled.get(user_id, True)
    new_state = not current
    user_adapters_enabled[user_id] = new_state
    state_label = "ON" if new_state else "OFF"

    await update.message.reply_text(
        f"LoRA adapters: {state_label}\n\n"
        f"{'Skill-specific adapters will be used for responses.' if new_state else 'Adapters disabled — using base fine-tuned weights only.'}\n\n"
        "Use /adapters again to toggle."
    )
    logger.info("User %s set adapters=%s for model=%s", user_id, state_label, model_key)


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model Telegram bot for mental health counseling"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen-ft"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Model keys to make available (default: qwen-ft)",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default="qwen-ft",
        choices=list(MODEL_REGISTRY.keys()),
        help="Default model for new users (default: qwen-ft)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=False,
        help="Pre-load all models across GPUs at startup (requires multi-GPU)",
    )
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="Load models on demand (default)",
    )
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default=None,
        help="Directory containing Qwen skill LoRA adapter subdirectories",
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

    global default_model_key
    default_model_key = args.default_model
    if default_model_key not in args.models:
        args.models.insert(0, default_model_key)

    global skill_router
    skill_router = SkillRouter()
    print(f"Skill router loaded with {len(skill_router.list_skills())} skills")

    global model_manager
    model_manager = ModelManager(
        model_keys=args.models,
        preload=args.preload,
        adapters_dir=args.adapters_dir,
    )

    load_whisper_model(args.whisper_model)

    global default_whisper_language
    default_whisper_language = args.whisper_language

    # Initialise LLM judges for auto-scoring (optional — needs API keys)
    global _judges_active
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    prompt_path = str(_PROJECT_ROOT / "evaluation" / "llm_judge_prompt.md")
    active_judges = init_judges(deepseek_key, gemini_key, prompt_path)
    if active_judges:
        _judges_active = True
        logger.info("LLM judges active: %s", active_judges)
    else:
        logger.info("No judge API keys found — scoring disabled")

    # Configure proxy if HTTPS_PROXY is set (e.g. CSE CUHK HPC cluster)
    builder = Application.builder().token(token)
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy_url:
        logger.info(f"Using proxy: {proxy_url}")
        builder = builder.proxy(proxy_url)
    app = builder.build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("score", score_command))
    app.add_handler(CommandHandler("adapters", adapters_command))
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    model_names = [model_manager.display_name(k) for k in args.models]
    logger.info("Bot started — models: %s, default: %s", model_names, default_model_key)
    app.run_polling()


if __name__ == "__main__":
    main()
