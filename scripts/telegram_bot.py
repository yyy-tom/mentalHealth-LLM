#!/usr/bin/env python3
"""
Telegram bot for mental health counseling with multi-model support.

Supports 6 model variants (3 fine-tuned + 3 base) with per-user model
selection via /model command. Skill-based routing selects system prompts;
LoRA adapters can be activated for all fine-tuned models when available.

Usage:
    TELEGRAM_BOT_TOKEN="xxx" python3 scripts/telegram_bot.py \
        --models qwen-ft gemma-ft mistral-ft \
        --adapters-dir adapters/qwen \
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
from mental_health_llm.session_outcome import SessionOutcome, OutcomeLogger, format_stats_report
from mental_health_llm.streaming import send_streaming_response
from mental_health_llm.session_store import SQLiteSessionStore
from mental_health_llm.adapter_cache import AdapterCache
from mental_health_llm.context_integration import EnhancedContextManager

# Evaluation harness integration
from evaluation.harness.config import HarnessConfig, FeatureFlags
from evaluation.harness.runner import EvaluationHarness
from evaluation.harness.baseline import BaselineManager
from evaluation.harness.metrics import MetricsAggregator

# Orchestration pipeline (Phase A)
from mental_health_llm.orchestration import (
    TurnState,
    PipelineConfig,
    CounselingPipeline,
    PipelineTrace,
    run_pipeline,
)
from mental_health_llm.orchestration.state import (
    CrisisLevel,
    GuardAction,
    GenerationResult,
    PersistResult,
)

from scripts.evaluation.judge_scoring import (
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

HISTORY_LIMIT = 6  # Kept for backward compatibility; EnhancedContextManager uses hot_size
USE_ENHANCED_CONTEXT = True  # Toggle to enable claw-code context management
USE_ORCHESTRATION = True  # Toggle to enable pipeline orchestration (Phase A)

# ── Harness Globals ───────────────────────────────────────────────

harness_config: HarnessConfig | None = None
evaluation_harness: EvaluationHarness | None = None
baseline_manager: BaselineManager | None = None
metrics_aggregator: MetricsAggregator | None = None
harness_enabled: bool = False

# ── Orchestration Globals ─────────────────────────────────────────

pipeline_config: PipelineConfig | None = None
counseling_pipeline: CounselingPipeline | None = None
last_pipeline_traces: dict[int, PipelineTrace] = {}  # user_id -> last trace

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
ADAPTER_FAMILIES = {"qwen", "gemma", "mistral"}


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

        # Load LoRA adapters for fine-tuned models when configured
        loaded_skills: list[str] = []
        if model_key.endswith("-ft") and self._adapters_dir:
            adapters_root = Path(self._adapters_dir)
            if not adapters_root.is_absolute():
                adapters_root = _PROJECT_ROOT / adapters_root
            adapters_root = _normalize_adapters_root(adapters_root)
            adapters_path = _resolve_adapters_dir_for_model(adapters_root, model_key)
            if adapters_path is None:
                print(f"  No adapters found for {model_key} under {adapters_root}")
            else:
                print(f"  Loading adapters for {model_key} from {adapters_path}")
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


def _has_adapter_files(adapter_path: Path) -> bool:
    """Check whether a skill adapter directory contains LoRA artifacts."""
    return (
        (adapter_path / "adapter_config.json").exists()
        or (adapter_path / "adapter_model.safetensors").exists()
        or (adapter_path / "adapter_model.bin").exists()
    )


def _normalize_adapters_root(adapters_path: Path) -> Path:
    """Normalize adapter path to a shared root.

    Allows passing either:
      - shared root:   <root>/adapters
      - family folder: <root>/adapters/qwen
    """
    if adapters_path.name in ADAPTER_FAMILIES:
        return adapters_path.parent
    return adapters_path


def _resolve_adapters_dir_for_model(adapters_root: Path, model_key: str) -> Path | None:
    """Resolve adapter directory for a model.

    Supported layouts:
      1) Legacy Qwen-only: <adapters_root>/<skill_name>
      2) Per-model: <adapters_root>/<model_key>/<skill_name>
      3) Per-family: <adapters_root>/<family>/<skill_name> (e.g., qwen/gemma/mistral)
    """
    family = model_key.replace("-ft", "")
    candidates = [
        adapters_root / model_key,
        adapters_root / family,
    ]

    # Backward compatibility: older setups store Qwen adapters directly in root.
    if model_key == "qwen-ft":
        candidates.append(adapters_root)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists() or not candidate.is_dir():
            continue
        if any(_has_adapter_files(candidate / skill_name) for skill_name in SKILL_NAMES):
            return candidate
    return None


# ── Global state ──────────────────────────────────────────────────

model_manager: ModelManager | None = None
whisper_model = None
skill_router: SkillRouter | None = None
default_model_key: str = "qwen-ft"

# Session persistence (SQLite-backed, replaces in-memory user_histories)
session_store: SQLiteSessionStore | None = None

# Enhanced context manager (claw-code integration)
enhanced_context: EnhancedContextManager | None = None

# Per-user state (in-memory fallback when session_store is None)
user_histories: dict[int, list[tuple[str, str]]] = {}
user_models: dict[int, str] = {}
user_languages: dict[int, str | None] = {}
user_adapters_enabled: dict[int, bool] = {}  # True = adapters on (default)
user_streaming_enabled: dict[int, bool] = {}  # True = streaming on
user_prompt_enabled: dict[int, bool] = {}  # True = system prompt on (default)

default_whisper_language: str | None = None


def _get_history(user_id: int) -> list[tuple[str, str]]:
    """Get conversation history — from enhanced context, session store, or in-memory."""
    if USE_ENHANCED_CONTEXT and enhanced_context is not None:
        return enhanced_context.get_history(user_id)
    if session_store is not None:
        return session_store.restore_history(user_id)
    return user_histories.get(user_id, [])


def _save_turn(
    user_id: int,
    user_msg: str,
    assistant_msg: str,
    *,
    skill: str = "",
    is_crisis: bool = False,
) -> None:
    """Save a conversation turn to enhanced context, session store, or in-memory cache."""
    model_key = user_models.get(user_id, default_model_key)

    if USE_ENHANCED_CONTEXT and enhanced_context is not None:
        # Use enhanced context manager (handles tiering + compaction + persistence)
        enhanced_context.save_turn(
            user_id=user_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            skill=skill,
            is_crisis=is_crisis,
            model_key=model_key,
        )
        return

    # Fallback to original behavior
    if session_store is not None:
        session_store.save_turn(
            user_id=user_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            skill=skill,
            is_crisis=is_crisis,
            model_key=model_key,
        )
    # Also maintain in-memory cache for the current process
    history = user_histories.get(user_id, [])
    history.append((user_msg, assistant_msg))
    if len(history) > HISTORY_LIMIT:
        history.pop(0)
    user_histories[user_id] = history


def _clear_history(user_id: int, persist_memory: bool = True) -> None:
    """Clear conversation history, optionally persisting to memory."""
    if USE_ENHANCED_CONTEXT and enhanced_context is not None:
        if persist_memory:
            enhanced_context.end_session(user_id, persist_memory=True)
        else:
            enhanced_context.clear_session(user_id)
        return

    # Fallback to original behavior
    if session_store is not None:
        session_store.delete_session(user_id)
    user_histories.pop(user_id, None)


# Per-user scoring state
user_scores: dict[int, list[dict]] = {}
user_pending_scores: dict[int, int] = {}
_judges_active = False

# Session outcome logger
outcome_logger: OutcomeLogger | None = None

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

    # Handle empty/disabled system prompt
    has_prompt = bool(system_prompt and system_prompt.strip())

    if use_system_role and has_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    else:
        # No system prompt, or models that don't support system role
        messages = []

    if history:
        for i, (user_turn, counselor_turn) in enumerate(history):
            content = user_turn
            # Only prepend system prompt if: no system role support, has prompt, and first message
            if not use_system_role and has_prompt and i == 0:
                content = f"{system_prompt}\n\n{user_turn}"
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": counselor_turn})

    user_content = question
    if not use_system_role and has_prompt and not history:
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
    prompt_enabled: bool = True,
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

    # Get system prompt only if prompt is enabled
    system_prompt = skill_router.get_system_prompt(skill_name) if prompt_enabled else ""
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


def load_whisper_model(model_size: str = "large-v3") -> None:
    """Load faster-whisper on the last available GPU (float16), falling back to CPU (int8)."""
    global whisper_model
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        whisper_gpu = gpu_count - 1
        device = "cuda"
        compute_type = "float16"
        print(f"Loading Whisper model: {model_size} on cuda:{whisper_gpu} ({compute_type})")
        whisper_model = WhisperModel(
            model_size, device=device, device_index=whisper_gpu, compute_type=compute_type
        )
    else:
        device = "cpu"
        compute_type = "int8"
        print(f"Loading Whisper model: {model_size} on {device} ({compute_type})")
        whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
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

    # Log session outcome if there was an active session
    prev_history = _get_history(user_id)
    if prev_history and outcome_logger:
        model_key = user_models.get(user_id, default_model_key)
        outcome_logger.log(
            user_id=user_id,
            outcome=SessionOutcome.USER_ENDED,
            model_key=model_key,
            turns=len(prev_history),
        )

    _clear_history(user_id)
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
        "/prompt - Toggle system prompt on/off\n"
        "/streaming - Toggle streaming response mode\n"
        "/language - Set voice transcription language\n"
        "/score - View conversation quality scores\n"
        "/memory - View cross-session memory\n"
        "/harness - View evaluation harness status\n"
        "/stats - View session outcome analytics\n"
        "/clear - Reset our conversation history\n\n"
        "Feel free to start whenever you're ready."
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command."""
    user_id = update.effective_user.id

    # Log session outcome if there was an active session
    prev_history = _get_history(user_id)
    if prev_history and outcome_logger:
        model_key = user_models.get(user_id, default_model_key)
        outcome_logger.log(
            user_id=user_id,
            outcome=SessionOutcome.USER_ENDED,
            model_key=model_key,
            turns=len(prev_history),
        )

    _clear_history(user_id)
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
    _clear_history(user_id)  # Clear history on model switch

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

    history = _get_history(user_id)
    history_snapshot = list(history)
    use_streaming = user_streaming_enabled.get(user_id, False)

    # ── Orchestration Pipeline Path (Phase A) ─────────────────────
    if USE_ORCHESTRATION and counseling_pipeline and not use_streaming:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        try:
            # Build generation function that wraps the existing route_and_generate
            adapters_on = user_adapters_enabled.get(user_id, True)
            prompt_on = user_prompt_enabled.get(user_id, True)

            def _generation_fn(state, triage, retrieval):
                """Generate response using existing model infrastructure."""
                resp, skill = route_and_generate(
                    state.user_message,
                    model_key,
                    state.conversation_history,
                    adapters_on,
                    prompt_on,
                )
                return GenerationResult(
                    node_name="generate",
                    success=True,
                    duration_ms=0.0,
                    response=resp,
                    model_id=model_key,
                    adapter_used=skill,
                )

            def _persist_fn(state, generation, guard, triage):
                """Persist turn using existing save logic."""
                skill = generation.adapter_used if generation else ""
                is_crisis = triage.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL) if triage else False
                _save_turn(user_id, state.user_message, state.final_response, skill=skill, is_crisis=is_crisis)
                return PersistResult(
                    node_name="persist",
                    success=True,
                    duration_ms=0.0,
                    session_saved=True,
                    memory_saved=USE_ENHANCED_CONTEXT,
                )

            # Create per-request pipeline with injected functions
            request_pipeline = CounselingPipeline(
                config=pipeline_config,
                generation_fn=_generation_fn,
                persist_fn=_persist_fn,
            )

            # Build state
            state = TurnState(
                user_id=user_id,
                user_message=user_text,
                conversation_history=history,
                model_id=model_key,
            )

            # Run pipeline in thread pool (generation is CPU-bound)
            state = await asyncio.to_thread(request_pipeline.run, state)

            # Store trace for /trace command
            last_pipeline_traces[user_id] = state.to_trace()

            response = state.final_response
            skill_name = state.generation.adapter_used if state.generation else ""

            # Log crisis escalation
            if state.is_crisis and outcome_logger:
                outcome_logger.log(
                    user_id=user_id,
                    outcome=SessionOutcome.CRISIS_ESCALATED,
                    skill=skill_name,
                    model_key=model_key,
                    turns=len(history) + 1,
                    crisis_detected=True,
                )

            logger.info(
                "User %s [%s] pipeline: %s",
                user_id,
                model_key,
                state.to_trace().summary(),
            )

        except Exception:
            logger.exception("handle_message: pipeline error for user %s model %s", user_id, model_key)
            if outcome_logger:
                outcome_logger.log(
                    user_id=user_id,
                    outcome=SessionOutcome.ERROR,
                    model_key=model_key,
                    turns=len(history),
                )
            await update.message.reply_text(
                "Sorry, something went wrong generating a response. "
                "Please try again or switch models with /model."
            )
            return

        if not response or not response.strip():
            response = (
                "I'm not sure how to respond to that. "
                "Could you tell me more about what's on your mind?"
            )

        await update.message.reply_text(response)

        if _judges_active:
            asyncio.create_task(
                _score_in_background(user_id, user_text, response, history_snapshot)
            )
        return

    # ── Streaming Path ────────────────────────────────────────────
    if use_streaming and skill_router and model_manager:
        # ── Streaming path ───────────────────────────────────────
        try:
            adapters_on = user_adapters_enabled.get(user_id, True)
            prompt_on = user_prompt_enabled.get(user_id, True)
            response, skill_name, crisis_event = await send_streaming_response(
                update,
                context,
                user_text,
                model_key,
                history,
                adapters_on,
                prompt_enabled=prompt_on,
                skill_router=skill_router,
                model_manager=model_manager,
            )
        except Exception:
            logger.exception("handle_message: streaming error for user %s model %s", user_id, model_key)
            if outcome_logger:
                outcome_logger.log(
                    user_id=user_id,
                    outcome=SessionOutcome.ERROR,
                    model_key=model_key,
                    turns=len(history),
                )
            await update.message.reply_text(
                "Sorry, something went wrong generating a response. "
                "Please try again or switch models with /model."
            )
            return

        # Log crisis escalation
        if crisis_event and crisis_event.is_crisis and outcome_logger:
            outcome_logger.log(
                user_id=user_id,
                outcome=SessionOutcome.CRISIS_ESCALATED,
                skill=skill_name,
                model_key=model_key,
                turns=len(history) + 1,
                crisis_detected=True,
            )

    else:
        # ── Synchronous path (original) ──────────────────────────
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        try:
            adapters_on = user_adapters_enabled.get(user_id, True)
            prompt_on = user_prompt_enabled.get(user_id, True)
            response, skill_name = await asyncio.to_thread(
                route_and_generate, user_text, model_key, history, adapters_on, prompt_on
            )
        except Exception:
            logger.exception("handle_message: generation error for user %s model %s", user_id, model_key)
            if outcome_logger:
                outcome_logger.log(
                    user_id=user_id,
                    outcome=SessionOutcome.ERROR,
                    model_key=model_key,
                    turns=len(history),
                )
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

    is_crisis = skill_name == "crisis-intervention"
    _save_turn(user_id, user_text, response, skill=skill_name, is_crisis=is_crisis)

    # Only send reply for non-streaming path (streaming already sent it)
    if not use_streaming:
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

        history = _get_history(user_id)
        history_snapshot = list(history)

        async def generate_and_reply() -> str:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            adapters_on = user_adapters_enabled.get(user_id, True)
            prompt_on = user_prompt_enabled.get(user_id, True)
            resp, skill_name = await asyncio.to_thread(
                route_and_generate, transcript, model_key, history, adapters_on, prompt_on
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

        _save_turn(user_id, transcript, response)

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
    total = len(_get_history(user_id))

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
            "Adapters are only available when skill checkpoints exist for the current fine-tuned model."
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


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command — show session outcome analytics."""
    user_id = update.effective_user.id

    if outcome_logger is None:
        await update.message.reply_text("Session analytics are not available.")
        return

    # Check for --all flag (admin/debug use)
    show_all = context.args and context.args[0] == "--all"

    if show_all:
        records = outcome_logger.load_all()
        report = format_stats_report(records)
    else:
        records = outcome_logger.load_for_user(user_id)
        report = format_stats_report(records, user_id=user_id)

    await update.message.reply_text(report)


async def streaming_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /streaming command — toggle streaming mode on/off."""
    user_id = update.effective_user.id
    current = user_streaming_enabled.get(user_id, False)
    new_state = not current
    user_streaming_enabled[user_id] = new_state
    state_label = "ON" if new_state else "OFF"

    await update.message.reply_text(
        f"Streaming mode: {state_label}\n\n"
        f"{'Responses will be delivered in chunks as they generate.' if new_state else 'Responses will be delivered all at once (default).'}\n\n"
        "Use /streaming again to toggle."
    )
    logger.info("User %s set streaming=%s", user_id, state_label)


async def prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /prompt command — toggle system prompt on/off for the current user."""
    user_id = update.effective_user.id

    current = user_prompt_enabled.get(user_id, True)
    new_state = not current
    user_prompt_enabled[user_id] = new_state
    state_label = "ON" if new_state else "OFF"

    await update.message.reply_text(
        f"System prompt: {state_label}\n\n"
        f"{'Skill-specific system prompts will guide responses.' if new_state else 'System prompts disabled — using only your input and conversation history.'}\n\n"
        "Use /prompt again to toggle."
    )
    logger.info("User %s set prompt=%s", user_id, state_label)


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /memory command — show cross-session memory and key facts."""
    user_id = update.effective_user.id

    if not USE_ENHANCED_CONTEXT or enhanced_context is None:
        await update.message.reply_text(
            "Memory features are not enabled. "
            "The bot is using basic session persistence."
        )
        return

    # Get key facts about the user
    key_facts = enhanced_context.get_user_key_facts(user_id, limit=5)

    # Get current context to search for relevant memories
    history = _get_history(user_id)
    if history:
        recent_context = " ".join(msg for msg, _ in history[-3:])
        relevant_memories = enhanced_context.recall_relevant_memories(
            user_id, recent_context, limit=3
        )
    else:
        relevant_memories = []

    # Format response
    response_parts = ["📚 **Your Memory Profile**\n"]

    if key_facts:
        response_parts.append("**Key Facts:**")
        for fact in key_facts:
            response_parts.append(f"• {fact}")
        response_parts.append("")

    if relevant_memories:
        response_parts.append("**Related Past Sessions:**")
        for i, memory in enumerate(relevant_memories, 1):
            # Truncate long memories
            short = memory[:100] + "..." if len(memory) > 100 else memory
            response_parts.append(f"{i}. {short}")
        response_parts.append("")

    if not key_facts and not relevant_memories:
        response_parts.append(
            "No memory stored yet. As we talk, I'll remember "
            "important details to provide more personalized support."
        )

    await update.message.reply_text("\n".join(response_parts))


async def harness_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /harness command with lightweight subcommands."""
    action = (context.args[0].strip().lower() if context.args else "status")

    if not harness_enabled or evaluation_harness is None:
        await update.message.reply_text(
            "⚙️ **Evaluation Harness**\n\n"
            "Status: Disabled\n\n"
            "Start the bot with `--enable-harness` to enable evaluation tracking."
        )
        return

    if action not in {"status", "features", "baseline"}:
        await update.message.reply_text(
            "Unknown harness command.\n\n"
            "Use one of:\n"
            "• /harness status\n"
            "• /harness features\n"
            "• /harness baseline"
        )
        return

    if action == "features":
        lines = ["⚙️ **Harness Features**\n"]
        if harness_config:
            for feat, enabled in harness_config.features.to_dict().items():
                status = "✓" if enabled else "✗"
                lines.append(f"{status} {feat.replace('_', ' ').title()}")
        else:
            lines.append("Harness config not loaded.")
        await update.message.reply_text("\n".join(lines))
        return

    if action == "baseline":
        lines = ["⚙️ **Harness Baseline**\n"]
        if baseline_manager:
            baselines = baseline_manager.list_baselines()
            if baselines:
                latest = baseline_manager.get_latest()
                if latest:
                    lines.append(f"Latest baseline: {latest.id}")
                    lines.append(f"Commit: {latest.commit}")
                    lines.append(f"Model: {latest.model}")
                    dims = latest.metrics.get("dimensions", {})
                    if dims:
                        lines.append("")
                        lines.append("Top dimensions:")
                        for dim in (
                            "empathy",
                            "cbt_techniques",
                            "guided_discovery",
                            "safety_awareness",
                        ):
                            dim_stats = dims.get(dim)
                            if isinstance(dim_stats, dict) and "mean" in dim_stats:
                                lines.append(f"• {dim}: {dim_stats['mean']:.3f}")
                else:
                    lines.append("No baseline metadata available.")
            else:
                lines.append("No baselines captured yet.")
        else:
            lines.append("Baseline manager not initialized.")
        await update.message.reply_text("\n".join(lines))
        return

    # status
    lines = ["⚙️ **Evaluation Harness**\n", "Status: ✅ Enabled\n"]
    if harness_config:
        lines.append("**Active Features:**")
        for feat, enabled in harness_config.features.to_dict().items():
            status = "✓" if enabled else "✗"
            lines.append(f"  {status} {feat.replace('_', ' ').title()}")
        lines.append("")
    if baseline_manager:
        baselines = baseline_manager.list_baselines()
        if baselines:
            lines.append(f"**Baselines:** {len(baselines)} captured")
            latest = baseline_manager.get_latest()
            if latest:
                lines.append(f"  Latest: {latest.id} @ {latest.commit}")
        else:
            lines.append("**Baselines:** None captured yet")
        lines.append("")
    lines.append("**Commands:**")
    lines.append("• /harness status")
    lines.append("• /harness baseline")
    lines.append("• /harness features")

    await update.message.reply_text("\n".join(lines))


async def trace_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /trace command — show the last pipeline trace for this user."""
    user_id = update.effective_user.id

    if not USE_ORCHESTRATION:
        await update.message.reply_text("Orchestration pipeline is disabled.")
        return

    trace = last_pipeline_traces.get(user_id)
    if not trace:
        await update.message.reply_text("No pipeline trace available yet. Send a message first.")
        return

    # Build trace report
    lines = ["🔍 **Last Pipeline Trace**\n"]
    lines.append(f"Turn: `{trace.turn_id}`")
    lines.append(f"Duration: {trace.total_duration_ms:.0f}ms")
    if trace.is_crisis:
        lines.append("⚠️ **CRISIS DETECTED**")
    lines.append("")
    lines.append("**Node Timings:**")
    for node in trace.nodes:
        status = "✓" if node.get("success") else "✗"
        duration = node.get("duration_ms", 0)
        name = node.get("node", "?")
        extra = ""
        if name == "triage":
            crisis = node.get("crisis_level", "unknown")
            skill = node.get("detected_skill", "")
            extra = f" [crisis={crisis}, skill={skill}]"
        elif name == "guard":
            action = node.get("action", "?")
            extra = f" [action={action}]"
        lines.append(f"  {status} {name}: {duration:.0f}ms{extra}")

    lines.append("")
    lines.append(f"Response length: {trace.final_response_length} chars")

    await update.message.reply_text("\n".join(lines))


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
        help=(
            "Directory containing skill LoRA adapters. "
            "Use either adapters root (e.g. adapters/) or a family path (e.g. adapters/qwen/). "
            "Supports <dir>/<skill>, <dir>/<model-key>/<skill>, or <dir>/<family>/<skill>."
        ),
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="Whisper model size for voice transcription (default: large-v3)",
    )
    parser.add_argument(
        "--whisper_language",
        type=str,
        default=None,
        help="Force whisper language code, e.g. 'yue' for Cantonese (default: auto-detect)",
    )
    parser.add_argument(
        "--session-db",
        type=str,
        default=None,
        help="SQLite database path for session persistence (enables session resumption on restart)",
    )
    parser.add_argument(
        "--enable-harness",
        action="store_true",
        default=False,
        help="Enable evaluation harness for metrics tracking and baseline comparison",
    )
    parser.add_argument(
        "--harness-config",
        type=str,
        default=None,
        help="Path to harness configuration file (YAML or JSON)",
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

    # Initialise session persistence
    global session_store
    if args.session_db:
        session_store = SQLiteSessionStore(db_path=args.session_db)
        print(f"Session persistence enabled: {args.session_db}")
    else:
        # Default location
        default_db = str(_PROJECT_ROOT / "data" / "sessions.db")
        session_store = SQLiteSessionStore(db_path=default_db)
        print(f"Session persistence enabled: {default_db}")

    # Initialize enhanced context manager (claw-code integration)
    global enhanced_context
    if USE_ENHANCED_CONTEXT:
        memory_db = str(_PROJECT_ROOT / "data" / "memory.db")
        enhanced_context = EnhancedContextManager(
            db_path=session_store._db_path if session_store else default_db,
            memory_db_path=memory_db,
            hot_size=4,  # Recent 4 turn pairs always verbatim
            warm_size=6,  # Important turns kept in detail
            target_tokens=1024,  # Context token budget
        )
        print(f"Enhanced context manager enabled (memory: {memory_db})")

    # Initialise session outcome logger
    global outcome_logger
    outcome_logger = OutcomeLogger(str(_PROJECT_ROOT / "logs" / "session_outcomes.jsonl"))
    print(f"Session outcome logging to: {outcome_logger.path}")

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

    # Initialize evaluation harness (optional — for metrics tracking)
    global harness_enabled, harness_config, evaluation_harness, baseline_manager, metrics_aggregator
    if args.enable_harness:
        try:
            # Load config
            if args.harness_config:
                config_path = Path(args.harness_config)
                if config_path.suffix in (".yaml", ".yml"):
                    harness_config = HarnessConfig.from_yaml(config_path)
                else:
                    harness_config = HarnessConfig.from_json(config_path)
            else:
                harness_config = HarnessConfig(project_root=_PROJECT_ROOT)
            
            # Sync feature flags with USE_ENHANCED_CONTEXT
            if USE_ENHANCED_CONTEXT:
                harness_config.features.tiered_context = True
                harness_config.features.multi_layer_compaction = True
                harness_config.features.memory_persistence = True
            
            # Initialize components
            evaluation_harness = EvaluationHarness(harness_config)
            baseline_manager = BaselineManager(harness_config)
            metrics_aggregator = MetricsAggregator(
                bootstrap_samples=harness_config.bootstrap_samples,
                confidence_level=harness_config.confidence_level,
            )
            
            harness_enabled = True
            
            # Log status
            baselines = baseline_manager.list_baselines()
            logger.info(
                "Evaluation harness enabled: %d baselines, features=%s",
                len(baselines),
                harness_config.features.to_dict(),
            )
            print(f"Evaluation harness enabled ({len(baselines)} baselines)")
        except Exception as e:
            logger.warning("Failed to initialize evaluation harness: %s", e)
            harness_enabled = False

    # Initialize orchestration pipeline (Phase A)
    global pipeline_config, counseling_pipeline
    if USE_ORCHESTRATION:
        pipeline_config = PipelineConfig(
            enable_retrieval=False,  # Phase B — not yet implemented
            enable_guard=True,
            enable_memory_persist=USE_ENHANCED_CONTEXT,
            crisis_escalation_threshold=0.8,
        )
        # Pipeline is created without generation_fn / persist_fn here;
        # those are provided per-request in handle_message to capture user context.
        counseling_pipeline = CounselingPipeline(config=pipeline_config)
        logger.info("Orchestration pipeline enabled: %s", pipeline_config)
        print("Orchestration pipeline enabled (5-node executor)")

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
    app.add_handler(CommandHandler("prompt", prompt_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("streaming", streaming_command))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("harness", harness_command))
    app.add_handler(CommandHandler("trace", trace_command))
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Log startup configuration
    model_names = [model_manager.display_name(k) for k in args.models]
    logger.info("Bot started — models: %s, default: %s", model_names, default_model_key)
    if harness_enabled:
        logger.info("Harness enabled — use /harness to view status")
    if USE_ORCHESTRATION:
        logger.info("Orchestration enabled — use /trace to view last pipeline trace")
    app.run_polling()


if __name__ == "__main__":
    main()
