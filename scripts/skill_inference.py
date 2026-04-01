#!/usr/bin/env python3
"""
Multi-adapter inference with skill-based routing.

Loads the frozen base model once and all 6 LoRA skill adapters. Uses the
SkillRouter to classify each user message and activate the appropriate adapter
via model.set_adapter().

Usage:
    # Interactive mode
    python scripts/skill_inference.py --interactive

    # Single question
    python scripts/skill_inference.py --question "I want to kill myself"

    # Force a specific skill (bypass router)
    python scripts/skill_inference.py --question "What is depression?" --force-skill psychoeducation

    # Disable 4-bit quantization
    python scripts/skill_inference.py --interactive --no-4bit
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configure HF cache before importing torch/transformers
# ---------------------------------------------------------------------------
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


_configure_large_disk_cache()

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path for mental_health_llm imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mental_health_llm.skill_router import SkillRouter
from mental_health_llm.compaction import ConversationCompactor
from mental_health_llm.response_guard import ResponseGuard
from mental_health_llm.prompt_builder import TherapyPromptBuilder
from mental_health_llm.adapter_cache import AdapterCache
from mental_health_llm.session_store import SQLiteSessionStore

# ---------------------------------------------------------------------------
# Skill adapter definitions
# ---------------------------------------------------------------------------
SKILLS = [
    "crisis-intervention",
    "general-support",
    "cbt-therapy",
    "empathetic-listening",
    "psychoeducation",
    "professional-counseling",
]

DEFAULT_BASE_MODEL = "models/qwen2.5-7b-mental-health-fullft-a100"
DEFAULT_ADAPTERS_DIR = "adapters"


def load_base_model(
    base_model_path: str,
    use_4bit: bool = True,
) -> tuple:
    """Load the base model and tokenizer (without adapters).

    Returns:
        (model, tokenizer)
    """
    # Resolve paths
    if not os.path.isabs(base_model_path):
        resolved = PROJECT_ROOT / base_model_path
        if resolved.exists():
            base_model_path = str(resolved)

    print(f"Loading base model: {base_model_path}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("Using 4-bit quantization")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


def load_model_with_adapters(
    base_model_path: str,
    adapters_dir: str,
    use_4bit: bool = True,
    *,
    lazy: bool = False,
    max_cached: int = 3,
) -> tuple:
    """Load the base model and skill adapters (eager or lazy).

    Args:
        base_model_path: Path to the base model.
        adapters_dir: Directory containing adapter subdirectories.
        use_4bit: Enable 4-bit quantization.
        lazy: If True, use AdapterCache for lazy LRU loading instead of
              loading all adapters at startup.
        max_cached: Maximum adapters in cache (only used when lazy=True).

    Returns:
        If lazy=False (default): ``(model, tokenizer, loaded_skills)``
        If lazy=True: ``(adapter_cache, tokenizer, [])`` — adapters are
        loaded on demand via ``adapter_cache.ensure_loaded(skill)``.
    """
    model, tokenizer = load_base_model(base_model_path, use_4bit)

    if not os.path.isabs(adapters_dir):
        adapters_dir = str(PROJECT_ROOT / adapters_dir)

    if lazy:
        cache = AdapterCache(
            max_size=max_cached,
            adapters_dir=adapters_dir,
        )
        cache.attach(model)
        # Pre-load the crisis adapter so it's always available
        cache.preload_pinned()
        print(f"Adapter cache initialized (max={max_cached}, pinned=crisis-intervention)")
        return cache, tokenizer, cache.loaded_skills

    # --- Eager loading (original behaviour) ---
    loaded_skills = []
    first_adapter = True

    for skill_name in SKILLS:
        adapter_path = os.path.join(adapters_dir, skill_name)
        if not os.path.exists(adapter_path):
            print(f"  Adapter not found: {adapter_path} — skipping {skill_name}")
            continue

        has_adapter = (
            os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
            or os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))
            or os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
        )
        if not has_adapter:
            print(f"  No adapter files in {adapter_path} — skipping {skill_name}")
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

    if not loaded_skills:
        print("WARNING: No adapters loaded. Model will use base weights only.")

    return model, tokenizer, loaded_skills


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    history: list = None,
    max_length: int = 1024,
) -> str:
    """Generate a response using the Qwen chat template."""
    messages = []

    # System prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Conversation history
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    # Current question
    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

    # Decode only the generated tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Clean up response
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

    return response.strip()


def _set_eval_mode(model):
    """Set model to evaluation mode (disables dropout etc.)."""
    model.training = False
    for module in model.modules():
        module.training = False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-adapter skill-based inference"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default=DEFAULT_ADAPTERS_DIR,
        help=f"Directory containing adapter subdirectories (default: {DEFAULT_ADAPTERS_DIR})",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--force-skill",
        type=str,
        choices=SKILLS,
        default=None,
        help="Force a specific skill adapter (bypass router)",
    )
    parser.add_argument(
        "--router-config",
        type=str,
        default=None,
        help="Path to skills_config.json for the router",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Enable lazy adapter loading with LRU cache (max 3 adapters)",
    )
    parser.add_argument(
        "--max-cached",
        type=int,
        default=3,
        help="Maximum adapters in LRU cache (default: 3, used with --lazy)",
    )
    parser.add_argument(
        "--session-db",
        type=str,
        default=None,
        help="SQLite database path for session persistence (default: no persistence)",
    )
    args = parser.parse_args()

    # Load router
    router = SkillRouter(config_path=args.router_config)
    print(f"Router loaded with {len(router.list_skills())} skills")

    # Initialize compaction, guard, and prompt builder
    compactor = ConversationCompactor(max_tokens=768, trigger_threshold=600, preserve_recent=4)
    guard = ResponseGuard()
    prompt_builder = TherapyPromptBuilder(config_path=args.router_config)

    # Load model + adapters
    use_4bit = not args.no_4bit
    result = load_model_with_adapters(
        args.base_model, args.adapters_dir, use_4bit=use_4bit,
        lazy=args.lazy, max_cached=args.max_cached,
    )

    adapter_cache = None
    if args.lazy:
        adapter_cache, tokenizer, loaded_skills = result
        model = adapter_cache.model
        _set_eval_mode(model)
        print(f"\nLazy adapter cache active (max {args.max_cached})")
    else:
        model, tokenizer, loaded_skills = result
        _set_eval_mode(model)
        print(f"\nLoaded {len(loaded_skills)} adapters: {', '.join(loaded_skills)}")

    has_adapters = len(loaded_skills) > 0 or adapter_cache is not None

    # Initialize session store if requested
    session_store = None
    if args.session_db:
        session_store = SQLiteSessionStore(db_path=args.session_db)
        print(f"Session persistence enabled: {args.session_db}")

    def respond(
        question: str,
        history: list = None,
        force_skill: str = None,
        compacted=None,
        crisis_turn_indices: set = None,
    ) -> tuple:
        """Route, activate adapter, and generate response. Returns (response, skill_name)."""
        nonlocal model  # model ref may change when adapter_cache wraps it

        if force_skill:
            skill_name = force_skill
        else:
            skill_name = router.route(question, history=history)

        # Activate adapter — lazy or eager
        if adapter_cache is not None:
            if adapter_cache.ensure_loaded(skill_name):
                model = adapter_cache.model
                model.set_adapter(skill_name)
            elif adapter_cache.loaded_skills:
                fallback = (
                    "general-support"
                    if "general-support" in adapter_cache.loaded_skills
                    else adapter_cache.loaded_skills[0]
                )
                model = adapter_cache.model
                model.set_adapter(fallback)
                if skill_name != fallback:
                    print(f"  (adapter '{skill_name}' not available, falling back to '{fallback}')")
        elif has_adapters and skill_name in loaded_skills:
            model.set_adapter(skill_name)
        elif has_adapters and loaded_skills:
            fallback = "general-support" if "general-support" in loaded_skills else loaded_skills[0]
            model.set_adapter(fallback)
            if skill_name != fallback:
                print(f"  (adapter '{skill_name}' not loaded, falling back to '{fallback}')")

        # Determine crisis level for prompt building
        crisis_level = "none"
        if skill_name == "crisis-intervention":
            crisis_level = "high"

        # Build dynamic system prompt
        session_summary = ""
        if compacted and compacted.was_compacted:
            session_summary = compacted.summary

        system_prompt = (
            prompt_builder
            .with_skill(skill_name)
            .with_crisis_context(crisis_level)
            .with_user_profile(region="HK")
            .with_session_summary(session_summary)
            .build()
        )

        # Use compacted history if available
        gen_history = compacted.to_history_pairs() if compacted else history

        response = generate_response(
            model, tokenizer, question,
            system_prompt=system_prompt,
            history=gen_history,
        )

        # Post-generation guard
        guard_result = guard.validate(response, skill=skill_name, crisis_level=crisis_level)
        response = guard_result.response

        return response, skill_name

    # ---------- Interactive mode ----------
    if args.interactive:
        print("\n" + "=" * 60)
        print("Skill-Based Mental Health Counselor (Interactive)")
        print("Type 'quit' to exit, '/skill' to see current routing")
        print("=" * 60)

        # Restore session from persistence if available
        user_id = 0  # single-user CLI session
        conversation_history = []
        crisis_turn_indices: set[int] = set()
        compacted = None

        if session_store:
            session = session_store.load_session(user_id)
            if session:
                conversation_history = [tuple(p) for p in session["messages"]]
                crisis_turn_indices = set(session["crisis_flags"])
                print(f"Restored session: {len(conversation_history)} turns")
                if crisis_turn_indices:
                    print(f"  Crisis turns: {crisis_turn_indices}")
            else:
                print("No previous session found — starting fresh.")

        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            if question == "/skill":
                if conversation_history:
                    last_q = conversation_history[-1][0]
                    _, conf, details = router.route_with_confidence(last_q)
                    print(f"Last routing: confidence={conf:.2f}")
                    for name, info in details["scores"].items():
                        if info["score"] > 0:
                            print(f"  {name}: {info['score']:.2f} (kw={info['keyword_matches']}, pat={info['pattern_matches']})")
                else:
                    print("No messages yet.")
                continue

            if not question:
                continue

            # Show routing
            skill_name_preview = args.force_skill or router.route(question)
            print(f"[{skill_name_preview}]")

            response, skill_name = respond(
                question,
                history=conversation_history,
                force_skill=args.force_skill,
                compacted=compacted,
                crisis_turn_indices=crisis_turn_indices,
            )
            print(f"\nCounselor: {response}")
            print("-" * 50)

            conversation_history.append((question, response))
            is_crisis = skill_name == "crisis-intervention"
            if is_crisis:
                crisis_turn_indices.add(len(conversation_history) - 1)

            # Persist turn
            if session_store:
                session_store.save_turn(
                    user_id=user_id,
                    user_msg=question,
                    assistant_msg=response,
                    skill=skill_name,
                    is_crisis=is_crisis,
                )

            # Compact instead of hard-drop
            compacted = compactor.compact(conversation_history, crisis_turn_indices)

    # ---------- Single question ----------
    elif args.question:
        response, skill_name = respond(args.question, force_skill=args.force_skill)
        print(f"\n[Skill: {skill_name}]")
        print(f"Question: {args.question}")
        print(f"\nCounselor: {response}")

    # ---------- Default test prompts ----------
    else:
        test_prompts = [
            "I want to kill myself. I don't see the point anymore.",
            "I've been feeling really anxious about my job interview. How can I calm my nerves?",
            "What is cognitive behavioral therapy and how does it work?",
            "What are the symptoms of depression?",
            "I feel so lonely. No one understands what I'm going through.",
            "My partner and I keep arguing. How can we improve our communication?",
        ]

        print("\nRunning test prompts:")
        print("=" * 60)

        for prompt in test_prompts:
            skill_name_preview = router.route(prompt)
            print(f"\n[{skill_name_preview}]")
            print(f"User: {prompt}")

            response, skill_name = respond(prompt, force_skill=args.force_skill)
            print(f"Counselor: {response}")
            print("-" * 50)


if __name__ == "__main__":
    main()
