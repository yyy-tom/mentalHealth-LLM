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


def load_model_with_adapters(
    base_model_path: str,
    adapters_dir: str,
    use_4bit: bool = True,
) -> tuple:
    """
    Load the base model and all skill adapters.

    Returns:
        (model, tokenizer, loaded_skills) where loaded_skills is a list of
        skill names that were successfully loaded.
    """
    # Resolve paths
    if not os.path.isabs(base_model_path):
        resolved = PROJECT_ROOT / base_model_path
        if resolved.exists():
            base_model_path = str(resolved)

    if not os.path.isabs(adapters_dir):
        adapters_dir = str(PROJECT_ROOT / adapters_dir)

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

    # Load adapters
    loaded_skills = []
    first_adapter = True

    for skill_name in SKILLS:
        adapter_path = os.path.join(adapters_dir, skill_name)
        if not os.path.exists(adapter_path):
            print(f"  Adapter not found: {adapter_path} — skipping {skill_name}")
            continue

        # Check for adapter files
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
    args = parser.parse_args()

    # Load router
    router = SkillRouter(config_path=args.router_config)
    print(f"Router loaded with {len(router.list_skills())} skills")

    # Load model + adapters
    use_4bit = not args.no_4bit
    model, tokenizer, loaded_skills = load_model_with_adapters(
        args.base_model, args.adapters_dir, use_4bit=use_4bit
    )
    _set_eval_mode(model)

    print(f"\nLoaded {len(loaded_skills)} adapters: {', '.join(loaded_skills)}")
    has_adapters = len(loaded_skills) > 0

    def respond(question: str, history: list = None, force_skill: str = None) -> tuple:
        """Route, activate adapter, and generate response. Returns (response, skill_name)."""
        if force_skill:
            skill_name = force_skill
        else:
            skill_name = router.route(question, history=history)

        # Activate adapter if available
        if has_adapters and skill_name in loaded_skills:
            model.set_adapter(skill_name)
        elif has_adapters and loaded_skills:
            # Fall back to general-support or first loaded adapter
            fallback = "general-support" if "general-support" in loaded_skills else loaded_skills[0]
            model.set_adapter(fallback)
            if skill_name != fallback:
                print(f"  (adapter '{skill_name}' not loaded, falling back to '{fallback}')")

        system_prompt = router.get_system_prompt(skill_name)
        response = generate_response(
            model, tokenizer, question,
            system_prompt=system_prompt,
            history=history,
        )
        return response, skill_name

    # ---------- Interactive mode ----------
    if args.interactive:
        print("\n" + "=" * 60)
        print("Skill-Based Mental Health Counselor (Interactive)")
        print("Type 'quit' to exit, '/skill' to see current routing")
        print("=" * 60)

        conversation_history = []
        history_limit = 6

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
            )
            print(f"\nCounselor: {response}")
            print("-" * 50)

            conversation_history.append((question, response))
            if len(conversation_history) > history_limit:
                conversation_history.pop(0)

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
