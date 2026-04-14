#!/usr/bin/env python3
"""
Generate responses from each trained model for all test samples.

Processes one model at a time, clears CUDA memory between models.
Reuses load_model / generate_response patterns from scripts/evaluate_model.py.

Usage:
    python scripts/evaluation/generate_responses.py \
        --test-set evaluation/test_set.json \
        --output-dir evaluation/responses

    # Run specific models only
    python scripts/evaluation/generate_responses.py \
        --test-set evaluation/test_set.json \
        --output-dir evaluation/responses \
        --models qwen2.5-7b gemma2-9b
"""

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model registry: key -> path (matches config output_dirs)
MODEL_REGISTRY = {
    "qwen2.5-7b": "models/qwen2.5-7b-mental-health-fullft-a100",
    "gemma2-9b": "models/gemma2-9b-mental-health-fullft-a100",
    "mistral-7b": "models/mistral-7b-mental-health-fullft-a100",
    "llama-3.1-8b": "models/llama-3.1-8b-mental-health-fullft-a100",
}

SYSTEM_PROMPT = "You are a compassionate and professional mental health counselor."


def load_model(model_path: str, use_4bit: bool = True):
    """Load the fine-tuned model with optional LoRA adapters.

    Reuses the pattern from scripts/evaluate_model.py.
    """
    print(f"Loading model from {model_path}...")

    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()

    if is_lora:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", model_path)
        print(f"  Base model: {base_model_name}")
    else:
        base_model_name = model_path

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if is_lora:
        print(f"  Loading LoRA adapters from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_lora else base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, user_message: str, max_new_tokens: int = 512) -> str:
    """Generate a single response using chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def unload_model(model, tokenizer):
    """Delete model and free CUDA memory."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_model(model_key: str, model_path: str, samples: list[dict], output_dir: Path):
    """Generate responses for all samples with one model."""
    output_path = output_dir / f"{model_key}.json"

    if output_path.exists():
        print(f"  Output already exists: {output_path}, skipping")
        return

    model, tokenizer = load_model(model_path)
    model.eval()

    responses = []
    for sample in tqdm(samples, desc=f"  {model_key}"):
        t0 = time.time()
        try:
            resp = generate_response(model, tokenizer, sample["user_message"])
        except Exception as e:
            print(f"\n  ERROR on {sample['sample_id']}: {e}")
            resp = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        responses.append({
            "sample_id": sample["sample_id"],
            "user_message": sample["user_message"],
            "model_response": resp,
            "generation_time_seconds": round(elapsed, 2),
        })

    result = {
        "metadata": {
            "model_key": model_key,
            "model_path": model_path,
            "total_samples": len(responses),
            "system_prompt": SYSTEM_PROMPT,
            "generation_params": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "responses": responses,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(responses)} responses to {output_path}")

    unload_model(model, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for LLM judge evaluation")
    parser.add_argument("--test-set", type=str, default="evaluation/test_set.json", help="Path to test set JSON")
    parser.add_argument("--output-dir", type=str, default="evaluation/responses", help="Output directory")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to process (default: all)")
    args = parser.parse_args()

    # Load test set
    with open(args.test_set) as f:
        test_data = json.load(f)
    samples = test_data["samples"]
    print(f"Loaded {len(samples)} test samples from {args.test_set}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to run
    model_keys = args.models or list(MODEL_REGISTRY.keys())

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"Unknown model key: {model_key}, skipping")
            continue

        model_path = MODEL_REGISTRY[model_key]
        print(f"\n{'=' * 60}")
        print(f"Model: {model_key} ({model_path})")
        print(f"{'=' * 60}")

        process_model(model_key, model_path, samples, output_dir)

    print("\nAll models processed.")


if __name__ == "__main__":
    main()
