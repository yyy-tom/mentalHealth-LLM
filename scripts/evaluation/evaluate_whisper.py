#!/usr/bin/env python3
"""Evaluate Whisper ASR accuracy on mental-health audio files."""

import argparse
import re
import string
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel
from jiwer import cer, wer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TYPE_MAP = {
    1: "Type1_Clear_Neutral",
    2: "Type2_Fast_Speech",
    3: "Type3_Background_Noise",
    4: "Type4_Emotional_Speech",
    5: "Type5_Long_Speech",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Whisper ASR accuracy")
    p.add_argument("--audio-dir", default=str(PROJECT_ROOT / "Fyp audio"))
    p.add_argument("--ground-truth", default=str(PROJECT_ROOT / "Fyp audio" / "fyp audio message.md"))
    p.add_argument("--model-size", default="base", choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--output-csv", default=str(PROJECT_ROOT / "evaluation" / "whisper_results.csv"))
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def parse_ground_truth(md_path: Path) -> dict[int, str]:
    pattern = re.compile(r"^(\d+)\\?\.\s+(.+)")
    results = {}
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                idx = int(m.group(1))
                text = m.group(2).strip().rstrip("\\").strip()
                results[idx] = text
    return results


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def load_model(model_size: str, device: str) -> WhisperModel:
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = "float16" if device.startswith("cuda") else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_file(model: WhisperModel, audio_path: Path, language: str = "en") -> str:
    segments, _ = model.transcribe(str(audio_path), beam_size=5, language=language)
    return " ".join(seg.text.strip() for seg in segments)


def compute_metrics(hypothesis: str, reference: str) -> dict:
    return {
        "wer": round(wer(reference, hypothesis), 4),
        "cer": round(cer(reference, hypothesis), 4),
    }


def get_type(file_num: int) -> int:
    return (file_num - 1) // 10 + 1


def run_evaluation(args) -> pd.DataFrame:
    ground_truth = parse_ground_truth(Path(args.ground_truth))
    model = load_model(args.model_size, args.device)
    rows = []

    for i in tqdm(range(1, 51), desc="Transcribing"):
        audio_path = Path(args.audio_dir) / f"{i}.m4a"
        if not audio_path.exists():
            print(f"Warning: {audio_path} not found, skipping")
            continue
        if i not in ground_truth:
            print(f"Warning: no ground truth for file {i}, skipping")
            continue

        hypothesis_raw = transcribe_file(model, audio_path, args.language)
        reference_norm = normalize(ground_truth[i])
        hypothesis_norm = normalize(hypothesis_raw)
        metrics = compute_metrics(hypothesis_norm, reference_norm)
        speech_type = get_type(i)

        rows.append({
            "file": i,
            "type_id": speech_type,
            "type_name": TYPE_MAP[speech_type],
            "reference": ground_truth[i],
            "hypothesis": hypothesis_raw,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    summary = df.groupby("type_name")[["wer", "cer"]].mean().round(4)
    overall = pd.DataFrame(
        [df[["wer", "cer"]].mean().round(4)],
        index=["OVERALL"],
    )
    summary = pd.concat([summary, overall])
    print("\n=== Whisper Evaluation Summary ===\n")
    print(summary.to_string())
    print()


def main():
    args = parse_args()
    df = run_evaluation(args)
    if df.empty:
        print("No files were evaluated.")
        return

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = df.groupby("type_name")[["wer", "cer"]].mean().round(4)
    overall = pd.DataFrame(
        [df[["wer", "cer"]].mean().round(4)],
        index=["OVERALL"],
    )
    summary = pd.concat([summary, overall])
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path)

    print_summary(df)
    print(f"Results saved to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
