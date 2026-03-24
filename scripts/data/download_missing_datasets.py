#!/usr/bin/env python3
"""
Download and process the 3 missing datasets:
  1. counsel_chat_processed        — from GitHub (no auth)
  2. kaggle_mental_health_nguyen   — from Kaggle (needs API key)
  3. crisis_detection_processed    — from Kaggle (needs API key)

Usage:
    python scripts/data/download_missing_datasets.py

    # Skip Kaggle datasets (no API key)
    python scripts/data/download_missing_datasets.py --skip_kaggle
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATASETS_DIR = PROJECT_ROOT / "datasets"


def run(cmd, cwd=None):
    """Run a shell command and stream output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or PROJECT_ROOT)
    return result.returncode == 0


def download_counsel_chat():
    """Download Counsel Chat from GitHub and process it."""
    output = DATASETS_DIR / "counsel_chat_processed"
    if output.exists():
        print("counsel_chat_processed already exists, skipping.")
        return True

    print("\n" + "=" * 60)
    print("Downloading Counsel Chat from GitHub...")
    print("=" * 60)

    repo_dir = PROJECT_ROOT / "counsel-chat"
    csv_path = repo_dir / "data" / "20200325_counsel_chat.csv"

    # Clone if needed
    if not csv_path.exists():
        if repo_dir.exists():
            # Directory exists but CSV missing, try pulling
            run("git pull", cwd=repo_dir)
        else:
            if not run("git clone https://github.com/nbertagnolli/counsel-chat.git"):
                print("ERROR: Failed to clone counsel-chat repo")
                return False

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return False

    # Process
    print("Processing Counsel Chat...")
    return run(
        f"python3 scripts/data/prepare_counsel_dataset.py "
        f"--input_csv {csv_path} "
        f"--output_dir {output}"
    )


def download_kaggle_nguyen():
    """Download Kaggle Mental Health (Nguyen) dataset."""
    output = DATASETS_DIR / "kaggle_mental_health_nguyen_processed_combined"
    if output.exists():
        print("kaggle_mental_health_nguyen_processed_combined already exists, skipping.")
        return True

    print("\n" + "=" * 60)
    print("Downloading Kaggle Mental Health (Nguyen)...")
    print("=" * 60)

    raw_dir = DATASETS_DIR / "kaggle_mental_health_nguyen"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Try downloading via kaggle CLI
    if not run(
        f"kaggle datasets download -d thedevastator/nlp-mental-health-conversations "
        f"-p {raw_dir} --unzip"
    ):
        print("ERROR: Kaggle download failed.")
        print("  Make sure kaggle is installed: pip install kaggle")
        print("  And credentials are at ~/.kaggle/kaggle.json")
        return False

    # Process
    print("Processing Kaggle Nguyen dataset...")
    return run(
        f"python3 scripts/data/prepare_kaggle_dataset.py "
        f"--input_dir {raw_dir} "
        f"--output_dir {output}"
    )


def download_crisis_detection():
    """Download Suicide Watch dataset from Kaggle and process it."""
    output = DATASETS_DIR / "crisis_detection_processed"
    if output.exists():
        print("crisis_detection_processed already exists, skipping.")
        return True

    print("\n" + "=" * 60)
    print("Downloading Suicide Watch (Crisis Detection)...")
    print("=" * 60)

    raw_dir = DATASETS_DIR / "kaggle_suicide_watch"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "Suicide_Detection.csv"

    # Download if CSV doesn't exist
    if not csv_path.exists():
        if not run(
            f"kaggle datasets download -d nikhileswarkomati/suicide-watch "
            f"-p {raw_dir} --unzip"
        ):
            print("ERROR: Kaggle download failed.")
            print("  Make sure kaggle is installed: pip install kaggle")
            print("  And credentials are at ~/.kaggle/kaggle.json")
            return False

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return False

    # Process using the main download script (it has the crisis processing logic)
    print("Processing crisis detection dataset...")
    return run(
        f"python3 scripts/download_and_process_datasets.py "
        f"--suicide_csv {csv_path} "
        f"--skip_downloads"
    )


def main():
    parser = argparse.ArgumentParser(description="Download missing datasets")
    parser.add_argument("--skip_kaggle", action="store_true",
                        help="Skip datasets that require Kaggle API key")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup Kaggle credentials if not present
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        print("Setting up Kaggle credentials...")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text('{"username":"yuyanyuk","key":"b2732b941a0d12912406a92478d7f3c2"}')
        kaggle_json.chmod(0o600)
        print(f"  Saved to {kaggle_json}")

    results = {}

    # 1. Counsel Chat (always available)
    results["counsel_chat"] = download_counsel_chat()

    if not args.skip_kaggle:
        # Check kaggle is available
        try:
            subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("\nKaggle CLI not found. Installing...")
            run("pip install kaggle")

        # 2. Kaggle Nguyen
        results["kaggle_nguyen"] = download_kaggle_nguyen()

        # 3. Crisis Detection
        results["crisis_detection"] = download_crisis_detection()
    else:
        print("\nSkipping Kaggle datasets (--skip_kaggle)")
        results["kaggle_nguyen"] = None
        results["crisis_detection"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        if ok is None:
            status = "SKIPPED"
        elif ok:
            status = "OK"
        else:
            status = "FAILED"
        print(f"  {name:40s} {status}")

    print("\nNext: re-run combine to rebuild the combined dataset:")
    print("  python3 scripts/data/combine_all_datasets.py --output_dir datasets/all_mental_health_combined")


if __name__ == "__main__":
    main()
