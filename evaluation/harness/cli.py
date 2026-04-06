#!/usr/bin/env python3
"""
Command-line interface for evaluation harness.

Usage:
    # Run evaluation
    python -m evaluation.harness.cli run --model qwen-ft --test-suite all
    
    # Capture baseline
    python -m evaluation.harness.cli baseline capture --id baseline_v1 --model qwen-ft
    
    # Compare to baseline
    python -m evaluation.harness.cli compare --baseline baseline_v1
    
    # Run ablation study
    python -m evaluation.harness.cli ablation --model qwen-ft
    
    # List baselines
    python -m evaluation.harness.cli baseline list
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.harness.config import HarnessConfig, FeatureFlags
from evaluation.harness.runner import EvaluationHarness
from evaluation.harness.baseline import BaselineManager


def cmd_run(args, config: HarnessConfig):
    """Run evaluation."""
    harness = EvaluationHarness(config)
    
    # Parse feature flags
    features = FeatureFlags()
    if args.features:
        for feat in args.features:
            if feat.startswith("no-"):
                setattr(features, feat[3:].replace("-", "_"), False)
            else:
                setattr(features, feat.replace("-", "_"), True)
    
    print(f"Running evaluation: model={args.model}, suite={args.test_suite}")
    print(f"Features: {features.to_dict()}")
    
    results = harness.run_evaluation(
        model_id=args.model,
        test_suite=args.test_suite,
        features=features,
        save_results=True,
    )
    
    print(f"\n=== Results ===")
    print(f"Timestamp: {results.timestamp}")
    print(f"Commit: {results.commit}")
    print(f"\nMetrics:")
    
    metrics = results.metrics.to_dict()
    for dim, stats in metrics.get("dimensions", {}).items():
        print(f"  {dim}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n']})")
    
    if metrics.get("overall"):
        overall = metrics["overall"]
        print(f"\n  Overall: {overall['mean']:.3f} ± {overall['std']:.3f}")
    
    if args.compare_baseline:
        print(f"\n=== Comparing to baseline: {args.compare_baseline} ===")
        report = harness.compare_to_baseline(results, args.compare_baseline)
        print(report.to_markdown())


def cmd_baseline_capture(args, config: HarnessConfig):
    """Capture a new baseline."""
    harness = EvaluationHarness(config)
    
    print(f"Capturing baseline: {args.id}")
    print(f"Model: {args.model}, Suite: {args.test_suite}")
    
    baseline = harness.capture_baseline(
        model_id=args.model,
        baseline_id=args.id,
        test_suite=args.test_suite,
        description=args.description or "",
    )
    
    print(f"\nBaseline captured:")
    print(f"  ID: {baseline.id}")
    print(f"  Commit: {baseline.commit}")
    print(f"  Timestamp: {baseline.timestamp}")
    print(f"  Path: {config.baselines_dir / f'{baseline.id}.json'}")


def cmd_baseline_list(args, config: HarnessConfig):
    """List available baselines."""
    manager = BaselineManager(config)
    baselines = manager.list_baselines()
    
    if not baselines:
        print("No baselines found.")
        return
    
    print("Available baselines:")
    for bid in baselines:
        try:
            b = manager.load(bid)
            print(f"  {bid}: {b.model} @ {b.commit} ({b.timestamp[:10]})")
        except Exception as e:
            print(f"  {bid}: [error loading: {e}]")


def cmd_baseline_show(args, config: HarnessConfig):
    """Show baseline details."""
    manager = BaselineManager(config)
    baseline = manager.load(args.id)
    
    print(f"Baseline: {baseline.id}")
    print(f"  Commit: {baseline.commit}")
    print(f"  Timestamp: {baseline.timestamp}")
    print(f"  Model: {baseline.model}")
    print(f"  Test Suite: {baseline.test_suite}")
    print(f"  Description: {baseline.description or '(none)'}")
    print(f"\nFeatures:")
    for feat, enabled in baseline.features.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feat}")
    print(f"\nMetrics:")
    for dim, stats in baseline.metrics.get("dimensions", {}).items():
        print(f"  {dim}: {stats.get('mean', 'N/A'):.3f}")


def cmd_compare(args, config: HarnessConfig):
    """Compare two baselines."""
    manager = BaselineManager(config)
    
    comparison = manager.compare_baselines(args.baseline1, args.baseline2)
    
    print(f"Comparing {args.baseline1} vs {args.baseline2}")
    print(f"\nDimension Changes:")
    for dim, changes in comparison.get("dimension_changes", {}).items():
        diff = changes["difference"]
        pct = changes["percent_change"]
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {dim}: {changes['baseline_1']:.3f} → {changes['baseline_2']:.3f} ({arrow}{abs(pct):.1f}%)")
    
    if comparison.get("feature_changes"):
        print(f"\nFeature Changes:")
        for feat, changes in comparison["feature_changes"].items():
            print(f"  {feat}: {changes['baseline_1']} → {changes['baseline_2']}")


def cmd_ablation(args, config: HarnessConfig):
    """Run ablation study."""
    harness = EvaluationHarness(config)
    
    print(f"Running ablation study: model={args.model}, suite={args.test_suite}")
    
    report = harness.run_ablation(
        model_id=args.model,
        test_suite=args.test_suite,
    )
    
    # Print markdown report
    print(report.to_markdown())
    
    # Save report
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(report.to_markdown())
        print(f"\nReport saved to: {output_path}")
        
        # Also save JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation harness for Mental Health LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to config file (YAML or JSON)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run evaluation")
    run_parser.add_argument("--model", type=str, default="qwen-ft", help="Model identifier")
    run_parser.add_argument("--test-suite", type=str, default="all", help="Test suite to run")
    run_parser.add_argument("--features", nargs="+", help="Feature flags (use no-X to disable)")
    run_parser.add_argument("--compare-baseline", type=str, help="Compare to this baseline")
    
    # Baseline commands
    baseline_parser = subparsers.add_parser("baseline", help="Baseline management")
    baseline_sub = baseline_parser.add_subparsers(dest="baseline_command")
    
    capture_parser = baseline_sub.add_parser("capture", help="Capture new baseline")
    capture_parser.add_argument("--id", type=str, required=True, help="Baseline identifier")
    capture_parser.add_argument("--model", type=str, default="qwen-ft", help="Model identifier")
    capture_parser.add_argument("--test-suite", type=str, default="all", help="Test suite")
    capture_parser.add_argument("--description", type=str, help="Baseline description")
    
    list_parser = baseline_sub.add_parser("list", help="List baselines")
    
    show_parser = baseline_sub.add_parser("show", help="Show baseline details")
    show_parser.add_argument("id", type=str, help="Baseline identifier")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare baselines")
    compare_parser.add_argument("baseline1", type=str, help="First baseline ID")
    compare_parser.add_argument("baseline2", type=str, help="Second baseline ID")
    
    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument("--model", type=str, default="qwen-ft", help="Model identifier")
    ablation_parser.add_argument("--test-suite", type=str, default="all", help="Test suite")
    ablation_parser.add_argument("--output", type=str, help="Output path for report")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in (".yaml", ".yml"):
            config = HarnessConfig.from_yaml(config_path)
        else:
            config = HarnessConfig.from_json(config_path)
    else:
        config = HarnessConfig()
    
    # Dispatch command
    if args.command == "run":
        cmd_run(args, config)
    elif args.command == "baseline":
        if args.baseline_command == "capture":
            cmd_baseline_capture(args, config)
        elif args.baseline_command == "list":
            cmd_baseline_list(args, config)
        elif args.baseline_command == "show":
            cmd_baseline_show(args, config)
        else:
            baseline_parser.print_help()
    elif args.command == "compare":
        cmd_compare(args, config)
    elif args.command == "ablation":
        cmd_ablation(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
