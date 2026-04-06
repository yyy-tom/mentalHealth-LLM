"""
Tests for evaluation harness.
"""
import json
import tempfile
from pathlib import Path

import pytest

from evaluation.harness.config import HarnessConfig, FeatureFlags, JudgeConfig
from evaluation.harness.metrics import (
    MetricsAggregator,
    StatisticalAnalyzer,
    DimensionStats,
    EvalMetrics,
)
from evaluation.harness.baseline import BaselineManager, Baseline
from evaluation.harness.ablation import AblationRunner, FeatureImpact
from evaluation.harness.runner import EvaluationHarness


class TestFeatureFlags:
    """Tests for FeatureFlags configuration."""
    
    def test_default_all_on(self):
        flags = FeatureFlags()
        for key, val in flags.to_dict().items():
            assert val is True, f"{key} should default to True"
    
    def test_all_off(self):
        flags = FeatureFlags.all_off()
        for key, val in flags.to_dict().items():
            assert val is False, f"{key} should be False"
    
    def test_with_only(self):
        flags = FeatureFlags.all_off().with_only("compaction")
        assert flags.compaction is True
        assert flags.response_guard is False
        assert flags.dynamic_prompts is False
    
    def test_from_dict(self):
        data = {"compaction": True, "response_guard": False}
        flags = FeatureFlags.from_dict(data)
        assert flags.compaction is True
        assert flags.response_guard is False


class TestHarnessConfig:
    """Tests for HarnessConfig."""
    
    def test_default_config(self):
        config = HarnessConfig()
        assert config.seed == 42
        assert config.bootstrap_samples == 1000
        assert config.confidence_level == 0.95
        assert config.multiple_comparison_correction == "none"
        assert config.allow_placeholder_fallback is True

    def test_invalid_multiple_comparison_setting(self):
        with pytest.raises(ValueError):
            HarnessConfig(multiple_comparison_correction="invalid")
    
    def test_config_paths_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
                results_dir=Path(tmpdir) / "results",
            )
            assert config.baselines_dir.exists()
            assert config.results_dir.exists()
    
    def test_save_and_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(project_root=Path(tmpdir))
            config_path = Path(tmpdir) / "config.json"
            config.save_json(config_path)
            
            loaded = HarnessConfig.from_json(config_path)
            assert loaded.seed == config.seed
            assert loaded.bootstrap_samples == config.bootstrap_samples


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""
    
    def test_aggregate_basic(self):
        aggregator = MetricsAggregator(bootstrap_samples=100)
        
        results = [
            {
                "turns": [
                    {"scores": {"empathy": {"score": 1.5}, "cbt_techniques": {"score": 1.2}}},
                    {"scores": {"empathy": {"score": 1.8}, "cbt_techniques": {"score": 1.4}}},
                ],
                "coherence_scores": {"memory": {"score": 1.3}},
            },
            {
                "turns": [
                    {"scores": {"empathy": {"score": 1.6}, "cbt_techniques": {"score": 1.3}}},
                ],
                "coherence_scores": {"memory": {"score": 1.5}},
            },
        ]
        
        metrics = aggregator.aggregate(results)
        
        assert "empathy" in metrics.dimensions
        assert metrics.dimensions["empathy"].n == 3
        assert 1.5 <= metrics.dimensions["empathy"].mean <= 1.7
        
        assert "memory" in metrics.coherence
        assert metrics.coherence["memory"].n == 2
    
    def test_aggregate_with_risk_levels(self):
        aggregator = MetricsAggregator(bootstrap_samples=100)
        
        results = [
            {
                "risk_level": "HIGH",
                "turns": [{"scores": {"safety_awareness": {"score": 1.9}}}],
                "coherence_scores": {},
            },
            {
                "risk_level": "LOW",
                "turns": [{"scores": {"safety_awareness": {"score": 1.2}}}],
                "coherence_scores": {},
            },
        ]
        
        metrics = aggregator.aggregate(results)
        
        assert "HIGH" in metrics.by_risk_level
        assert "LOW" in metrics.by_risk_level
        assert metrics.by_risk_level["HIGH"]["safety_awareness"].mean > metrics.by_risk_level["LOW"]["safety_awareness"].mean
    
    def test_bootstrap_ci(self):
        aggregator = MetricsAggregator(bootstrap_samples=500, confidence_level=0.95)
        
        import numpy as np
        np.random.seed(42)
        scores = list(np.random.normal(1.5, 0.3, 50))
        
        stats = aggregator._compute_stats(scores)
        
        assert stats.ci_lower < stats.mean < stats.ci_upper
        assert stats.ci_lower > 1.0  # Mean - 2*std should still be > 1
        assert stats.ci_upper < 2.0  # Mean + 2*std should still be < 2


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer."""
    
    def test_compare_significant_difference(self):
        analyzer = StatisticalAnalyzer(alpha=0.05)
        
        import numpy as np
        np.random.seed(42)
        baseline = {"empathy": list(np.random.normal(1.0, 0.2, 30))}
        current = {"empathy": list(np.random.normal(1.5, 0.2, 30))}
        
        results = analyzer.compare(baseline, current)
        
        assert len(results) == 1
        assert results[0].dimension == "empathy"
        assert results[0].difference > 0
        assert results[0].is_significant  # Large difference should be significant
    
    def test_compare_no_significant_difference(self):
        analyzer = StatisticalAnalyzer(alpha=0.05)
        
        import numpy as np
        np.random.seed(42)
        baseline = {"empathy": list(np.random.normal(1.5, 0.3, 30))}
        current = {"empathy": list(np.random.normal(1.52, 0.3, 30))}
        
        results = analyzer.compare(baseline, current)
        
        assert len(results) == 1
        # Very small difference may or may not be significant
        assert abs(results[0].difference) < 0.1

    def test_compare_respects_min_sample_size(self):
        analyzer = StatisticalAnalyzer(alpha=0.05, min_sample_size=30)
        baseline = {"empathy": [1.0] * 10}
        current = {"empathy": [1.5] * 10}
        results = analyzer.compare(baseline, current)
        assert results == []

    def test_compare_applies_bonferroni_correction(self):
        analyzer = StatisticalAnalyzer(alpha=0.05, multiple_comparison="bonferroni")

        import numpy as np
        np.random.seed(42)
        baseline = {
            "empathy": list(np.random.normal(1.0, 0.1, 40)),
            "guided_discovery": list(np.random.normal(1.1, 0.1, 40)),
        }
        current = {
            "empathy": list(np.random.normal(1.5, 0.1, 40)),
            "guided_discovery": list(np.random.normal(1.4, 0.1, 40)),
        }

        results = analyzer.compare(baseline, current)
        assert len(results) == 2
        assert all(r.p_value >= r.raw_p_value for r in results)
        assert all(r.p_value_method == "bonferroni" for r in results)
    
    def test_effect_size_interpretation(self):
        analyzer = StatisticalAnalyzer()
        
        # Small effect
        interp = analyzer._interpret_effect(0.3, True, 0.1)
        assert "small" in interp.lower()
        
        # Large effect
        interp = analyzer._interpret_effect(1.0, True, 0.5)
        assert "large" in interp.lower()
        
        # Not significant
        interp = analyzer._interpret_effect(0.5, False, 0.2)
        assert "no significant" in interp.lower()


class TestBaselineManager:
    """Tests for BaselineManager."""
    
    def test_capture_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
            )
            manager = BaselineManager(config)
            aggregator = MetricsAggregator(bootstrap_samples=100)
            
            # Create mock metrics
            results = [
                {"turns": [{"scores": {"empathy": {"score": 1.5}}}], "coherence_scores": {}},
            ]
            metrics = aggregator.aggregate(results)
            
            # Capture baseline
            baseline = manager.capture(
                baseline_id="test_baseline",
                model="test-model",
                test_suite="all",
                metrics=metrics,
                description="Test baseline",
            )
            
            assert baseline.id == "test_baseline"
            assert baseline.model == "test-model"
            
            # Load baseline
            loaded = manager.load("test_baseline")
            assert loaded.id == baseline.id
            assert loaded.model == baseline.model
    
    def test_list_baselines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
            )
            manager = BaselineManager(config)
            aggregator = MetricsAggregator(bootstrap_samples=100)
            
            metrics = aggregator.aggregate([
                {"turns": [{"scores": {"empathy": {"score": 1.5}}}], "coherence_scores": {}},
            ])
            
            manager.capture("baseline_1", "model", "all", metrics)
            manager.capture("baseline_2", "model", "all", metrics)
            
            baselines = manager.list_baselines()
            assert "baseline_1" in baselines
            assert "baseline_2" in baselines
    
    def test_delete_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
            )
            manager = BaselineManager(config)
            aggregator = MetricsAggregator(bootstrap_samples=100)
            
            metrics = aggregator.aggregate([
                {"turns": [{"scores": {"empathy": {"score": 1.5}}}], "coherence_scores": {}},
            ])
            
            manager.capture("to_delete", "model", "all", metrics)
            assert "to_delete" in manager.list_baselines()
            
            manager.delete("to_delete")
            assert "to_delete" not in manager.list_baselines()


class TestAblationRunner:
    """Tests for AblationRunner."""
    
    def test_placeholder_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(project_root=Path(tmpdir))
            runner = AblationRunner(config)
            
            flags = FeatureFlags()
            results = runner._placeholder_eval(flags)
            
            assert len(results) == 10
            for result in results:
                assert "turns" in result
                assert "coherence_scores" in result
    
    def test_full_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(project_root=Path(tmpdir))
            runner = AblationRunner(config)
            
            report = runner.run_full_ablation()
            
            assert report.baseline_config
            assert len(report.feature_impacts) > 0
            
            # Check each feature has an impact analysis
            feature_names = list(FeatureFlags().to_dict().keys())
            for impact in report.feature_impacts:
                assert impact.feature in feature_names
                assert impact.summary
    
    def test_single_feature_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(project_root=Path(tmpdir))
            runner = AblationRunner(config)
            
            impact = runner.run_single_feature_ablation("compaction")
            
            assert impact.feature == "compaction"
            assert impact.summary


class TestEvaluationHarness:
    """Tests for main EvaluationHarness."""
    
    def test_placeholder_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
                results_dir=Path(tmpdir) / "results",
            )
            
            # Create a mock cases.json
            cases_dir = Path(tmpdir) / "evaluation"
            cases_dir.mkdir(parents=True)
            cases_file = cases_dir / "cases.json"
            with open(cases_file, "w") as f:
                json.dump({
                    "cases": [
                        {"case_id": "test1", "title": "Test", "user_turns": ["Hello"]},
                    ]
                }, f)
            
            harness = EvaluationHarness(config)
            results = harness.run_evaluation(
                model_id="qwen-ft",
                test_suite="all",
                save_results=False,
            )
            
            assert results.model == "qwen-ft"
            assert results.metrics is not None

    def test_strict_mode_disables_placeholder_fallback(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
                results_dir=Path(tmpdir) / "results",
                allow_placeholder_fallback=False,
            )

            cases_dir = Path(tmpdir) / "evaluation"
            cases_dir.mkdir(parents=True)
            with open(cases_dir / "cases.json", "w") as f:
                json.dump({"cases": [{"case_id": "test1", "user_turns": ["Hello"]}]}, f)

            real_import = __import__

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "scripts.evaluation.generate_responses":
                    raise ImportError("forced import error")
                return real_import(name, globals, locals, fromlist, level)

            monkeypatch.setattr("builtins.__import__", fake_import)

            harness = EvaluationHarness(config)
            with pytest.raises(RuntimeError, match="Real evaluation failed at import"):
                harness.run_evaluation("qwen-ft", save_results=False)

    def test_placeholder_mode_reports_reason(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
                results_dir=Path(tmpdir) / "results",
                allow_placeholder_fallback=True,
            )

            cases_dir = Path(tmpdir) / "evaluation"
            cases_dir.mkdir(parents=True)
            with open(cases_dir / "cases.json", "w") as f:
                json.dump({"cases": [{"case_id": "test1", "user_turns": ["Hello"]}]}, f)

            real_import = __import__

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "scripts.evaluation.generate_responses":
                    raise ImportError("forced import error")
                return real_import(name, globals, locals, fromlist, level)

            monkeypatch.setattr("builtins.__import__", fake_import)

            harness = EvaluationHarness(config)
            results = harness.run_evaluation("qwen-ft", save_results=False)
            assert results.evaluation_mode == "placeholder"
            assert "forced import error" in (results.placeholder_reason or "")
            assert all(r.get("evaluation_mode") == "placeholder" for r in results.raw_results)
    
    def test_baseline_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HarnessConfig(
                project_root=Path(tmpdir),
                baselines_dir=Path(tmpdir) / "baselines",
                results_dir=Path(tmpdir) / "results",
            )
            
            # Create mock cases
            cases_dir = Path(tmpdir) / "evaluation"
            cases_dir.mkdir(parents=True)
            with open(cases_dir / "cases.json", "w") as f:
                json.dump({
                    "cases": [{"case_id": "t1", "user_turns": ["Hi"]}]
                }, f)
            
            harness = EvaluationHarness(config)
            
            # Capture baseline
            baseline = harness.capture_baseline(
                model_id="qwen-ft",
                baseline_id="test_baseline",
                description="Test",
            )
            
            assert baseline.id == "test_baseline"
            
            # Run new evaluation and compare
            results = harness.run_evaluation("qwen-ft", save_results=False)
            report = harness.compare_to_baseline(results, "test_baseline")
            
            assert report.baseline_id == "test_baseline"
            assert report.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
