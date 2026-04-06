"""
Ablation testing framework for evaluation harness.

Runs evaluations with individual features toggled to measure their impact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import HarnessConfig, FeatureFlags
from .metrics import EvalMetrics, MetricsAggregator, StatisticalAnalyzer, ComparisonResult


@dataclass
class AblationResult:
    """Result of a single ablation run."""
    feature: str
    feature_enabled: bool
    metrics: EvalMetrics
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "feature_enabled": self.feature_enabled,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class FeatureImpact:
    """Impact analysis for a single feature."""
    feature: str
    dimension_impacts: dict[str, ComparisonResult]
    overall_impact: ComparisonResult | None
    coherence_impacts: dict[str, ComparisonResult]
    is_beneficial: bool
    summary: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "dimension_impacts": {k: v.to_dict() for k, v in self.dimension_impacts.items()},
            "overall_impact": self.overall_impact.to_dict() if self.overall_impact else None,
            "coherence_impacts": {k: v.to_dict() for k, v in self.coherence_impacts.items()},
            "is_beneficial": self.is_beneficial,
            "summary": self.summary,
        }


@dataclass 
class AblationReport:
    """Complete ablation analysis report."""
    baseline_config: dict[str, bool]
    feature_impacts: list[FeatureImpact]
    interaction_effects: dict[str, float]  # Feature pair -> interaction effect
    recommended_config: dict[str, bool]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_config": self.baseline_config,
            "feature_impacts": [f.to_dict() for f in self.feature_impacts],
            "interaction_effects": self.interaction_effects,
            "recommended_config": self.recommended_config,
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = ["# Ablation Study Report\n"]
        
        # Baseline config
        lines.append("## Baseline Configuration\n")
        for feat, enabled in self.baseline_config.items():
            status = "✓" if enabled else "✗"
            lines.append(f"- {status} {feat}")
        lines.append("")
        
        # Feature impacts table
        lines.append("## Feature Impacts\n")
        lines.append("| Feature | Overall Impact | Significant? | Recommendation |")
        lines.append("|---------|---------------|--------------|----------------|")
        
        for impact in self.feature_impacts:
            if impact.overall_impact:
                oi = impact.overall_impact
                sig = "Yes" if oi.is_significant else "No"
                diff = f"{oi.difference:+.3f}"
                rec = "Keep" if impact.is_beneficial else "Remove"
            else:
                sig = "N/A"
                diff = "N/A"
                rec = "Unknown"
            lines.append(f"| {impact.feature} | {diff} | {sig} | {rec} |")
        
        lines.append("")
        
        # Detailed impacts
        lines.append("## Detailed Analysis\n")
        for impact in self.feature_impacts:
            lines.append(f"### {impact.feature}\n")
            lines.append(f"**Summary**: {impact.summary}\n")
            
            if impact.dimension_impacts:
                lines.append("**Dimension Impacts**:")
                for dim, comp in impact.dimension_impacts.items():
                    lines.append(f"- {dim}: {comp.interpretation}")
            lines.append("")
        
        # Recommended configuration
        lines.append("## Recommended Configuration\n")
        for feat, enabled in self.recommended_config.items():
            status = "Enable" if enabled else "Disable"
            lines.append(f"- {feat}: {status}")
        
        return "\n".join(lines)


class AblationRunner:
    """Run ablation studies to measure feature impact."""
    
    def __init__(
        self,
        config: HarnessConfig,
        eval_fn: Callable[[FeatureFlags], list[dict]] | None = None,
    ):
        """
        Initialize ablation runner.
        
        Args:
            config: Harness configuration
            eval_fn: Function that takes FeatureFlags and returns evaluation results.
                     If None, a placeholder is used.
        """
        self.config = config
        self.eval_fn = eval_fn or self._placeholder_eval
        self.aggregator = MetricsAggregator(
            bootstrap_samples=config.bootstrap_samples,
            confidence_level=config.confidence_level,
        )
        self.analyzer = StatisticalAnalyzer(alpha=0.05)
    
    def run_full_ablation(self, baseline_features: FeatureFlags | None = None) -> AblationReport:
        """
        Run complete ablation study.
        
        For each feature, runs evaluation with that feature disabled while
        all others remain enabled, then compares to baseline.
        """
        baseline_features = baseline_features or FeatureFlags.all_on()
        
        # Run baseline evaluation
        baseline_results = self.eval_fn(baseline_features)
        baseline_metrics = self.aggregator.aggregate(baseline_results)
        baseline_scores = self._extract_scores(baseline_results)
        
        feature_impacts = []
        
        # Run ablation for each feature
        for feature in self._get_feature_names():
            # Create config with this feature disabled
            ablation_flags = FeatureFlags.from_dict(baseline_features.to_dict())
            setattr(ablation_flags, feature, False)
            
            # Run evaluation
            ablation_results = self.eval_fn(ablation_flags)
            ablation_metrics = self.aggregator.aggregate(ablation_results)
            ablation_scores = self._extract_scores(ablation_results)
            
            # Compare (note: we're comparing baseline WITH feature vs WITHOUT)
            # So positive difference means feature helps
            comparisons = self.analyzer.compare(ablation_scores, baseline_scores)
            
            # Organize comparisons
            dim_impacts = {}
            coh_impacts = {}
            overall_impact = None
            
            for comp in comparisons:
                if comp.dimension == "overall":
                    overall_impact = comp
                elif comp.dimension in ["memory", "therapeutic_arc", "repetition_avoidance"]:
                    coh_impacts[comp.dimension] = comp
                else:
                    dim_impacts[comp.dimension] = comp
            
            # Determine if beneficial
            is_beneficial = True
            if overall_impact:
                is_beneficial = overall_impact.difference > 0 or not overall_impact.is_significant
            
            # Generate summary
            summary = self._generate_impact_summary(feature, dim_impacts, overall_impact)
            
            feature_impacts.append(FeatureImpact(
                feature=feature,
                dimension_impacts=dim_impacts,
                overall_impact=overall_impact,
                coherence_impacts=coh_impacts,
                is_beneficial=is_beneficial,
                summary=summary,
            ))
        
        # Compute interaction effects (simplified: pairwise feature interactions)
        interaction_effects = self._compute_interactions(baseline_features)
        
        # Recommend configuration
        recommended = self._recommend_config(feature_impacts, baseline_features)
        
        return AblationReport(
            baseline_config=baseline_features.to_dict(),
            feature_impacts=feature_impacts,
            interaction_effects=interaction_effects,
            recommended_config=recommended,
        )
    
    def run_single_feature_ablation(
        self,
        feature: str,
        baseline_features: FeatureFlags | None = None,
    ) -> FeatureImpact:
        """Run ablation for a single feature."""
        baseline_features = baseline_features or FeatureFlags.all_on()
        
        # Run baseline
        baseline_results = self.eval_fn(baseline_features)
        baseline_scores = self._extract_scores(baseline_results)
        
        # Run with feature disabled
        ablation_flags = FeatureFlags.from_dict(baseline_features.to_dict())
        setattr(ablation_flags, feature, False)
        ablation_results = self.eval_fn(ablation_flags)
        ablation_scores = self._extract_scores(ablation_results)
        
        # Compare
        comparisons = self.analyzer.compare(ablation_scores, baseline_scores)
        
        dim_impacts = {}
        coh_impacts = {}
        overall_impact = None
        
        for comp in comparisons:
            if comp.dimension == "overall":
                overall_impact = comp
            elif comp.dimension in ["memory", "therapeutic_arc", "repetition_avoidance"]:
                coh_impacts[comp.dimension] = comp
            else:
                dim_impacts[comp.dimension] = comp
        
        is_beneficial = True
        if overall_impact:
            is_beneficial = overall_impact.difference > 0 or not overall_impact.is_significant
        
        summary = self._generate_impact_summary(feature, dim_impacts, overall_impact)
        
        return FeatureImpact(
            feature=feature,
            dimension_impacts=dim_impacts,
            overall_impact=overall_impact,
            coherence_impacts=coh_impacts,
            is_beneficial=is_beneficial,
            summary=summary,
        )
    
    def _get_feature_names(self) -> list[str]:
        """Get list of feature names from FeatureFlags."""
        return list(FeatureFlags().to_dict().keys())
    
    def _extract_scores(self, results: list[dict]) -> dict[str, list[float]]:
        """Extract dimension scores from results."""
        scores: dict[str, list[float]] = {}
        
        for result in results:
            for turn in result.get("turns", []):
                turn_scores = turn.get("scores", {})
                for dim, data in turn_scores.items():
                    score = data.get("score") if isinstance(data, dict) else data
                    if isinstance(score, (int, float)):
                        if dim not in scores:
                            scores[dim] = []
                        scores[dim].append(float(score))
            
            # Add coherence scores
            coherence = result.get("coherence_scores", {})
            for dim, data in coherence.items():
                score = data.get("score") if isinstance(data, dict) else data
                if isinstance(score, (int, float)):
                    if dim not in scores:
                        scores[dim] = []
                    scores[dim].append(float(score))
        
        # Compute overall
        all_scores = []
        for dim_scores in scores.values():
            all_scores.extend(dim_scores)
        if all_scores:
            scores["overall"] = all_scores
        
        return scores
    
    def _generate_impact_summary(
        self,
        feature: str,
        dim_impacts: dict[str, ComparisonResult],
        overall_impact: ComparisonResult | None,
    ) -> str:
        """Generate human-readable impact summary."""
        if overall_impact and overall_impact.is_significant:
            direction = "improves" if overall_impact.difference > 0 else "hurts"
            effect = abs(overall_impact.effect_size)
            if effect < 0.2:
                magnitude = "slightly"
            elif effect < 0.5:
                magnitude = "moderately"
            else:
                magnitude = "significantly"
            
            return f"{feature} {magnitude} {direction} overall performance (Δ={overall_impact.difference:+.3f}, p={overall_impact.p_value:.3f})"
        
        # Check individual dimensions
        sig_improvements = []
        sig_declines = []
        
        for dim, comp in dim_impacts.items():
            if comp.is_significant:
                if comp.difference > 0:
                    sig_improvements.append(dim)
                else:
                    sig_declines.append(dim)
        
        if sig_improvements and not sig_declines:
            return f"{feature} improves {', '.join(sig_improvements)}"
        elif sig_declines and not sig_improvements:
            return f"{feature} hurts {', '.join(sig_declines)}"
        elif sig_improvements and sig_declines:
            return f"{feature} has mixed effects: improves {', '.join(sig_improvements)}, hurts {', '.join(sig_declines)}"
        else:
            return f"{feature} has no significant impact on any dimension"
    
    def _compute_interactions(self, baseline: FeatureFlags) -> dict[str, float]:
        """Compute pairwise feature interaction effects (simplified)."""
        # This is a placeholder - full interaction analysis would require
        # running 2^n evaluations for n features
        return {}
    
    def _recommend_config(
        self,
        impacts: list[FeatureImpact],
        baseline: FeatureFlags,
    ) -> dict[str, bool]:
        """Recommend optimal feature configuration."""
        recommended = baseline.to_dict().copy()
        
        for impact in impacts:
            # Only recommend disabling if clearly harmful
            if not impact.is_beneficial:
                if impact.overall_impact and impact.overall_impact.is_significant:
                    if impact.overall_impact.effect_size < -0.2:  # Meaningful negative effect
                        recommended[impact.feature] = False
        
        return recommended
    
    def _placeholder_eval(self, features: FeatureFlags) -> list[dict]:
        """Placeholder evaluation function for testing."""
        import random
        random.seed(42)
        
        results = []
        for i in range(10):
            turns = []
            for j in range(3):
                scores = {}
                for dim in ["empathy", "cbt_techniques", "guided_discovery", "safety_awareness", "clinical_appropriateness"]:
                    base_score = 1.0 + random.random()
                    # Features add small bonuses
                    if features.compaction:
                        base_score += 0.05
                    if features.response_guard:
                        base_score += 0.03
                    scores[dim] = {"score": min(2.0, base_score)}
                turns.append({"scores": scores})
            
            coherence = {
                "memory": {"score": 1.0 + random.random()},
                "therapeutic_arc": {"score": 1.0 + random.random()},
                "repetition_avoidance": {"score": 1.0 + random.random()},
            }
            
            results.append({
                "case_id": f"case_{i}",
                "turns": turns,
                "coherence_scores": coherence,
            })
        
        return results
