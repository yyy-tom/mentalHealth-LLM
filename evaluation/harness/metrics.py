"""
Metrics aggregation and statistical analysis for evaluation harness.

Implements:
- Dimension-wise score aggregation
- Bootstrap confidence intervals
- Wilcoxon signed-rank test for paired comparisons
- Krippendorff's alpha for inter-rater reliability
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# Scoring dimensions (matching existing rubric)
DIMENSIONS = [
    "empathy",
    "cbt_techniques",
    "guided_discovery",
    "safety_awareness",
    "clinical_appropriateness",
]

COHERENCE_DIMENSIONS = [
    "memory",
    "therapeutic_arc",
    "repetition_avoidance",
]


@dataclass
class DimensionStats:
    """Statistics for a single evaluation dimension."""
    mean: float
    std: float
    median: float
    n: int
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 3),
            "std": round(self.std, 3),
            "median": round(self.median, 3),
            "n": self.n,
            "ci_lower": round(self.ci_lower, 3),
            "ci_upper": round(self.ci_upper, 3),
        }


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    dimensions: dict[str, DimensionStats] = field(default_factory=dict)
    coherence: dict[str, DimensionStats] = field(default_factory=dict)
    overall: DimensionStats | None = None
    by_risk_level: dict[str, dict[str, DimensionStats]] = field(default_factory=dict)
    by_skill: dict[str, dict[str, DimensionStats]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        result = {
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "coherence": {k: v.to_dict() for k, v in self.coherence.items()},
        }
        if self.overall:
            result["overall"] = self.overall.to_dict()
        if self.by_risk_level:
            result["by_risk_level"] = {
                level: {k: v.to_dict() for k, v in dims.items()}
                for level, dims in self.by_risk_level.items()
            }
        if self.by_skill:
            result["by_skill"] = {
                skill: {k: v.to_dict() for k, v in dims.items()}
                for skill, dims in self.by_skill.items()
            }
        return result


class MetricsAggregator:
    """Aggregate evaluation scores into metrics."""
    
    def __init__(self, bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
    
    def aggregate(self, results: list[dict]) -> EvalMetrics:
        """
        Aggregate evaluation results into metrics.
        
        Args:
            results: List of case results, each containing:
                - turns: list of turn results with scores
                - coherence_scores: dict of coherence dimension scores
                - risk_level: optional risk level (HIGH/MEDIUM/LOW)
                - skill: optional skill category
        """
        metrics = EvalMetrics()
        
        # Collect all dimension scores
        dim_scores: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
        coh_scores: dict[str, list[float]] = {d: [] for d in COHERENCE_DIMENSIONS}
        risk_scores: dict[str, dict[str, list[float]]] = {}
        skill_scores: dict[str, dict[str, list[float]]] = {}
        
        for result in results:
            # Extract turn-level dimension scores
            for turn in result.get("turns", []):
                scores = turn.get("scores", {})
                for dim in DIMENSIONS:
                    dim_data = scores.get(dim, {})
                    score = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                    if isinstance(score, (int, float)):
                        dim_scores[dim].append(float(score))
            
            # Extract coherence scores
            coherence = result.get("coherence_scores", {})
            for dim in COHERENCE_DIMENSIONS:
                dim_data = coherence.get(dim, {})
                score = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                if isinstance(score, (int, float)):
                    coh_scores[dim].append(float(score))
            
            # Group by risk level
            risk_level = result.get("risk_level", "UNKNOWN")
            if risk_level not in risk_scores:
                risk_scores[risk_level] = {d: [] for d in DIMENSIONS}
            for turn in result.get("turns", []):
                scores = turn.get("scores", {})
                for dim in DIMENSIONS:
                    dim_data = scores.get(dim, {})
                    score = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                    if isinstance(score, (int, float)):
                        risk_scores[risk_level][dim].append(float(score))
            
            # Group by skill
            skill = result.get("skill", "UNKNOWN")
            if skill not in skill_scores:
                skill_scores[skill] = {d: [] for d in DIMENSIONS}
            for turn in result.get("turns", []):
                scores = turn.get("scores", {})
                for dim in DIMENSIONS:
                    dim_data = scores.get(dim, {})
                    score = dim_data.get("score") if isinstance(dim_data, dict) else dim_data
                    if isinstance(score, (int, float)):
                        skill_scores[skill][dim].append(float(score))
        
        # Compute dimension statistics
        for dim in DIMENSIONS:
            if dim_scores[dim]:
                metrics.dimensions[dim] = self._compute_stats(dim_scores[dim])
        
        # Compute coherence statistics
        for dim in COHERENCE_DIMENSIONS:
            if coh_scores[dim]:
                metrics.coherence[dim] = self._compute_stats(coh_scores[dim])
        
        # Compute overall
        all_scores = []
        for dim in DIMENSIONS:
            all_scores.extend(dim_scores[dim])
        if all_scores:
            metrics.overall = self._compute_stats(all_scores)
        
        # Compute by risk level
        for level, level_scores in risk_scores.items():
            metrics.by_risk_level[level] = {}
            for dim in DIMENSIONS:
                if level_scores[dim]:
                    metrics.by_risk_level[level][dim] = self._compute_stats(level_scores[dim])
        
        # Compute by skill
        for skill, skill_dim_scores in skill_scores.items():
            metrics.by_skill[skill] = {}
            for dim in DIMENSIONS:
                if skill_dim_scores[dim]:
                    metrics.by_skill[skill][dim] = self._compute_stats(skill_dim_scores[dim])
        
        return metrics
    
    def _compute_stats(self, scores: list[float]) -> DimensionStats:
        """Compute statistics for a list of scores with bootstrap CI."""
        arr = np.array(scores)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        median = float(np.median(arr))
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(arr)
        
        return DimensionStats(
            mean=mean,
            std=std,
            median=median,
            n=n,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
    
    def _bootstrap_ci(self, arr: np.ndarray) -> tuple[float, float]:
        """Compute bootstrap confidence interval for the mean."""
        if len(arr) < 2:
            val = float(arr[0]) if len(arr) == 1 else 0.0
            return (val, val)
        
        rng = np.random.default_rng(42)
        bootstrap_means = []
        
        for _ in range(self.bootstrap_samples):
            sample = rng.choice(arr, size=len(arr), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - self.confidence_level
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
        
        return lower, upper


@dataclass
class ComparisonResult:
    """Result of comparing two evaluation runs."""
    dimension: str
    baseline_mean: float
    current_mean: float
    difference: float
    percent_change: float
    raw_p_value: float
    p_value: float
    is_significant: bool
    effect_size: float  # Cohen's d
    interpretation: str
    p_value_method: str = "none"
    
    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "baseline_mean": round(self.baseline_mean, 3),
            "current_mean": round(self.current_mean, 3),
            "difference": round(self.difference, 3),
            "percent_change": round(self.percent_change, 2),
            "raw_p_value": round(self.raw_p_value, 4),
            "p_value": round(self.p_value, 4),
            "is_significant": self.is_significant,
            "effect_size": round(self.effect_size, 3),
            "interpretation": self.interpretation,
            "p_value_method": self.p_value_method,
        }


class StatisticalAnalyzer:
    """Statistical analysis for evaluation comparisons."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        min_sample_size: int = 5,
        multiple_comparison: str = "none",
    ):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.multiple_comparison = multiple_comparison
    
    def compare(
        self,
        baseline_scores: dict[str, list[float]],
        current_scores: dict[str, list[float]],
    ) -> list[ComparisonResult]:
        """
        Compare baseline and current evaluation scores.
        
        Uses Wilcoxon signed-rank test for paired comparisons when paired data
        is available, otherwise uses Mann-Whitney U test.
        """
        candidates = []
        
        for dim in set(baseline_scores.keys()) | set(current_scores.keys()):
            baseline = np.array(baseline_scores.get(dim, []))
            current = np.array(current_scores.get(dim, []))
            
            if len(baseline) < self.min_sample_size or len(current) < self.min_sample_size:
                continue  # Skip dimensions with insufficient data
            
            baseline_mean = float(np.mean(baseline))
            current_mean = float(np.mean(current))
            difference = current_mean - baseline_mean
            percent_change = (difference / baseline_mean * 100) if baseline_mean != 0 else 0.0
            
            # Compute p-value using appropriate test
            if len(baseline) == len(current):
                # Paired: Wilcoxon signed-rank test
                p_value = self._wilcoxon_test(baseline, current)
            else:
                # Unpaired: Mann-Whitney U test
                p_value = self._mannwhitney_test(baseline, current)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(current, ddof=1)) / 2)
            effect_size = difference / pooled_std if pooled_std > 0 else 0.0
            candidates.append({
                "dimension": dim,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "difference": difference,
                "percent_change": percent_change,
                "raw_p_value": p_value,
                "effect_size": effect_size,
            })

        if not candidates:
            return []

        raw_p_values = [c["raw_p_value"] for c in candidates]
        corrected_p_values = self._adjust_p_values(raw_p_values, self.multiple_comparison)

        results = []
        for candidate, corrected_p in zip(candidates, corrected_p_values):
            is_significant = corrected_p < self.alpha
            interpretation = self._interpret_effect(
                candidate["effect_size"], is_significant, candidate["difference"]
            )
            results.append(ComparisonResult(
                dimension=candidate["dimension"],
                baseline_mean=candidate["baseline_mean"],
                current_mean=candidate["current_mean"],
                difference=candidate["difference"],
                percent_change=candidate["percent_change"],
                raw_p_value=candidate["raw_p_value"],
                p_value=corrected_p,
                is_significant=is_significant,
                effect_size=candidate["effect_size"],
                interpretation=interpretation,
                p_value_method=self.multiple_comparison,
            ))

        return results

    def _adjust_p_values(self, p_values: list[float], method: str) -> list[float]:
        """Apply multiple-comparison correction to p-values."""
        if not p_values:
            return []

        if method == "none":
            return p_values

        m = len(p_values)

        if method == "bonferroni":
            return [min(p * m, 1.0) for p in p_values]

        if method == "fdr_bh":
            indexed = list(enumerate(p_values))
            indexed.sort(key=lambda item: item[1])  # Ascending by raw p-value

            adjusted = [0.0] * m
            for rank, (idx, p) in enumerate(indexed, start=1):
                adjusted[idx] = min((p * m) / rank, 1.0)

            # Enforce monotonicity in ranked order
            ranked_adj = [adjusted[idx] for idx, _ in indexed]
            for i in range(m - 2, -1, -1):
                ranked_adj[i] = min(ranked_adj[i], ranked_adj[i + 1])

            for (idx, _), adj in zip(indexed, ranked_adj):
                adjusted[idx] = adj
            return adjusted

        raise ValueError(
            "multiple_comparison must be one of: none, bonferroni, fdr_bh"
        )
    
    def _wilcoxon_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Wilcoxon signed-rank test for paired samples."""
        try:
            from scipy import stats
            # Remove pairs where difference is zero
            diff = y - x
            diff = diff[diff != 0]
            if len(diff) < 5:
                return 1.0  # Not enough data
            stat, p = stats.wilcoxon(diff)
            return float(p)
        except ImportError:
            # Fallback: simple permutation test
            return self._permutation_test(x, y)
    
    def _mannwhitney_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mann-Whitney U test for unpaired samples."""
        try:
            from scipy import stats
            stat, p = stats.mannwhitneyu(x, y, alternative='two-sided')
            return float(p)
        except ImportError:
            return self._permutation_test(x, y)
    
    def _permutation_test(self, x: np.ndarray, y: np.ndarray, n_permutations: int = 1000) -> float:
        """Simple permutation test fallback when scipy is not available."""
        observed_diff = np.mean(y) - np.mean(x)
        combined = np.concatenate([x, y])
        n_x = len(x)
        
        rng = np.random.default_rng(42)
        count_extreme = 0
        
        for _ in range(n_permutations):
            rng.shuffle(combined)
            perm_x = combined[:n_x]
            perm_y = combined[n_x:]
            perm_diff = np.mean(perm_y) - np.mean(perm_x)
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1
        
        return count_extreme / n_permutations
    
    def _interpret_effect(self, effect_size: float, is_significant: bool, difference: float) -> str:
        """Interpret the effect size and significance."""
        direction = "improved" if difference > 0 else "declined"
        
        if not is_significant:
            return f"No significant change (p ≥ {self.alpha})"
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"Significantly {direction} with {magnitude} effect (d={effect_size:.2f})"
    
    def krippendorff_alpha(
        self,
        ratings: list[list[float | None]],
        level: str = "interval",
    ) -> float:
        """
        Compute Krippendorff's alpha for inter-rater reliability.
        
        Args:
            ratings: List of ratings, where each inner list is one rater's scores
                     (None for missing ratings)
            level: Measurement level ("nominal", "ordinal", "interval", "ratio")
        
        Returns:
            Krippendorff's alpha coefficient
        """
        # Convert to numpy array with NaN for missing
        n_raters = len(ratings)
        n_items = max(len(r) for r in ratings)
        
        matrix = np.full((n_raters, n_items), np.nan)
        for i, rater_scores in enumerate(ratings):
            for j, score in enumerate(rater_scores):
                if score is not None:
                    matrix[i, j] = score
        
        # Compute observed disagreement
        n_c = np.sum(~np.isnan(matrix), axis=0)  # Coders per unit
        valid_units = n_c >= 2
        
        if not np.any(valid_units):
            return 0.0
        
        observed_disagreement = 0.0
        expected_disagreement = 0.0
        total_pairs = 0
        all_values = []
        
        for j in np.where(valid_units)[0]:
            values = matrix[:, j][~np.isnan(matrix[:, j])]
            all_values.extend(values)
            n = len(values)
            
            # Observed disagreement for this unit
            for i in range(n):
                for k in range(i + 1, n):
                    diff = self._distance(values[i], values[k], level)
                    observed_disagreement += diff
                    total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        observed_disagreement /= total_pairs
        
        # Expected disagreement (from marginal distribution)
        all_values = np.array(all_values)
        n_total = len(all_values)
        
        for i in range(n_total):
            for j in range(i + 1, n_total):
                diff = self._distance(all_values[i], all_values[j], level)
                expected_disagreement += diff
        
        expected_disagreement /= (n_total * (n_total - 1) / 2)
        
        if expected_disagreement == 0:
            return 1.0
        
        alpha = 1 - (observed_disagreement / expected_disagreement)
        return float(alpha)
    
    def _distance(self, a: float, b: float, level: str) -> float:
        """Compute distance metric based on measurement level."""
        if level == "nominal":
            return 0.0 if a == b else 1.0
        elif level == "ordinal":
            # Use rank distance
            return abs(a - b)
        elif level in ("interval", "ratio"):
            return (a - b) ** 2
        else:
            return (a - b) ** 2
