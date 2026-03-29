"""
Keyword/regex-based skill router.

CPU-only classifier that routes user messages to skill-specific LoRA adapters
using keyword substring matching and regex patterns. This is the original
routing logic extracted from skill_router.py.
"""

import json
import re
from pathlib import Path
from typing import Optional

from mental_health_llm.router_base import BaseRouter


class KeywordRouter(BaseRouter):
    """Routes user messages to skill-specific LoRA adapters using keyword/regex matching."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the router from a skills config JSON file.

        Args:
            config_path: Path to skills_config.json. Defaults to the one
                         bundled with the mental_health_llm package.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "skills_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.default_skill = config.get("default_skill", "general-support")
        self.confidence_threshold = config.get("confidence_threshold", 0.3)

        # Parse skills and sort by priority (highest first)
        self.skills = []
        for skill_def in config["skills"]:
            compiled_patterns = []
            for p in skill_def.get("patterns", []):
                try:
                    compiled_patterns.append(re.compile(p, re.IGNORECASE))
                except re.error:
                    pass  # skip invalid patterns

            self.skills.append({
                "name": skill_def["name"],
                "description": skill_def.get("description", ""),
                "priority": skill_def.get("priority", 0),
                "adapter_path": skill_def.get("adapter_path", ""),
                "system_prompt": skill_def.get("system_prompt", ""),
                "keywords": [kw.lower() for kw in skill_def.get("keywords", [])],
                "patterns": compiled_patterns,
            })

        self.skills.sort(key=lambda s: s["priority"], reverse=True)

        # Build lookup by name
        self._by_name = {s["name"]: s for s in self.skills}

    def route(self, message: str, history: Optional[list] = None) -> str:
        """
        Route a user message to the best-matching skill.

        Args:
            message: The user's input message.
            history: Ignored by keyword router (accepted for interface compat).

        Returns:
            Skill name string (e.g. "crisis-intervention").
        """
        skill, _, _ = self.route_with_confidence(message, history)
        return skill

    def route_with_confidence(
        self, message: str, history: Optional[list] = None
    ) -> tuple:
        """
        Route a message and return confidence details for debugging.

        Args:
            message: The user's input message.
            history: Ignored by keyword router.

        Returns:
            Tuple of (skill_name, confidence, details_dict).
            details_dict contains 'keyword_matches', 'pattern_matches',
            'scores' per skill, and 'router_type'.
        """
        msg_lower = message.lower()
        details = {"scores": {}, "router_type": "keyword"}
        best_skill = self.default_skill
        best_score = 0.0

        for skill in self.skills:
            name = skill["name"]
            if name == self.default_skill and skill["priority"] == 0:
                # Default skill is the fallback — no matching needed
                continue

            keyword_matches = []
            pattern_matches = []

            # Keyword matching (exact substring in lowered message)
            for kw in skill["keywords"]:
                if kw in msg_lower:
                    keyword_matches.append(kw)

            # Regex pattern matching
            for pattern in skill["patterns"]:
                match = pattern.search(message)
                if match:
                    pattern_matches.append(match.group())

            # Score: weighted combination of matches
            # Keywords contribute 0.3 each (capped at 1.0)
            # Patterns contribute 0.4 each (capped at 1.0)
            kw_score = min(len(keyword_matches) * 0.3, 1.0)
            pat_score = min(len(pattern_matches) * 0.4, 1.0)
            score = min(kw_score + pat_score, 1.0)

            details["scores"][name] = {
                "score": score,
                "keyword_matches": keyword_matches,
                "pattern_matches": pattern_matches,
            }

            if score > best_score:
                best_score = score
                best_skill = name

        # If no skill scored above threshold, use default
        if best_score < self.confidence_threshold:
            best_skill = self.default_skill

        return best_skill, best_score, details

    def get_system_prompt(self, skill_name: str) -> str:
        """Get the system prompt for a given skill."""
        skill = self._by_name.get(skill_name)
        if skill:
            return skill["system_prompt"]
        # Fallback
        default = self._by_name.get(self.default_skill)
        return default["system_prompt"] if default else ""

    def get_adapter_path(self, skill_name: str) -> str:
        """Get the adapter path for a given skill."""
        skill = self._by_name.get(skill_name)
        if skill:
            return skill["adapter_path"]
        return ""

    def list_skills(self) -> list:
        """Return list of all skill names in priority order."""
        return [s["name"] for s in self.skills]
