"""
Dynamic system prompt builder for mental health counseling sessions.

Replaces static system prompt strings with context-aware prompts that
adapt to crisis level, user locale, and session state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Crisis context blocks
# ---------------------------------------------------------------------------

CRISIS_BLOCKS = {
    "high": (
        "CRITICAL: The user may be in immediate danger. Prioritize safety above all else. "
        "Validate their feelings, assess immediate risk, and provide crisis resources. "
        "Do not attempt therapy techniques — focus entirely on safety and de-escalation."
    ),
    "immediate": (
        "CRITICAL: The user may be in immediate danger. Prioritize safety above all else. "
        "Validate their feelings, assess immediate risk, and provide crisis resources. "
        "Do not attempt therapy techniques — focus entirely on safety and de-escalation."
    ),
    "medium": (
        "IMPORTANT: The user has shown signs of distress. Be especially gentle and validating. "
        "Include crisis resources in your response. Watch for escalation cues."
    ),
    "low": (
        "Note: Some mild distress indicators detected. Be attentive to emotional cues "
        "and gently offer professional resources if appropriate."
    ),
}

# ---------------------------------------------------------------------------
# Region-specific crisis resources
# ---------------------------------------------------------------------------

REGION_RESOURCES = {
    "HK": (
        "Hong Kong crisis resources: "
        "The Samaritan Befrienders Hong Kong (2389 2222, 24hrs), "
        "Suicide Prevention Services (2382 0000, 24hrs), "
        "Hospital Authority Mental Health Line (2466 7350). "
        "For immediate danger, call 999."
    ),
    "US": (
        "US crisis resources: "
        "988 Suicide & Crisis Lifeline (call or text 988, 24/7)."
    ),
    "UK": (
        "UK crisis resources: "
        "Samaritans (116 123, 24/7, free from any phone)."
    ),
    "CA": (
        "Canada crisis resources: "
        "Crisis Services Canada (1-833-456-4566, 24/7)."
    ),
    "AU": (
        "Australia crisis resources: "
        "Lifeline (13 11 14, 24/7)."
    ),
}


class TherapyPromptBuilder:
    """Fluent builder for constructing dynamic system prompts.

    Usage::

        prompt = (
            TherapyPromptBuilder(config_path="skills_config.json")
            .with_skill("crisis-intervention")
            .with_crisis_context("high")
            .with_user_profile(region="HK")
            .with_session_summary("User discussed anxiety earlier.")
            .build()
        )
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Load skills_config.json for base prompts."""
        if config_path is None:
            config_path = str(Path(__file__).parent / "skills_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self._skill_prompts: dict[str, str] = {}
        for skill in config.get("skills", []):
            self._skill_prompts[skill["name"]] = skill.get("system_prompt", "")

        self._default_skill = config.get("default_skill", "general-support")

        # Builder state (reset on each chain)
        self._base_prompt: str = ""
        self._crisis_block: str = ""
        self._session_summary: str = ""
        self._locale_block: str = ""

    def with_skill(self, skill_name: str) -> TherapyPromptBuilder:
        """Set the base skill prompt from config."""
        self._base_prompt = self._skill_prompts.get(
            skill_name, self._skill_prompts.get(self._default_skill, "")
        )
        return self

    def with_crisis_context(self, crisis_level: str) -> TherapyPromptBuilder:
        """Inject crisis-awareness instructions based on level."""
        self._crisis_block = CRISIS_BLOCKS.get(crisis_level, "")
        return self

    def with_user_profile(
        self, language: str = "en", region: str = "US"
    ) -> TherapyPromptBuilder:
        """Add locale-specific resources and cultural context."""
        self._locale_block = REGION_RESOURCES.get(region, "")
        return self

    def with_session_summary(self, summary: str) -> TherapyPromptBuilder:
        """Inject compacted session summary from ConversationCompactor."""
        self._session_summary = summary or ""
        return self

    def build(self) -> str:
        """Compose and return the final system prompt string.

        Composition order:
        1. Base skill prompt
        2. Crisis context block
        3. Session summary block
        4. User profile / locale block
        """
        parts: list[str] = []

        if self._base_prompt:
            parts.append(self._base_prompt)

        if self._crisis_block:
            parts.append(self._crisis_block)

        if self._session_summary:
            parts.append(f"Session context: {self._session_summary}")

        if self._locale_block:
            parts.append(self._locale_block)

        prompt = "\n\n".join(parts)

        # Reset builder state for next use
        self._base_prompt = ""
        self._crisis_block = ""
        self._session_summary = ""
        self._locale_block = ""

        return prompt
