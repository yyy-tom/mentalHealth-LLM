"""
Post-generation response guard for mental health counseling LLM.

Validates model outputs for harmful advice, dismissive language,
medication hallucinations, and missing crisis resources before
delivering responses to users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


SAFE_FALLBACK = (
    "I want to make sure I'm giving you the most helpful response. "
    "What you're going through matters, and I'd like to understand better. "
    "Could you tell me more about how you're feeling? "
    "If you're in crisis, please reach out to the 988 Suicide & Crisis Lifeline (US), "
    "The Samaritan Befrienders Hong Kong (2389 2222), or your local emergency services."
)

CRISIS_RESOURCES = (
    "\n\nIf you need immediate support, please reach out:\n"
    "- Hong Kong: The Samaritan Befrienders (2389 2222), SPS (2382 0000)\n"
    "- US: 988 Suicide & Crisis Lifeline\n"
    "- UK: Samaritans (116 123)\n"
    "- Canada: Crisis Services Canada (1-833-456-4566)\n"
    "- Australia: Lifeline (13 11 14)"
)

# ---------------------------------------------------------------------------
# Pattern lists
# ---------------------------------------------------------------------------

HARMFUL_ADVICE_PATTERNS = [
    re.compile(r"stop\s+tak(e|ing)\s+(your\s+)?medication", re.IGNORECASE),
    re.compile(
        r"you\s+don'?t\s+need\s+(therapy|medication|treatment|help)", re.IGNORECASE
    ),
    re.compile(r"just\s+(get\s+over|snap\s+out|move\s+on|forget)", re.IGNORECASE),
    re.compile(r"(try|use)\s+(alcohol|drugs|drinking)\s+(to|for)", re.IGNORECASE),
    re.compile(
        r"harm(ing)?\s+(yourself|themselves)\s+is\s+(normal|okay|fine)", re.IGNORECASE
    ),
]

DISMISSIVE_LANGUAGE_PATTERNS = [
    re.compile(r"just\s+think\s+positive", re.IGNORECASE),
    re.compile(r"others?\s+ha(ve|s)\s+it\s+worse", re.IGNORECASE),
    re.compile(r"(it'?s|that'?s)\s+(not\s+)?(a\s+)?big\s+deal", re.IGNORECASE),
    re.compile(
        r"you('re|\s+are)\s+(just|being)\s+(dramatic|sensitive|lazy|weak)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(every|no)one\s+(goes|deals|feels)\s+through\s+(this|that)", re.IGNORECASE
    ),
    re.compile(
        r"have\s+you\s+tried\s+(not\s+being|just\s+being\s+happy)", re.IGNORECASE
    ),
    re.compile(r"(man|woman|grow)\s+up", re.IGNORECASE),
]

MEDICATION_HALLUCINATION_PATTERNS = [
    re.compile(
        r"\b(take|prescribe[ds]?|recommend|start\s+with)\s+\d+\s*m?g\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(prescribe|prescribed|recommend|suggesting)\s+(you\s+)?\w+\s+\d+",
        re.IGNORECASE,
    ),
]


@dataclass
class GuardResult:
    """Outcome of response validation."""

    action: str  # "pass" | "modified" | "blocked"
    response: str  # Original, modified, or replacement response
    flags: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class ResponseGuard:
    """Validates generated responses before delivery to users."""

    def __init__(self) -> None:
        self.harmful_patterns = HARMFUL_ADVICE_PATTERNS
        self.dismissive_patterns = DISMISSIVE_LANGUAGE_PATTERNS
        self.medication_patterns = MEDICATION_HALLUCINATION_PATTERNS

    def validate(
        self, response: str, skill: str, crisis_level: str = "none"
    ) -> GuardResult:
        """Run all checks and return the result.

        Priority: blocked > modified > pass.
        """
        flags: list[str] = []
        details: dict = {}
        action = "pass"
        final_response = response

        # --- Blocking checks ---
        harmful, harmful_matches = self._check_harmful_advice(response)
        if harmful:
            flags.append("harmful_advice")
            details["harmful_advice"] = harmful_matches
            action = "blocked"
            final_response = SAFE_FALLBACK

        med_halluc, med_matches = self._check_medication_hallucination(response)
        if med_halluc:
            flags.append("medication_hallucination")
            details["medication_hallucination"] = med_matches
            action = "blocked"
            final_response = SAFE_FALLBACK

        # If already blocked, return immediately
        if action == "blocked":
            return GuardResult(
                action=action,
                response=final_response,
                flags=flags,
                details=details,
            )

        # --- Modification checks ---
        dismissive, dismissive_matches = self._check_dismissive_language(response)
        if dismissive:
            flags.append("dismissive_language")
            details["dismissive_language"] = dismissive_matches
            action = "modified"
            final_response = (
                "I hear you, and your feelings are completely valid. " + final_response
            )

        missing, resource_note = self._check_missing_crisis_resources(
            response, crisis_level
        )
        if missing:
            flags.append("missing_crisis_resources")
            details["missing_crisis_resources"] = resource_note
            action = "modified"
            final_response = self._inject_crisis_resources(final_response)

        return GuardResult(
            action=action,
            response=final_response,
            flags=flags,
            details=details,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_harmful_advice(self, response: str) -> tuple[bool, list[str]]:
        """Detect harmful advice patterns."""
        matches: list[str] = []
        for pattern in self.harmful_patterns:
            m = pattern.search(response)
            if m:
                matches.append(m.group())
        return bool(matches), matches

    def _check_dismissive_language(self, response: str) -> tuple[bool, list[str]]:
        """Detect dismissive / invalidating language."""
        matches: list[str] = []
        for pattern in self.dismissive_patterns:
            m = pattern.search(response)
            if m:
                matches.append(m.group())
        return bool(matches), matches

    def _check_medication_hallucination(self, response: str) -> tuple[bool, list[str]]:
        """Detect hallucinated medication dosage information."""
        matches: list[str] = []
        for pattern in self.medication_patterns:
            m = pattern.search(response)
            if m:
                matches.append(m.group())
        return bool(matches), matches

    def _check_missing_crisis_resources(
        self, response: str, crisis_level: str
    ) -> tuple[bool, str]:
        """Check that crisis-level responses include crisis resources."""
        if crisis_level not in ("high", "medium", "immediate"):
            return False, ""

        # Check if response already contains some crisis resource indicators
        resource_indicators = [
            "988", "crisis", "emergency", "hotline", "lifeline",
            "2389 2222", "2382 0000", "116 123", "13 11 14",
        ]
        has_resources = any(ind in response.lower() for ind in resource_indicators)
        if has_resources:
            return False, ""

        return True, f"Crisis level is {crisis_level} but no resources found in response"

    def _inject_crisis_resources(self, response: str) -> str:
        """Append standard crisis resources to a response."""
        return response + CRISIS_RESOURCES
