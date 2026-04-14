"""
Crisis resource lookup tool.

Returns verified crisis hotlines and resources by locale.
These are hardcoded (not hallucinated) for safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CrisisResource:
    """A crisis support resource."""

    name: str
    phone: str
    hours: str
    languages: list[str]
    note: Optional[str] = None
    website: Optional[str] = None

    def format(self) -> str:
        """Format for display."""
        parts = [f"**{self.name}**"]
        if self.phone:
            parts.append(f"Phone: {self.phone}")
        if self.website:
            parts.append(f"Web: {self.website}")
        if self.hours != "24/7":
            parts.append(f"Hours: {self.hours}")
        if self.note:
            parts.append(f"({self.note})")
        return " | ".join(parts)


# Verified crisis resources by region
CRISIS_RESOURCES: dict[str, list[CrisisResource]] = {
    "HK": [
        CrisisResource(
            name="Samaritans Hong Kong",
            phone="2389 2222",
            hours="24/7",
            languages=["English", "Cantonese"],
        ),
        CrisisResource(
            name="Suicide Prevention Services",
            phone="2382 0000",
            hours="24/7",
            languages=["Cantonese", "English"],
        ),
        CrisisResource(
            name="The Samaritan Befrienders HK",
            phone="2389 2222",
            hours="24/7",
            languages=["Cantonese", "English", "Mandarin"],
        ),
    ],
    "US": [
        CrisisResource(
            name="988 Suicide & Crisis Lifeline",
            phone="988",
            hours="24/7",
            languages=["English", "Spanish"],
            note="Call or text 988",
        ),
        CrisisResource(
            name="Crisis Text Line",
            phone="Text HOME to 741741",
            hours="24/7",
            languages=["English"],
        ),
        CrisisResource(
            name="SAMHSA National Helpline",
            phone="1-800-662-4357",
            hours="24/7",
            languages=["English", "Spanish"],
            note="Mental health and substance abuse referrals",
        ),
    ],
    "UK": [
        CrisisResource(
            name="Samaritans",
            phone="116 123",
            hours="24/7",
            languages=["English"],
            note="Free to call",
        ),
        CrisisResource(
            name="SHOUT Crisis Text Line",
            phone="Text SHOUT to 85258",
            hours="24/7",
            languages=["English"],
        ),
        CrisisResource(
            name="CALM",
            phone="0800 58 58 58",
            hours="5pm-midnight daily",
            languages=["English"],
            note="For men",
        ),
    ],
    "AU": [
        CrisisResource(
            name="Lifeline Australia",
            phone="13 11 14",
            hours="24/7",
            languages=["English"],
        ),
        CrisisResource(
            name="Beyond Blue",
            phone="1300 22 4636",
            hours="24/7",
            languages=["English"],
        ),
        CrisisResource(
            name="Kids Helpline",
            phone="1800 55 1800",
            hours="24/7",
            languages=["English"],
            note="For ages 5-25",
        ),
    ],
    "CA": [
        CrisisResource(
            name="988 Suicide Crisis Helpline",
            phone="988",
            hours="24/7",
            languages=["English", "French"],
            note="Call or text 988",
        ),
        CrisisResource(
            name="Crisis Services Canada",
            phone="1-833-456-4566",
            hours="24/7",
            languages=["English", "French"],
        ),
        CrisisResource(
            name="Kids Help Phone",
            phone="1-800-668-6868",
            hours="24/7",
            languages=["English", "French"],
            note="For youth",
        ),
    ],
    "INTL": [
        CrisisResource(
            name="International Association for Suicide Prevention",
            phone="",
            hours="24/7",
            languages=["Multiple"],
            website="https://www.iasp.info/resources/Crisis_Centres/",
            note="Directory of crisis centers worldwide",
        ),
        CrisisResource(
            name="Befrienders Worldwide",
            phone="",
            hours="24/7",
            languages=["Multiple"],
            website="https://www.befrienders.org/",
            note="Find a helpline in your country",
        ),
    ],
}


def get_crisis_resources(locale: str = "HK") -> list[CrisisResource]:
    """Get crisis resources for a locale.

    Args:
        locale: Two-letter region code (HK, US, UK, AU, CA) or INTL.

    Returns:
        List of CrisisResource objects.
    """
    locale = locale.upper()
    return CRISIS_RESOURCES.get(locale, CRISIS_RESOURCES.get("INTL", []))


def crisis_lookup(locale: str = "HK") -> str:
    """Return formatted crisis hotlines for the given locale.

    This is a safe tool - always appropriate to provide crisis resources.

    Args:
        locale: Two-letter region code (default: HK for Hong Kong).

    Returns:
        Formatted string with crisis resources.
    """
    resources = get_crisis_resources(locale)

    if not resources:
        resources = CRISIS_RESOURCES["INTL"]

    lines = [
        "**Crisis Support Resources**",
        "",
        "If you're in immediate danger, please call emergency services.",
        "",
    ]

    for resource in resources:
        lines.append(resource.format())
        lines.append("")

    lines.append(
        "Remember: You don't have to face this alone. "
        "These services are confidential and available to help."
    )

    return "\n".join(lines)
