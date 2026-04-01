"""
Skill Registry with validation for the Mental Health LLM.

Provides a decorator-based skill registration system that serves as the
single source of truth for skill definitions.  ``skills_config.json`` is
*generated* from the registry rather than hand-edited.

Features:
  - ``@registry.skill`` decorator for declarative registration
  - Auto-validation on startup: adapter files exist, prompt non-empty,
    keywords non-overlapping across skills, priorities unique
  - CLI:  ``python -m mental_health_llm.skill_registry validate``

Source pattern: claw-code execution_registry.py (registry + JSON snapshots).

Usage:

    from mental_health_llm.skill_registry import registry

    @registry.skill(
        name="crisis-intervention",
        priority=100,
        adapter_path="adapters/crisis-intervention",
        system_prompt="You are a crisis intervention specialist ...",
        keywords=["kill myself", "suicide", ...],
        patterns=[r"\\bsuicid\\w*\\b", ...],
    )
    def crisis_intervention():
        '''Immediate safety-focused response.'''

    # Validate all registered skills
    errors = registry.validate()

    # Export to skills_config.json
    registry.export_config("mental_health_llm/skills_config.json")
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill definition
# ---------------------------------------------------------------------------


@dataclass
class SkillDefinition:
    """Complete definition of a single counseling skill."""

    name: str
    description: str = ""
    priority: int = 0
    adapter_path: str = ""
    system_prompt: str = ""
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)

    def to_config_dict(self) -> dict:
        """Serialize to the format expected by skills_config.json."""
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "adapter_path": self.adapter_path,
            "system_prompt": self.system_prompt,
            "keywords": self.keywords,
            "patterns": self.patterns,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Central registry for all counseling skills.

    Skills are registered via the ``@registry.skill(...)`` decorator or
    by calling ``registry.register(...)`` directly.  The registry can
    validate consistency and export ``skills_config.json``.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        self._default_skill: str = "general-support"

    # -- Registration ------------------------------------------------------

    def skill(
        self,
        name: str,
        *,
        description: str = "",
        priority: int = 0,
        adapter_path: str = "",
        system_prompt: str = "",
        keywords: list[str] | None = None,
        patterns: list[str] | None = None,
    ):
        """Decorator that registers a skill definition.

        The decorated function is returned unchanged — its docstring is
        used as a fallback description if ``description`` is empty.
        """

        def decorator(fn):
            desc = description or (fn.__doc__ or "").strip()
            self.register(
                SkillDefinition(
                    name=name,
                    description=desc,
                    priority=priority,
                    adapter_path=adapter_path,
                    system_prompt=system_prompt,
                    keywords=keywords or [],
                    patterns=patterns or [],
                )
            )
            return fn

        return decorator

    def register(self, skill_def: SkillDefinition) -> None:
        """Register a skill definition directly."""
        if skill_def.name in self._skills:
            logger.warning("Overwriting existing skill registration: %s", skill_def.name)
        self._skills[skill_def.name] = skill_def

    def set_default_skill(self, name: str) -> None:
        self._default_skill = name

    # -- Accessors ---------------------------------------------------------

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return skills sorted by priority (highest first)."""
        return sorted(self._skills.values(), key=lambda s: s.priority, reverse=True)

    @property
    def skill_names(self) -> list[str]:
        return [s.name for s in self.list_skills()]

    # -- Validation --------------------------------------------------------

    def validate(self, *, project_root: str | Path | None = None) -> list[str]:
        """Validate all registered skills.

        Checks:
          1. Every skill has a non-empty system_prompt.
          2. Every skill's adapter_path directory exists (if project_root given).
          3. No two skills share the same priority (except priority 0).
          4. No keyword appears in more than one skill.
          5. Regex patterns compile without errors.

        Returns:
            List of error messages (empty = all valid).
        """
        errors: list[str] = []
        skills = self.list_skills()

        if not skills:
            errors.append("No skills registered.")
            return errors

        # 1. Non-empty system prompts
        for s in skills:
            if not s.system_prompt.strip():
                errors.append(f"[{s.name}] system_prompt is empty.")

        # 2. Adapter path existence
        if project_root is not None:
            root = Path(project_root)
            for s in skills:
                if s.adapter_path:
                    adapter_dir = root / s.adapter_path
                    if not adapter_dir.exists():
                        errors.append(
                            f"[{s.name}] adapter_path does not exist: {adapter_dir}"
                        )

        # 3. Unique priorities (excluding 0, which is the catch-all)
        seen_priorities: dict[int, str] = {}
        for s in skills:
            if s.priority == 0:
                continue
            if s.priority in seen_priorities:
                errors.append(
                    f"[{s.name}] duplicate priority {s.priority} "
                    f"(also used by {seen_priorities[s.priority]})."
                )
            else:
                seen_priorities[s.priority] = s.name

        # 4. Non-overlapping keywords
        keyword_owners: dict[str, str] = {}
        for s in skills:
            for kw in s.keywords:
                kw_lower = kw.lower()
                if kw_lower in keyword_owners:
                    errors.append(
                        f"[{s.name}] keyword '{kw}' overlaps with "
                        f"skill '{keyword_owners[kw_lower]}'."
                    )
                else:
                    keyword_owners[kw_lower] = s.name

        # 5. Pattern compilation
        for s in skills:
            for i, pat in enumerate(s.patterns):
                try:
                    re.compile(pat, re.IGNORECASE)
                except re.error as e:
                    errors.append(
                        f"[{s.name}] pattern[{i}] is invalid: {e}"
                    )

        return errors

    # -- Export / Import ---------------------------------------------------

    def export_config(
        self,
        output_path: str | Path,
        *,
        extra: dict | None = None,
    ) -> None:
        """Write skills_config.json from the registry.

        Args:
            output_path: Destination file path.
            extra: Additional top-level keys to include (e.g. crisis_gate,
                   embedding_router settings).
        """
        config: dict = {
            "skills": [s.to_config_dict() for s in self.list_skills()],
            "default_skill": self._default_skill,
        }
        if extra:
            config.update(extra)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        logger.info("Exported %d skills to %s", len(self._skills), path)

    def import_config(self, config_path: str | Path) -> None:
        """Import skill definitions from an existing skills_config.json.

        This allows bootstrapping the registry from the legacy hand-edited
        config file during migration.
        """
        path = Path(config_path)
        with open(path) as f:
            config = json.load(f)

        self._default_skill = config.get("default_skill", "general-support")

        for skill_dict in config.get("skills", []):
            self.register(
                SkillDefinition(
                    name=skill_dict["name"],
                    description=skill_dict.get("description", ""),
                    priority=skill_dict.get("priority", 0),
                    adapter_path=skill_dict.get("adapter_path", ""),
                    system_prompt=skill_dict.get("system_prompt", ""),
                    keywords=skill_dict.get("keywords", []),
                    patterns=skill_dict.get("patterns", []),
                )
            )

        logger.info("Imported %d skills from %s", len(self._skills), path)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = SkillRegistry()


# ---------------------------------------------------------------------------
# CLI entry point:  python -m mental_health_llm.skill_registry validate
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """CLI for validating and exporting the skill registry."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m mental_health_llm.skill_registry",
        description="Skill registry validation and export tool",
    )
    sub = parser.add_subparsers(dest="command")

    # --- validate ---------------------------------------------------------
    val_parser = sub.add_parser("validate", help="Validate skill definitions")
    val_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to skills_config.json to import before validating",
    )
    val_parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root for checking adapter paths (default: auto-detect)",
    )

    # --- export -----------------------------------------------------------
    exp_parser = sub.add_parser(
        "export", help="Export registry to skills_config.json"
    )
    exp_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to existing skills_config.json to import first",
    )
    exp_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: mental_health_llm/skills_config.json)",
    )

    # --- list -------------------------------------------------------------
    sub.add_parser("list", help="List registered skills")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Resolve project root
    pkg_dir = Path(__file__).parent
    project_root = pkg_dir.parent

    # Import config if specified or use default
    config_path = args.config if hasattr(args, "config") and args.config else None
    if config_path is None:
        default_config = pkg_dir / "skills_config.json"
        if default_config.exists():
            config_path = str(default_config)

    if config_path:
        registry.import_config(config_path)
        print(f"Imported {len(registry._skills)} skills from {config_path}")

    if args.command == "validate":
        root = args.project_root or str(project_root)
        errors = registry.validate(project_root=root)

        if errors:
            print(f"\n{len(errors)} validation error(s):\n")
            for i, err in enumerate(errors, 1):
                print(f"  {i}. {err}")
            sys.exit(1)
        else:
            print(f"\nAll {len(registry._skills)} skills passed validation.")
            # Print summary
            for s in registry.list_skills():
                kw_count = len(s.keywords)
                pat_count = len(s.patterns)
                print(
                    f"  {s.priority:3d}  {s.name:<25s}  "
                    f"{kw_count:2d} keywords  {pat_count:2d} patterns"
                )
            sys.exit(0)

    elif args.command == "export":
        output = args.output or str(pkg_dir / "skills_config.json")

        # Preserve extra config keys from original file
        extra: dict = {}
        if config_path:
            with open(config_path) as f:
                original = json.load(f)
            for key in ("confidence_threshold", "router_backend", "crisis_gate", "embedding_router"):
                if key in original:
                    extra[key] = original[key]

        registry.export_config(output, extra=extra)
        print(f"Exported {len(registry._skills)} skills to {output}")

    elif args.command == "list":
        if not registry._skills:
            print("No skills registered.")
            sys.exit(0)

        print(f"\n{len(registry._skills)} registered skills:\n")
        for s in registry.list_skills():
            print(f"  [{s.priority:3d}] {s.name}")
            print(f"        {s.description[:80]}")
            print(f"        adapter: {s.adapter_path}")
            print(f"        keywords: {len(s.keywords)}, patterns: {len(s.patterns)}")
            print()


if __name__ == "__main__":
    _cli_main()
