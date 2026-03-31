#!/usr/bin/env python3
"""
Parse the comparison markdown into structured JSON test cases.

Extracts cases from the markdown document, pulling out the situation
description and user turns from the "Our Model" section for each case.

Usage:
    python scripts/evaluation/parse_cases_md.py \
        --input "docs/FYP_ Comparison with other LLM (1).md" \
        --output evaluation/cases.json
"""

import argparse
import json
import re
from pathlib import Path


def split_cases(text: str) -> list[dict]:
    """Split markdown into individual cases by numbered headers."""
    # Match lines like: # 1\. Exam Anxiety  or  # 1. Exam Anxiety
    header_pattern = re.compile(r"^# (\d+)\\?\.\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    cases = []
    for i, match in enumerate(matches):
        case_num = int(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        cases.append({"case_num": case_num, "title": title, "body": body})

    return cases


def extract_situation(body: str) -> str:
    """Extract the Situation paragraph from a case body."""
    # Find ### **Situation** header
    sit_match = re.search(r"###\s+\*\*Situation\*\*\s*\n", body)
    if not sit_match:
        return ""

    rest = body[sit_match.end():]
    # Situation ends at the next ### or # header, or at a blank-line-then-header
    end_match = re.search(r"\n###?\s", rest)
    if end_match:
        situation = rest[: end_match.start()]
    else:
        situation = rest

    return situation.strip()


def extract_our_model_section(body: str) -> str:
    """Extract the Our Model section from a case body."""
    model_match = re.search(r"###\s+\*\*Our Model\*\*\s*\n", body)
    if not model_match:
        return ""

    rest = body[model_match.end():]
    # Ends at next ### header (Wysa, ChatPsychiatrist, etc.) or # header
    end_match = re.search(r"\n###?\s", rest)
    if end_match:
        section = rest[: end_match.start()]
    else:
        section = rest

    return section.strip()


def parse_table_user_turns(section: str) -> list[str]:
    """Parse user turns from a markdown table format.

    Table rows look like:
    | 1 | User | I know I need to study... |  |  |  |
    """
    user_turns = []
    for line in section.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue

        cells = [c.strip() for c in line.split("|")]
        # Split produces empty strings at start/end due to leading/trailing |
        cells = [c for c in cells if c != ""]

        if len(cells) < 3:
            continue

        # Skip header row and separator row
        if cells[0] in ("Turn", ":----", "----"):
            continue
        if cells[0].startswith(":") or cells[0].startswith("-"):
            continue

        role = cells[1].strip() if len(cells) > 1 else ""
        content = cells[2].strip() if len(cells) > 2 else ""

        if role == "User" and content:
            user_turns.append(content)

    return user_turns


def parse_numbered_user_turns(section: str) -> list[str]:
    """Parse user turns from numbered paragraph format.

    Lines look like:
    1\\. Rowan: I've felt low for as long as I can remember...
    2\\. Chatbot: It sounds like...
    """
    user_turns = []
    # Match numbered entries: N\. Name: Content (possibly multiline)
    entry_pattern = re.compile(r"^(\d+)\\?\.\s+(\w+):\s+(.+)", re.MULTILINE)
    matches = list(entry_pattern.finditer(section))

    for i, match in enumerate(matches):
        name = match.group(2).strip()
        if name.lower() == "chatbot":
            continue

        # Content extends until the next entry or end of section
        start = match.start(3)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        content = section[start:end].strip()

        # Clean up: remove trailing blank lines
        content = re.sub(r"\n\n+$", "", content)
        # Collapse internal newlines into spaces for multi-paragraph entries
        content = re.sub(r"\n+", " ", content)

        if content:
            user_turns.append(content)

    return user_turns


def parse_user_turns(section: str) -> list[str]:
    """Extract user turns from the Our Model section, handling both formats."""
    # Try table format first (look for table header row)
    if "| Turn |" in section or "| :----" in section:
        turns = parse_table_user_turns(section)
        if turns:
            return turns

    # Fall back to numbered paragraph format
    turns = parse_numbered_user_turns(section)
    return turns


def parse_cases(input_path: str) -> dict:
    """Parse the full markdown file into structured cases."""
    with open(input_path) as f:
        text = f.read()

    raw_cases = split_cases(text)
    cases = []

    for raw in raw_cases:
        # Skip non-numbered sections (like "template")
        if raw["case_num"] < 1 or raw["case_num"] > 10:
            continue

        situation = extract_situation(raw["body"])
        model_section = extract_our_model_section(raw["body"])
        user_turns = parse_user_turns(model_section)

        case_id = f"case_{raw['case_num']:02d}"
        cases.append({
            "case_id": case_id,
            "title": raw["title"],
            "situation": situation,
            "user_turns": user_turns,
        })

    return {
        "metadata": {
            "source": input_path,
            "total_cases": len(cases),
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse comparison markdown into structured JSON")
    parser.add_argument(
        "--input",
        type=str,
        default="docs/FYP_ Comparison with other LLM (1).md",
        help="Path to the comparison markdown file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/cases.json",
        help="Output path for structured JSON",
    )
    args = parser.parse_args()

    result = parse_cases(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Parsed {result['metadata']['total_cases']} cases from {args.input}")
    for case in result["cases"]:
        print(f"  {case['case_id']}: {case['title']} ({len(case['user_turns'])} user turns)")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
