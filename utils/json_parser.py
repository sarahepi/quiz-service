"""
Safe JSON parsing utilities for LLM-generated output.
LLM outputs are inherently unstable; this module normalizes and validates
raw completions into strict MCQ-compliant structures.
"""

import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Required keys for a valid MCQ entry
_MCQ_REQUIRED_KEYS: frozenset[str] = frozenset({"question", "A", "B", "C", "D", "answer"})
_VALID_ANSWERS: frozenset[str] = frozenset({"A", "B", "C", "D"})


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """
    Attempt to extract the first valid JSON object or array from raw text.

    Args:
        text: Raw LLM output string.

    Returns:
        Extracted JSON string or None if not found.
    """
    # Try to find a JSON array or object
    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return None


def _validate_mcq(mcq: Any) -> bool:
    """
    Validate that an MCQ entry has all required keys with non-empty string values.

    Args:
        mcq: Candidate MCQ dictionary.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(mcq, dict):
        return False
    if not _MCQ_REQUIRED_KEYS.issubset(mcq.keys()):
        return False
    for key in _MCQ_REQUIRED_KEYS:
        if not isinstance(mcq[key], str) or not mcq[key].strip():
            return False
    if mcq["answer"].strip().upper() not in _VALID_ANSWERS:
        return False
    return True


def _normalize_mcq(mcq: dict) -> dict:
    """
    Normalize MCQ keys and values to canonical form.

    Args:
        mcq: Raw MCQ dictionary.

    Returns:
        Normalized MCQ dictionary.
    """
    return {
        "question": mcq["question"].strip(),
        "A": mcq["A"].strip(),
        "B": mcq["B"].strip(),
        "C": mcq["C"].strip(),
        "D": mcq["D"].strip(),
        "answer": mcq["answer"].strip().upper(),
    }


def parse_llm_response(raw: str) -> list[dict]:
    """
    Parse and validate LLM-generated MCQ output.

    Handles:
    - Markdown code fences
    - JSON embedded in prose
    - {"mcqs": [...]} wrapper or bare array
    - Partial/malformed entries (skipped with warning)

    Args:
        raw: Raw string output from the LLM.

    Returns:
        List of validated, normalized MCQ dictionaries.

    Raises:
        ValueError: If no valid MCQs can be extracted.
    """
    text = _strip_markdown_fences(raw)

    # Attempt direct parse first
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parse failed; attempting extraction.")
        extracted = _extract_json_block(text)
        if extracted:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError as exc:
                logger.warning("JSON extraction failed: %s", exc)

    if parsed is None:
        raise ValueError(f"Could not extract valid JSON from LLM response: {raw[:200]!r}")

    # Unwrap {"mcqs": [...]} envelope if present
    if isinstance(parsed, dict):
        for key in ("mcqs", "questions", "data", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Single MCQ object
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not a list of MCQs.")

    valid_mcqs: list[dict] = []
    for idx, item in enumerate(parsed):
        if _validate_mcq(item):
            valid_mcqs.append(_normalize_mcq(item))
        else:
            logger.warning("Skipping invalid MCQ at index %d: %s", idx, item)

    if not valid_mcqs:
        raise ValueError("No valid MCQs found in LLM response.")

    return valid_mcqs


def deduplicate_mcqs(mcqs: list[dict]) -> list[dict]:
    """
    Remove duplicate MCQs based on question text similarity.

    Args:
        mcqs: List of MCQ dictionaries.

    Returns:
        Deduplicated list preserving insertion order.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for mcq in mcqs:
        key = re.sub(r"\s+", " ", mcq["question"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(mcq)
    return unique
