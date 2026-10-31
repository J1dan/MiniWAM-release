"""Exact language-instruction validation shared by extraction and caches."""

from __future__ import annotations


def require_instruction_string(value, source: str) -> str:
    """Return one nonempty instruction without coercion or normalization."""
    if not isinstance(value, str):
        raise TypeError(
            f"Language instruction from {source} must be a string, "
            f"got {type(value).__name__}."
        )
    if not value:
        raise ValueError(f"Language instruction from {source} is empty.")
    return value


def require_exact_instruction_match(
    source_instruction,
    current_instruction,
    *,
    source: str,
    current: str,
) -> str:
    """Require exact Python string equality without Unicode normalization."""
    source_value = require_instruction_string(source_instruction, source)
    current_value = require_instruction_string(current_instruction, current)
    if source_value != current_value:
        raise ValueError(
            "Language instruction mismatch: "
            f"source={source_value!r} ({source}), "
            f"current={current_value!r} ({current})."
        )
    return source_value
