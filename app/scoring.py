from __future__ import annotations

from difflib import SequenceMatcher


def calculate_similarity(user_answer: str, correct_answer: str) -> int:
    """Return integer similarity percentage using SequenceMatcher."""
    ratio = SequenceMatcher(None, user_answer, correct_answer).ratio()
    return int(ratio * 100)


def classify_score(similarity: int) -> int:
    """Map similarity to score_class per spec."""
    if similarity == 100:
        return 2
    if similarity == 0:
        return 0
    return 1
