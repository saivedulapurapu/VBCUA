"""
Pure display helpers for an evaluation returned by the backend API.
They only read stored values: the frontend never recomputes scores or
re-matches keywords, and never uses the sidebar selection for a past result.
"""
from typing import Any, Dict, List, Tuple

FOUND = "found"
MISSING = "missing"
NOT_RECORDED = "not recorded"


def fluency_text(eval_res: Dict[str, Any]) -> str:
    """Stored fluency score, or "n/a" for older attempts that did not store it."""
    score = eval_res.get("fluency_score")
    return f"{score:.0f}%" if score is not None else "n/a"


def keyword_badges(eval_res: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    (keyword, status) pairs for the evaluated attempt, from the stored keyword lists.
    Older attempts without stored lists show the evaluated concept's keywords as NOT_RECORDED.
    """
    mentioned = eval_res.get("mentioned_keywords")
    missing = eval_res.get("missing_keywords")
    if mentioned is not None and missing is not None:
        return [(kw, FOUND) for kw in mentioned] + [(kw, MISSING) for kw in missing]
    concept = eval_res.get("concept") or {}
    return [(kw, NOT_RECORDED) for kw in concept.get("keywords", [])]


def concept_title(eval_res: Dict[str, Any]) -> str:
    """Title of the evaluated concept (never the sidebar selection)."""
    concept = eval_res.get("concept")
    if concept and concept.get("title"):
        return concept["title"]
    return f"Concept #{eval_res.get('concept_id', '?')}"
