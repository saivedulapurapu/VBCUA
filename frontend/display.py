"""
Pure display helpers for an evaluation returned by the backend API.
They only read stored values: the frontend never recomputes scores or
re-matches keywords, and never uses the sidebar selection for a past result.
"""
import html
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


# --- HTML builders: every data value is escaped before it reaches st.html ---

PLAIN = "plain"

_BADGE_CLASS = {FOUND: "badge-success", MISSING: "badge-danger", NOT_RECORDED: "badge-muted", PLAIN: "badge-indigo"}
_BADGE_PREFIX = {FOUND: "✓ ", MISSING: "✗ "}


def esc(value: Any) -> str:
    """HTML-escape any value, including quotes, for text and attribute positions."""
    return html.escape(str(value), quote=True)


def description_html(text: str) -> str:
    return f"<p class='concept-description'>{esc(text)}</p>"


def keyword_badges_html(badges: List[Tuple[str, str]]) -> str:
    return "".join(
        f"<span class='badge {_BADGE_CLASS.get(status, 'badge-muted')}'>{_BADGE_PREFIX.get(status, '')}{esc(kw)}</span>"
        for kw, status in badges
    )


def transcript_html(text: str) -> str:
    return f"<div class='transcript'>{esc(text)}</div>"


# --- Session / navigation helpers ---

def initial_concept_id(concepts: List[Dict[str, Any]], requested: Any) -> Any:
    """Concept to preselect: the one named in the URL (?concept=<id>) if it exists, else the first."""
    ids = [c["id"] for c in concepts]
    try:
        wanted = int(requested)
    except (TypeError, ValueError):
        wanted = None
    if wanted in ids:
        return wanted
    return ids[0] if ids else None


def history_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Newest-first rows for the history table; values are plain text (no HTML)."""
    rows = []
    for item in sorted(items, key=lambda x: x["id"], reverse=True):
        created = str(item.get("created_at", ""))[:16].replace("T", " ")
        rows.append({
            "id": item["id"],
            "Date": created,
            "Concept": concept_title(item),
            "Overall": round(float(item["final_score"]), 1),
            "Grade": str(item.get("understanding_grade", "")).replace(" Understanding", ""),
            "Duration (s)": round(float(item.get("speech_duration") or 0.0), 1),
        })
    return rows


def feedback_notice(eval_res: Dict[str, Any]) -> Any:
    """Shown when the AI feedback was unavailable and the rule-based generator was used."""
    if eval_res.get("feedback_source") == "rule-based":
        return ("AI feedback was unavailable for this attempt (no API key, or the AI service was busy), "
                "so this feedback was produced by the built-in rule-based generator.")
    return None


# --- Results view helpers ---

def grade_badge_color(grade: str) -> str:
    """st.badge colour for a grade."""
    if "Strong" in grade:
        return "green"
    if "Moderate" in grade:
        return "orange"
    return "red"


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _pause_ratio(eval_res: Dict[str, Any]) -> Any:
    value = eval_res.get("pause_ratio_v1")
    return value if value is not None else eval_res.get("pause_ratio")


def score_cards(eval_res: Dict[str, Any]) -> List[Dict[str, str]]:
    """The three headline scores with a one-line explanation each."""
    pause = _pause_ratio(eval_res)
    return [
        {
            "label": "Overall",
            "value": _fmt(eval_res.get("final_score")),
            "caption": "70% understanding + 30% fluency",
            "help": "Weighted combination of the understanding and fluency scores (0–100).",
        },
        {
            "label": "Understanding",
            "value": _fmt(eval_res.get("understanding_score")),
            "caption": f"Similarity {_fmt(eval_res.get('semantic_score'))} · keywords {_fmt(eval_res.get('keyword_coverage'), 0, '%')}",
            "help": "70% semantic similarity to the reference definition + 30% keyword coverage.",
        },
        {
            "label": "Fluency",
            "value": fluency_text(eval_res).rstrip("%"),
            "caption": f"Pauses {_fmt(None if pause is None else pause * 100, 0, '%')} of speaking time",
            "help": "Based on the share of silence between the first and last speech; about 10–30% is natural.",
        },
    ]


def details_rows(eval_res: Dict[str, Any]) -> List[Dict[str, str]]:
    """Every stored sub-score and setting for the Details tab (plain text)."""
    pause_v1 = eval_res.get("pause_ratio_v1")
    rows = [
        ("Semantic similarity", _fmt(eval_res.get("semantic_score")), "max(0, cosine) × 100"),
        ("Cosine similarity", _fmt(eval_res.get("cosine_similarity"), 3), "Sentence-BERT, all-MiniLM-L6-v2"),
        ("Keyword coverage", _fmt(eval_res.get("keyword_coverage"), 1, "%"), "Whole-word matches"),
        ("Understanding score", _fmt(eval_res.get("understanding_score")), "70% similarity + 30% keywords"),
        ("Pause ratio", _fmt(None if pause_v1 is None else pause_v1 * 100, 1, "%"), "Between first and last speech"),
        ("Fluency score", _fmt(eval_res.get("fluency_score")), "Pause-based"),
        ("Overall score", _fmt(eval_res.get("final_score")), "70% understanding + 30% fluency"),
        ("Filler words", str(eval_res.get("filler_words_count", "n/a")), "Experimental, not scored"),
        ("Duration", _fmt(eval_res.get("speech_duration"), 1, " s"), ""),
        ("Speech-to-text model", str(eval_res.get("asr_model") or "n/a"), "ASR confidence " + _fmt(eval_res.get("asr_confidence"), 3)),
        ("Feedback source", str(eval_res.get("feedback_source") or "n/a"), str(eval_res.get("llm_model") or "")),
        ("Scoring methods", f"semantic {eval_res.get('semantic_method') or 'n/a'}, fluency {eval_res.get('fluency_method') or 'n/a'}", "B0 = original baseline"),
    ]
    return [{"Measure": m, "Value": v, "Note": n} for m, v, n in rows]


def demote_headings(markdown: str) -> str:
    """Render any feedback heading (#, ##, ###…) as a small "####" heading so it fits the results card."""
    lines = []
    for line in markdown.splitlines():
        stripped = line.lstrip()
        level = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= level <= 6 and stripped[level:level + 1] == " ":
            line = "#### " + stripped[level + 1:]
        lines.append(line)
    return "\n".join(lines)
