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
    return (
        "<p style='color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 20px;'>"
        f"{esc(text)}</p>"
    )


def keyword_badges_html(badges: List[Tuple[str, str]]) -> str:
    return "".join(
        f"<span class='badge {_BADGE_CLASS.get(status, 'badge-muted')}'>{_BADGE_PREFIX.get(status, '')}{esc(kw)}</span>"
        for kw, status in badges
    )


def transcript_html(text: str) -> str:
    return (
        "<div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); "
        "padding: 15px; border-radius: 8px; font-style: italic; color: #a0aec0;'>"
        f"\"{esc(text)}\"</div>"
    )


def grade_color(grade: str) -> str:
    if "Strong" in grade:
        return "#00ff66"
    if "Moderate" in grade:
        return "#ffe600"
    return "#ff416c"


def grade_html(grade: str) -> str:
    return f"<span style='color: {grade_color(grade)}; font-weight: bold;'>{esc(grade)}</span>"


def history_table_html(rows: List[Dict[str, Any]]) -> str:
    """rows: dicts with Date, Concept, Score, Grade (raw grade text), Duration, Fillers."""
    header = "".join(
        f"<th style='padding: 10px;'>{name}</th>"
        for name in ["Date", "Concept", "Overall Score", "Grade", "Duration", "Filler Words (experimental)", "Inspect"]
    )
    body = ""
    for i, row in enumerate(rows):
        body += (
            "<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>"
            f"<td style='padding: 10px;'>{esc(row['Date'])}</td>"
            f"<td style='padding: 10px; font-weight: 600;'>{esc(row['Concept'])}</td>"
            f"<td style='padding: 10px;'>{esc(row['Score'])}</td>"
            f"<td style='padding: 10px;'>{grade_html(row['Grade'])}</td>"
            f"<td style='padding: 10px;'>{esc(row['Duration'])}</td>"
            f"<td style='padding: 10px;'>{esc(row['Fillers'])}</td>"
            f"<td style='padding: 10px;'><a href='/?load_attempt={int(i)}' target='_self' "
            "style='color: #bd00ff; text-decoration: none; font-weight: bold;'>View</a></td>"
            "</tr>"
        )
    return (
        "<table style='width:100%; border-collapse: collapse; margin-top:15px; color:#f1f3f9;'>"
        "<thead><tr style='border-bottom: 2px solid rgba(255,255,255,0.1); text-align: left;'>"
        f"{header}</tr></thead><tbody>{body}</tbody></table>"
    )


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
