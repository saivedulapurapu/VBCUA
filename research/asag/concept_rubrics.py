"""
The app's concept rubrics (research/rubrics/*.json) and the readable review document built from them.

The JSON files are the source; the Markdown file is generated, so the two cannot drift apart:
    python -m research.asag.concept_rubrics      # rewrites research/rubrics/DRAFT_rubrics_for_review.md
"""
import json
from pathlib import Path
from typing import Dict, List

from research.asag.rubrics import Rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRIC_DIR = ROOT / "research" / "rubrics"
REVIEW_FILE = RUBRIC_DIR / "DRAFT_rubrics_for_review.md"
DRAFT = "DRAFT – NOT FACULTY REVIEWED"

# Proposed combination of the per-item marks, shown to the reviewer and used by the grader
SCORING_RULE = (
    "score = max(0, sum of key-point marks − number of misconceptions stated) ÷ (2 × number of key points) × 100"
)


def load_all(folder: Path = RUBRIC_DIR) -> List[Dict]:
    rubrics = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        Rubric.model_validate(data)
        rubrics.append(data)
    return rubrics


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(rubrics: List[Dict]) -> str:
    lines = [
        f"# Concept rubrics for VBCUA — {DRAFT}",
        "",
        f"> **{DRAFT}.** These rubrics were drafted by the student researcher and have not yet been",
        "> checked by a subject expert. They must not be used to grade anyone until reviewed.",
        "",
        "## What these are for",
        "",
        "VBCUA asks a student to explain a computer-science concept aloud for 1–3 minutes. The",
        "recording is transcribed and the explanation is scored. Today the score is a sentence-",
        "embedding similarity to a reference description. The research question (RQ1) is whether",
        "a rubric — the key points a good explanation contains, plus common misconceptions —",
        "agrees better with human markers than that similarity score.",
        "",
        "## How a rubric is applied",
        "",
        "- Each **key point** is marked **0** (missing or wrong), **1** (partly stated or vague) or",
        "  **2** (clearly and correctly stated). The *partial credit* column says what earns a 1.",
        "- Each **misconception** is marked as stated or not stated. It counts as stated only if the",
        "  grader can quote the student's words for it verbatim; otherwise it is treated as not stated.",
        f"- Proposed overall score: {SCORING_RULE}.",
        "- Explanations are spoken, so wording, order and informal language should not matter; only",
        "  whether the idea is there and correct.",
        "",
        "## What I would like you to check",
        "",
        "1. Are the key points correct, and are they the right ones for an undergraduate explanation?",
        "   Should any be added, removed, merged or split?",
        "2. Is a 1–3 minute spoken explanation expected to cover all of them, or should some be optional?",
        "3. Are the misconceptions ones students really state? Are there more common ones?",
        "4. Is the partial-credit description for each key point a fair boundary between 1 and 2?",
        "5. Is the proposed overall score (with a 1-mark deduction per misconception) reasonable?",
        "",
        "Please mark changes directly on this document or in comments.",
        "",
    ]
    for r in rubrics:
        lines += [
            f"## {r['concept']}",
            "",
            f"*{r['status']}* · file `research/rubrics/{r['id']}.json`",
            "",
            "| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |",
            "|---|---|---|",
        ]
        for k in r["key_points"]:
            lines.append(f"| {k['id']} | {_cell(k['text'])} | {_cell(k.get('partial_if', ''))} |")
        lines += ["", "| # | Misconception (flag if stated) |", "|---|---|"]
        for m in r["misconceptions"]:
            lines.append(f"| {m['id']} | {_cell(m['text'])} |")
        lines += ["", f"Maximum: {2 * len(r['key_points'])} marks.", ""]
    lines += [
        "## Review record",
        "",
        "| Reviewer | Date | Rubrics reviewed | Outcome (approved / approved with changes / not approved) |",
        "|---|---|---|---|",
        "| | | | |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    REVIEW_FILE.write_text(render_markdown(load_all()), encoding="utf-8")
    print(f"Wrote {REVIEW_FILE}")


if __name__ == "__main__":
    main()
