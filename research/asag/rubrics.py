"""
Analytic rubrics: key points scored 0/1/2 plus common misconceptions.

The same format is used for the app's concept rubrics (research/rubrics/*.json) and for the
per-question rubrics generated for a dataset pilot. Dataset rubrics are generated ONLY from the
question and the reference answer: generate_rubric() takes no student answers or labels.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

# Per key point: 0 = missing or wrong, 1 = partly stated or vague, 2 = clearly and correctly stated
SCALE = {0: "missing or wrong", 1: "partly stated or vague", 2: "clearly and correctly stated"}


class KeyPoint(BaseModel):
    id: str = Field(description="K1, K2, ...")
    text: str = Field(description="One idea a complete answer must contain, as a short statement")


class Misconception(BaseModel):
    id: str = Field(description="M1, M2, ...")
    text: str = Field(description="A specific wrong belief a student might state, as a short statement")


class Rubric(BaseModel):
    key_points: List[KeyPoint]
    misconceptions: List[Misconception]


# JSON schema sent to Gemini for dataset rubrics. Short reference answers may only support 1-2 key points.
GENERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "key_points": {
            "type": "array", "minItems": 1, "maxItems": 6,
            "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                      "required": ["id", "text"]},
        },
        "misconceptions": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                      "required": ["id", "text"]},
        },
    },
    "required": ["key_points", "misconceptions"],
}

GENERATION_SYSTEM = (
    "You write grading rubrics for short-answer questions in computer science and engineering courses. "
    "You only see the question and the instructor's reference answer."
)

GENERATION_PROMPT = """Write an analytic rubric for this question.

Question:
{question}

Reference answer:
{reference}

Rules:
- Key points: the separate ideas a fully correct answer must contain. Take them from the reference
  answer only; do not add requirements it does not state. Use 1 key point per distinct idea,
  at most 6. Each is one short statement that a grader can check, numbered K1, K2, ...
- Misconceptions: 1 to 3 specific wrong beliefs a student is likely to state for this question,
  numbered M1, M2, ... Each must contradict the reference answer, not just omit something.
- Each key point will be scored 0 (missing or wrong), 1 (partly stated or vague) or 2 (clearly and correctly stated).
Return JSON only."""


def generation_prompt(question: str, reference: str) -> str:
    return GENERATION_PROMPT.format(question=question.strip(), reference=reference.strip())


def generate_rubric(question: str, reference: str, llm) -> Dict[str, Any]:
    """
    One rubric from the question and reference answer only. `llm` is a CachedLLM.
    Returns {"key_points": [...], "misconceptions": [...]}, validated and with ids renumbered.
    """
    data = llm.generate_json(generation_prompt(question, reference), schema=GENERATION_SCHEMA, system=GENERATION_SYSTEM)
    return _renumber(Rubric.model_validate(data))


def _renumber(rubric: Rubric) -> Dict[str, Any]:
    if not rubric.key_points:
        raise ValueError("Rubric has no key points")
    return {
        "key_points": [{"id": f"K{i}", "text": k.text.strip()} for i, k in enumerate(rubric.key_points, 1)],
        "misconceptions": [{"id": f"M{i}", "text": m.text.strip()} for i, m in enumerate(rubric.misconceptions, 1)],
    }


# ---------------------------------------------------------------- batched generation (free-tier quota)
BATCH_SIZE = 10

BATCH_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "the label, e.g. Q1"},
                       **GENERATION_SCHEMA["properties"]},
        "required": ["question", "key_points", "misconceptions"],
    },
}

BATCH_PROMPT = """Write an analytic rubric for each of the {n} questions below. Treat every question
separately: a rubric uses only its own question and reference answer.

{questions}

Rules for every rubric:
- Key points: the separate ideas a fully correct answer must contain. Take them from the reference
  answer only; do not add requirements it does not state. Use 1 key point per distinct idea,
  at most 6. Each is one short statement that a grader can check, numbered K1, K2, ...
- Misconceptions: 1 to 3 specific wrong beliefs a student is likely to state for this question,
  numbered M1, M2, ... Each must contradict the reference answer, not just omit something.
- Each key point will be scored 0 (missing or wrong), 1 (partly stated or vague) or 2 (clearly and correctly stated).
Return a JSON array with exactly one rubric per question, each with "question" set to its label ({labels})."""


def batches(question_ids: List[str], size: int = BATCH_SIZE) -> List[List[str]]:
    """Sorted ids split into ceil(n / size) batches of near-equal size (81 -> 9 x 9), so batches are fixed by the sample."""
    ids = sorted(question_ids)
    if not ids:
        return []
    n = -(-len(ids) // size)
    bounds = [round(i * len(ids) / n) for i in range(n + 1)]
    return [ids[bounds[i]:bounds[i + 1]] for i in range(n)]


def batch_prompt(questions: List[Dict[str, str]]) -> str:
    """questions: [{"question", "reference"}] in batch order; labelled Q1..Qn (no dataset ids in the prompt)."""
    blocks = [f"Q{i}\nQuestion:\n{q['question'].strip()}\nReference answer:\n{q['reference'].strip()}"
              for i, q in enumerate(questions, 1)]
    labels = ", ".join(f"Q{i}" for i in range(1, len(questions) + 1))
    return BATCH_PROMPT.format(n=len(questions), questions="\n\n".join(blocks), labels=labels)


def generate_rubric_batch(questions: List[Dict[str, str]], llm) -> List[Dict[str, Any]]:
    """
    Rubrics for several questions in one call, from each question and reference answer only.
    Returns one rubric per input question, in input order; raises if any label is missing or repeated.
    """
    data = llm.generate_json(batch_prompt(questions), schema=BATCH_SCHEMA, system=GENERATION_SYSTEM)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of rubrics")
    labels = [str(item.get("question", "")).strip() for item in data]
    expected = [f"Q{i}" for i in range(1, len(questions) + 1)]
    if sorted(labels) != sorted(expected):
        raise ValueError(f"Expected rubrics for {expected}, got {labels}")
    by_label = {label: item for label, item in zip(labels, data)}
    return [_renumber(Rubric.model_validate(by_label[label])) for label in expected]


def save_rubric(path: Path, question_id: str, rubric: Dict[str, Any], model: str, prompt_key: str) -> None:
    """
    Saves a dataset rubric. The question and reference text are NOT copied into the file
    (dataset licences may forbid redistribution); prompt_key links it to the cached prompt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"question_id": question_id, "generated_from": "question + reference answer only",
              "model": model, "prompt_key": prompt_key, "scale": SCALE, **rubric}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rubric(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    Rubric.model_validate(data)
    return data
