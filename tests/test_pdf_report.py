from datetime import datetime
from types import SimpleNamespace

from backend.app.services.pdf_report import generate_pdf_report
from tests.pdf_text import pdf_text


def make_evaluation(**overrides):
    values = dict(
        id=1, concept_id=1, transcription="A transcript.", semantic_score=50.0,
        filler_words_count=3, filler_words_ratio=2.5, pause_ratio=0.5, pause_ratio_v1=0.2,
        rms_energy=0.05, speech_duration=10.0, final_score=50.0,
        understanding_grade="Moderate Understanding", feedback="### Report\n- ok",
        fluency_score=80.0, audio_path=None, created_at=datetime(2026, 10, 10),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def render(tmp_path, evaluation):
    path = tmp_path / "report.pdf"
    generate_pdf_report(evaluation, "Machine Learning", str(path))
    return pdf_text(path.read_bytes())


def test_fillers_marked_experimental(tmp_path):
    text = render(tmp_path, make_evaluation())
    assert b"Filler Word Count (experimental)" in text
    assert b"not scored" in text
    assert b"0 - 2 words" not in text


def test_pdf_shows_v1_pause_ratio(tmp_path):
    text = render(tmp_path, make_evaluation(pause_ratio=0.5, pause_ratio_v1=0.2))
    assert b"20.0%" in text
    assert b"50.0%" in text  # semantic score; pause B0 (50%) is not shown as pause


def test_pdf_falls_back_to_b0_pause_for_old_rows(tmp_path):
    text = render(tmp_path, make_evaluation(pause_ratio=0.37, pause_ratio_v1=None))
    assert b"37.0%" in text


def test_pdf_shows_stored_fluency_score(tmp_path):
    # filler ratio 2.5 would give 90.0% with the old approximation
    text = render(tmp_path, make_evaluation(fluency_score=73.0))
    assert b"73.0%" in text
    assert b"90.0%" not in text


def test_pdf_fluency_na_for_old_rows(tmp_path):
    assert b"n/a" in render(tmp_path, make_evaluation(fluency_score=None))
