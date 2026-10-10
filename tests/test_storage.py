"""
End-to-end /api/evaluate test on a temporary SQLite database. Transcription,
audio features, Sentence-BERT and the LLM are stubbed: no downloads, no API calls.
Startup is not run, so the real database is never touched.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app import main
from backend.app.config import settings
from backend.app.database import Base, get_db
from backend.app.models.concept import ReferenceConcept
from backend.app.models.evaluation import Evaluation
from backend.app.services import llm_client, semantic


class StubSbert:
    def encode(self, texts):
        return [np.array([0.6, 0.8]), np.array([1.0, 0.0])]  # cos = 0.6


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def client(session_factory, monkeypatch, tmp_path):
    db = session_factory()
    db.add(ReferenceConcept(
        id=1, title="DBMS", description="A database stores data.",
        keywords=["data", "sql", "tables"],
    ))
    db.commit()
    db.close()

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "transcribe_audio", lambda path: {
        "text": "A database is queried with SQL using tables.",
        "asr_model": "whisper-small", "asr_confidence": -0.25,
    })
    monkeypatch.setattr(main, "analyze_audio_features", lambda path, transcript: {
        "duration": 30.0, "rms_energy": 0.05, "pause_ratio": 0.5, "pause_ratio_v1": 0.4,
        "filler_words_count": 1, "filler_words_ratio": 2.0, "filler_words_list": ["um: 1"],
    })
    monkeypatch.setattr(semantic, "get_sbert_model", lambda: StubSbert())
    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    monkeypatch.setattr(llm_client, "generate_text", lambda prompt, **kw: "LLM feedback")
    monkeypatch.setattr(llm_client, "model_name", lambda: "gemini-test-model")

    main.app.dependency_overrides[get_db] = override_get_db
    yield TestClient(main.app)  # no `with`: startup (real DB) is not run
    main.app.dependency_overrides.clear()


def test_evaluate_stores_every_sub_score(client, session_factory):
    response = client.post(
        "/api/evaluate", data={"concept_id": "1"},
        files={"file": ("clip.wav", b"fake audio", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    db = session_factory()
    row = db.query(Evaluation).one()
    db.close()

    expected = {
        "semantic_score": 60.0,
        "cosine_similarity": 0.6,
        # "data" must not match inside "database"; "tables" matches "tables"
        "keyword_coverage": pytest.approx(66.7),
        "mentioned_keywords": ["sql", "tables"],
        "missing_keywords": ["data"],
        "understanding_score": pytest.approx(0.7 * 60 + 0.3 * 66.7, abs=0.1),
        "semantic_method": "v1",
        # pause_ratio_v1 0.4 -> deduct (0.4 - 0.3) * 100 = 10
        "fluency_score": 90.0,
        "fluency_method": "v1",
        "feedback": "LLM feedback",
        "feedback_source": "llm",
        "llm_model": "gemini-test-model",
        "asr_model": "whisper-small",
        "pause_ratio": 0.5,
        "pause_ratio_v1": 0.4,
    }
    for field, value in expected.items():
        assert getattr(row, field) == value, field
        assert body[field] == value, field

    assert row.final_score == pytest.approx(0.7 * row.understanding_score + 0.3 * row.fluency_score, abs=0.1)


OLD_SCHEMA = """
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY, concept_id INTEGER NOT NULL, transcription TEXT NOT NULL,
    semantic_score FLOAT NOT NULL, filler_words_count INTEGER, filler_words_ratio FLOAT,
    pause_ratio FLOAT, rms_energy FLOAT, speech_duration FLOAT,
    understanding_grade VARCHAR(50) NOT NULL, feedback TEXT NOT NULL,
    audio_path VARCHAR(500), created_at DATETIME
)
"""


def test_migration_adds_columns_and_old_rows_get_null():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text(OLD_SCHEMA))
        conn.execute(text(
            "INSERT INTO evaluations (concept_id, transcription, semantic_score, understanding_grade, feedback) "
            "VALUES (1, 'old', 60, 'Moderate', 'fb')"
        ))
    factory = sessionmaker(bind=engine)

    main.migrate_evaluations_table(factory)
    main.migrate_evaluations_table(factory)  # running twice is a no-op

    columns = {c["name"] for c in inspect(engine).get_columns("evaluations")}
    assert set(main.ADDED_EVALUATION_COLUMNS) <= columns

    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM evaluations")).mappings().one()
    for column in main.ADDED_EVALUATION_COLUMNS:
        if column != "final_score":  # has DEFAULT 0.0
            assert row[column] is None, column


def test_preload_models_loads_each_and_never_raises():
    calls = []

    def ok():
        calls.append("ok")

    def broken():
        calls.append("broken")
        raise RuntimeError("download failed")

    main.preload_models({"A": ok, "B": broken, "C": ok})  # must not raise
    assert calls == ["ok", "broken", "ok"]


def test_preload_enabled_by_default():
    assert type(settings).model_fields["PRELOAD_MODELS"].default is True
