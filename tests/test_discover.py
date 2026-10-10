"""Trending topics and AI-written references. Gemini is stubbed: no network, no API key."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app import main
from backend.app.database import Base, get_db
from backend.app.models.concept import ReferenceConcept, TrendingTopics
from backend.app.services import discover, llm_client

DAY = "2026-10-10"
TOPICS = [
    {"title": "Nobel Prize in Physics", "category": "science", "why": "Announced this week."},
    {"title": "Cricket World Cup final", "category": "Sports", "why": "Played today."},
    {"title": "Interest rate decision", "category": "Business", "why": "Central bank meets."},
    {"title": "Nobel prize in physics", "category": "Science", "why": "duplicate"},
    {"title": "", "category": "News", "why": "empty"},
    {"title": "New phone launch", "category": "Gadgets", "why": "Unknown category."},
]
GROUNDING = {
    "sources": [{"title": "Example News", "uri": "https://news.example/a"}],
    "search_entry_point": "<div>Google search suggestions</div>",
}
REFERENCE = {
    "title": "Nobel Prize in Physics 2026",
    "description": "The Nobel Prize in Physics is awarded each year for outstanding contributions to physics. " * 2,
    "keywords": ["Nobel Prize", "physics", "Royal Swedish Academy", "laureates", "discovery", "physics"],
}


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def gemini(monkeypatch):
    state = {"configured": True, "calls": [], "answers": [], "fail": False}

    def fake(prompt, *, model=None, thinking_level=None):
        state["calls"].append({"prompt": prompt, "model": model})
        if state["fail"]:
            raise llm_client.LLMError("503 UNAVAILABLE")
        return state["answers"].pop(0)

    monkeypatch.setattr(llm_client, "is_configured", lambda: state["configured"])
    monkeypatch.setattr(llm_client, "generate_grounded_json", fake)
    monkeypatch.setattr(discover, "_last_failure", {})
    return state


# --- Trending topics ---

def test_curated_topics_without_key(db, gemini):
    gemini["configured"] = False
    result = discover.get_trending(db, DAY)
    assert result["origin"] == "curated"
    assert len(result["topics"]) == discover.TOPIC_COUNT
    assert gemini["calls"] == []
    assert discover.get_trending(db, DAY)["topics"] == result["topics"]  # stable for the day


def test_grounded_topics_are_cleaned_and_cached(db, gemini):
    gemini["answers"] = [(TOPICS, GROUNDING)]
    first = discover.get_trending(db, DAY)
    assert first["origin"] == "google-search"
    assert [t["title"] for t in first["topics"]] == ["Nobel Prize in Physics", "Cricket World Cup final", "Interest rate decision", "New phone launch"]
    assert first["topics"][0]["category"] == "Science"
    assert first["topics"][3]["category"] == "News"  # unknown category
    assert first["sources"] == GROUNDING["sources"]
    assert first["search_entry_point"] == GROUNDING["search_entry_point"]
    assert DAY in gemini["calls"][0]["prompt"]
    # Cached: a second request does not call Gemini again
    assert discover.get_trending(db, DAY) == first
    assert len(gemini["calls"]) == 1
    assert db.get(TrendingTopics, DAY).origin == "google-search"


def test_failure_falls_back_and_waits_before_retrying(db, gemini):
    gemini["fail"] = True
    assert discover.get_trending(db, DAY)["origin"] == "curated"
    assert discover.get_trending(db, DAY)["origin"] == "curated"
    assert len(gemini["calls"]) == 1  # not retried within 30 minutes
    assert db.get(TrendingTopics, DAY) is None  # the fallback is never cached as trending


def test_ungrounded_answer_is_not_called_trending(db, gemini):
    gemini["answers"] = [(TOPICS, {"sources": [], "search_entry_point": None})]
    assert discover.get_trending(db, DAY)["origin"] == "curated"


def test_too_few_topics_rejected():
    with pytest.raises(llm_client.LLMError):
        discover.clean_topics([{"title": "Only one", "category": "News"}])
    with pytest.raises(llm_client.LLMError):
        discover.clean_topics("not a list")


def test_topics_object_form_accepted():
    assert len(discover.clean_topics({"topics": TOPICS})) == 4


# --- AI-written references ---

def test_generate_concept_creates_ai_generated(db, gemini):
    gemini["answers"] = [(REFERENCE, GROUNDING)]
    concept = discover.generate_concept(db, "  Nobel   Prize in Physics ")
    assert concept.source == "ai-generated"
    assert concept.title == "Nobel Prize in Physics 2026"
    assert concept.keywords == ["Nobel Prize", "physics", "Royal Swedish Academy", "laureates", "discovery"]  # deduplicated
    assert "Nobel   Prize" not in gemini["calls"][0]["prompt"]  # whitespace normalised


def test_generate_concept_reuses_existing_title(db, gemini):
    db.add(ReferenceConcept(title="Cloud Computing", description="d", keywords=["x"], source="library"))
    db.commit()
    concept = discover.generate_concept(db, "cloud computing")
    assert concept.title == "Cloud Computing"
    assert gemini["calls"] == []


@pytest.mark.parametrize("topic", ["", "x", "y" * 121])
def test_generate_concept_rejects_bad_topics(db, gemini, topic):
    with pytest.raises(ValueError):
        discover.generate_concept(db, topic)


def test_generate_concept_rejects_incomplete_reference(db, gemini):
    gemini["answers"] = [({"title": "T", "description": "too short", "keywords": ["a"]}, GROUNDING)]
    with pytest.raises(llm_client.LLMError):
        discover.generate_concept(db, "Some topic")
    assert db.query(ReferenceConcept).count() == 0


# --- Endpoints ---

@pytest.fixture
def client(db):
    def override():
        yield db
    main.app.dependency_overrides[get_db] = override
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_trending_endpoint(client, gemini):
    gemini["configured"] = False
    body = client.get("/api/discover/trending").json()
    assert body["origin"] == "curated" and len(body["topics"]) == discover.TOPIC_COUNT


def test_generate_endpoint_success(client, gemini):
    gemini["answers"] = [(REFERENCE, GROUNDING)]
    body = client.post("/api/concepts/generate", json={"topic": "Nobel Prize in Physics"}).json()
    assert body["source"] == "ai-generated" and body["id"]


def test_generate_endpoint_errors(client, gemini, monkeypatch):
    assert client.post("/api/concepts/generate", json={"topic": "x"}).status_code == 422
    gemini["fail"] = True

    def plain_fails(*args, **kwargs):
        raise llm_client.LLMError("429 quota")

    monkeypatch.setattr(llm_client, "generate_json", plain_fails)
    assert client.post("/api/concepts/generate", json={"topic": "Some topic"}).status_code == 502

    def not_configured(*args, **kwargs):
        raise llm_client.LLMNotConfigured("no key")

    monkeypatch.setattr(llm_client, "generate_grounded_json", not_configured)
    assert client.post("/api/concepts/generate", json={"topic": "Some topic"}).status_code == 503


def test_custom_concepts_marked_custom(client):
    body = client.post("/api/concepts", json={"title": "REST API", "description": "d" * 20, "keywords": ["http"]}).json()
    assert body["source"] == "custom"


# --- Grounding helpers ---

def test_parse_json_text_variants():
    assert llm_client.parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm_client.parse_json_text('Sure! [1, 2] Hope that helps.') == [1, 2]
    with pytest.raises(llm_client.LLMError):
        llm_client.parse_json_text("no json here")


def test_grounding_extracts_unique_sources_and_entry_point():
    web = lambda uri, title=None: SimpleNamespace(web=SimpleNamespace(uri=uri, title=title, domain="example.com"))
    response = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=SimpleNamespace(
        grounding_chunks=[web("https://a", "A"), web("https://a", "A again"), web("https://b"), SimpleNamespace(web=None)],
        search_entry_point=SimpleNamespace(rendered_content="<div>chips</div>"),
    ))])
    g = llm_client._grounding(response)
    assert g["sources"] == [{"title": "A", "uri": "https://a"}, {"title": "example.com", "uri": "https://b"}]
    assert g["search_entry_point"] == "<div>chips</div>"
    assert llm_client._grounding(SimpleNamespace(candidates=[])) == {"sources": [], "search_entry_point": None}


def test_grounded_call_uses_search_tool(monkeypatch):
    seen = {}

    class Models:
        def generate_content(self, *, model, contents, config):
            seen.update(model=model, config=config)
            return SimpleNamespace(text='[{"title": "x"}]', candidates=[])

    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    monkeypatch.setattr(llm_client, "_get_client", lambda: SimpleNamespace(models=Models()))
    data, grounding = llm_client.generate_grounded_json("p", model="m")
    assert data == [{"title": "x"}]
    assert seen["config"]["tools"] == [{"google_search": {}}]
    assert "response_mime_type" not in seen["config"]


# --- Migration ---

def test_migration_adds_concept_source_column():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE reference_concepts (id INTEGER PRIMARY KEY, title VARCHAR(100), keywords JSON, description TEXT, created_at DATETIME)"))
        conn.execute(text("CREATE TABLE evaluations (id INTEGER PRIMARY KEY, concept_id INTEGER, transcription TEXT, semantic_score FLOAT, understanding_grade VARCHAR(50), feedback TEXT)"))
    main.migrate_evaluations_table(sessionmaker(bind=engine))
    assert "source" in {c["name"] for c in inspect(engine).get_columns("reference_concepts")}


def test_reference_falls_back_to_ungrounded_when_search_fails(db, gemini, monkeypatch):
    gemini["fail"] = True
    plain_calls = []

    def fake_json(prompt, **kwargs):
        plain_calls.append({"prompt": prompt, **kwargs})
        return REFERENCE

    monkeypatch.setattr(llm_client, "generate_json", fake_json)
    concept = discover.generate_concept(db, "Nobel Prize in Physics")
    assert concept.source == "ai-generated"
    assert len(gemini["calls"]) == 1 and len(plain_calls) == 1
    assert "Google Search" not in plain_calls[0]["prompt"]
    assert plain_calls[0]["model"] == discover.settings.GEMINI_DISCOVER_MODEL


def test_quota_errors_are_not_retried(monkeypatch):
    from google.genai import errors as genai_errors
    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            raise genai_errors.ClientError(429, {"error": {"code": 429, "message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    monkeypatch.setattr(llm_client, "_get_client", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")
    assert len(calls) == 1


def test_generate_endpoint_reports_exhausted_quota(client, gemini, monkeypatch):
    from google.genai import errors as genai_errors
    quota = genai_errors.ClientError(429, {"error": {"code": 429, "message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED"}})

    def fail(*args, **kwargs):
        raise llm_client.LLMError("gemini call failed: 429 quota") from quota

    monkeypatch.setattr(llm_client, "generate_grounded_json", fail)
    monkeypatch.setattr(llm_client, "generate_json", fail)
    response = client.post("/api/concepts/generate", json={"topic": "T20 Cricket"})
    assert response.status_code == 429
    assert "quota" in response.json()["detail"]
