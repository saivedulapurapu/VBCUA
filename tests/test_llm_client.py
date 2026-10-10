from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.services import llm_client


class FakeModels:
    def __init__(self, text="hello", fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.fail:
            raise RuntimeError("api down")
        return SimpleNamespace(text=self.text)


class FakeFiles:
    def __init__(self):
        self.uploaded = []
        self.deleted = []

    def upload(self, *, file):
        self.uploaded.append(file)
        return SimpleNamespace(name="files/abc")

    def delete(self, *, name):
        self.deleted.append(name)


class FakeClient:
    def __init__(self, **model_kwargs):
        self.models = FakeModels(**model_kwargs)
        self.files = FakeFiles()


@pytest.fixture
def client(monkeypatch):
    """Installs a fake google-genai client; call it to change its responses."""
    holder = {"client": FakeClient()}

    def configure(**kwargs):
        holder["client"] = FakeClient(**kwargs)
        return holder["client"]

    monkeypatch.setattr(llm_client, "_get_client", lambda: holder["client"])
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-test-model")
    configure.get = lambda: holder["client"]
    return configure


class Verdict(BaseModel):
    score: int
    reason: str


def test_default_model_is_gemini_3_8_flash():
    assert type(settings).model_fields["GEMINI_MODEL"].default == "gemini-3.8-flash"


def test_generate_text_uses_configured_model(client):
    fake = client(text="  some feedback  ")
    assert llm_client.generate_text("prompt", system="be kind", temperature=0.2) == "some feedback"
    call = fake.models.calls[0]
    assert call["model"] == "gemini-test-model"
    assert call["contents"] == "prompt"
    assert call["config"] == {"system_instruction": "be kind", "temperature": 0.2}


def test_generate_json_with_pydantic_schema(client):
    fake = client(text='{"score": 2, "reason": "covers key point"}')
    result = llm_client.generate_json("grade this", schema=Verdict)
    assert result == {"score": 2, "reason": "covers key point"}
    config = fake.models.calls[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == Verdict.model_json_schema()
    assert config["temperature"] == 0.0


def test_generate_json_with_dict_schema(client):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    fake = client(text='{"ok": true}')
    assert llm_client.generate_json("p", schema=schema) == {"ok": True}
    assert fake.models.calls[0]["config"]["response_json_schema"] == schema


def test_invalid_json_raises(client):
    client(text="not json")
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_json("p")


def test_json_not_matching_schema_raises(client):
    client(text='{"score": "high"}')
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_json("p", schema=Verdict)


def test_api_failure_becomes_llm_error(client):
    client(fail=True)
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")


def test_empty_response_raises(client):
    client(text="   ")
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")


def test_no_key_raises_before_creating_client(monkeypatch):
    def must_not_be_called():
        raise AssertionError("client should not be created without a key")

    monkeypatch.setattr(llm_client, "_get_client", must_not_be_called)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    assert llm_client.is_configured() is False
    with pytest.raises(llm_client.LLMNotConfigured):
        llm_client.generate_text("p")
    with pytest.raises(llm_client.LLMNotConfigured):
        llm_client.generate_from_audio("clip.wav", "transcribe")


def test_audio_upload_deleted_after_success(client):
    fake = client(text="transcript")
    assert llm_client.generate_from_audio("clip.wav", "transcribe") == "transcript"
    assert fake.files.uploaded == ["clip.wav"]
    assert fake.files.deleted == ["files/abc"]


def test_audio_upload_deleted_after_failure(client):
    fake = client(fail=True)
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_from_audio("clip.wav", "transcribe")
    assert fake.files.deleted == ["files/abc"]


# --- Retries on temporary Gemini errors (real google-genai error classes) ---

from google.genai import errors as genai_errors  # noqa: E402


class FlakyModels:
    """Raises the given errors in order, then returns text."""

    def __init__(self, failures, text="recovered"):
        self.failures = list(failures)
        self.text = text
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(text=self.text)


def server_error(code):
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}})


@pytest.fixture
def flaky(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_client, "_sleep", sleeps.append)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key")

    def install(failures):
        models = FlakyModels(failures)
        monkeypatch.setattr(llm_client, "_get_client", lambda: SimpleNamespace(models=models))
        return models

    install.sleeps = sleeps
    return install


@pytest.mark.parametrize("code", [503, 429, 500])
def test_retries_temporary_errors_then_succeeds(flaky, code):
    models = flaky([server_error(code), server_error(code)])
    assert llm_client.generate_text("p") == "recovered"
    assert models.calls == 3
    assert flaky.sleeps == list(llm_client.RETRY_DELAYS_S)


def test_gives_up_after_retries(flaky):
    models = flaky([server_error(503)] * 5)
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")
    assert models.calls == 1 + len(llm_client.RETRY_DELAYS_S)


@pytest.mark.parametrize("error", [server_error(400), server_error(403), RuntimeError("bug")])
def test_does_not_retry_permanent_errors(flaky, error):
    models = flaky([error])
    with pytest.raises(llm_client.LLMError):
        llm_client.generate_text("p")
    assert models.calls == 1
    assert flaky.sleeps == []
