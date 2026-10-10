"""Safety net for the whole suite: tests must never call the real Gemini API."""
import pytest

from backend.app.services import llm_client


@pytest.fixture(autouse=True)
def no_real_gemini(monkeypatch):
    """Blocks the real google-genai client. Tests that need one install a fake with monkeypatch."""
    def blocked():
        raise AssertionError("A test tried to create the real Gemini client; stub llm_client instead.")
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.setattr(llm_client, "_get_client", blocked)
