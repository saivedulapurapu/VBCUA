import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontend"))
import api  # noqa: E402


class FakeResponse:
    def __init__(self, status, payload=None, text="", content=b""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def install(result):
        def fake_request(method, url, timeout, **kwargs):
            recorded.append(SimpleNamespace(method=method, url=url, timeout=timeout, kwargs=kwargs))
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(api.requests, "request", fake_request)
        return recorded

    return install


def test_connection_error_has_friendly_message(calls):
    calls(requests.exceptions.ConnectionError("refused"))
    with pytest.raises(api.BackendError) as info:
        api.get_concepts()
    assert "backend is running" in info.value.message
    assert "refused" in info.value.detail


def test_timeout_points_to_history(calls):
    calls(requests.exceptions.ReadTimeout("read timeout=600"))
    with pytest.raises(api.BackendError) as info:
        api.evaluate(1, b"x", "clip.wav")
    assert "History" in info.value.message
    assert "HTTPConnectionPool" not in info.value.message


def test_server_error_keeps_detail_out_of_message(calls):
    calls(FakeResponse(500, {"detail": "Transcription failed: [WinError 2]"}))
    with pytest.raises(api.BackendError) as info:
        api.evaluate(1, b"x", "clip.wav")
    assert "WinError" not in info.value.message
    assert "WinError" in info.value.detail


def test_evaluate_uses_long_timeout_and_mime(calls):
    recorded = calls(FakeResponse(200, {"id": 1}))
    assert api.evaluate(3, b"audio", "talk.mp3") == {"id": 1}
    call = recorded[0]
    assert call.timeout == api.EVALUATE_TIMEOUT_S >= 600
    assert call.kwargs["files"]["file"] == ("talk.mp3", b"audio", "audio/mpeg")
    assert call.kwargs["data"] == {"concept_id": 3}


def test_get_pdf_returns_bytes(calls):
    calls(FakeResponse(200, content=b"%PDF"))
    assert api.get_pdf(4) == b"%PDF"
