import pytest

from backend.app.config import settings
from backend.app.services import speech_to_text as stt


class StubWhisper:
    def __init__(self, segments=None, fail=False):
        self.segments = segments if segments is not None else [{"avg_logprob": -0.2}, {"avg_logprob": -0.4}]
        self.fail = fail
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("whisper broke")
        return {"text": "  hello world  ", "segments": self.segments}


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"not really audio")
    return str(path)


@pytest.fixture
def env(monkeypatch):
    """Stub out model loading and Gemini; record which models were requested."""
    state = {"loaded": [], "whisper": StubWhisper(), "gemini_calls": 0}

    def fake_get_whisper_model(model_name=None):
        state["loaded"].append(model_name)
        return state["whisper"]

    def fake_gemini(audio_path, api_key):
        state["gemini_calls"] += 1
        return "gemini text"

    monkeypatch.setattr(stt, "get_whisper_model", fake_get_whisper_model)
    monkeypatch.setattr(stt, "transcribe_audio_gemini", fake_gemini)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "small")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "STT_GEMINI_FALLBACK", False)
    return state


def test_default_whisper_model_is_small():
    assert type(settings).model_fields["WHISPER_MODEL"].default == "small"
    assert type(settings).model_fields["STT_GEMINI_FALLBACK"].default is False


def test_whisper_used_even_with_gemini_key(env, audio_file):
    result = stt.transcribe_audio(audio_file)
    assert result["text"] == "hello world"
    assert result["asr_model"] == "whisper-small"
    assert env["gemini_calls"] == 0
    assert env["whisper"].calls[0] == {"language": "en", "fp16": False}


def test_asr_confidence_is_mean_avg_logprob(env, audio_file):
    assert stt.transcribe_audio(audio_file)["asr_confidence"] == pytest.approx(-0.3)


def test_asr_confidence_none_without_segments(env, audio_file):
    env["whisper"] = StubWhisper(segments=[])
    assert stt.transcribe_audio(audio_file)["asr_confidence"] is None


def test_model_size_override(env, audio_file):
    result = stt.transcribe_audio_whisper(audio_file, model_name="tiny")
    assert env["loaded"] == ["tiny"]
    assert result["asr_model"] == "whisper-tiny"


def test_whisper_failure_raises_when_fallback_off(env, audio_file):
    env["whisper"] = StubWhisper(fail=True)
    with pytest.raises(RuntimeError):
        stt.transcribe_audio(audio_file)
    assert env["gemini_calls"] == 0


def test_whisper_failure_uses_gemini_when_fallback_on(env, audio_file, monkeypatch):
    monkeypatch.setattr(settings, "STT_GEMINI_FALLBACK", True)
    env["whisper"] = StubWhisper(fail=True)
    result = stt.transcribe_audio(audio_file)
    assert result == {"text": "gemini text", "asr_model": stt.GEMINI_STT_MODEL, "asr_confidence": None}


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        stt.transcribe_audio("does/not/exist.wav")
