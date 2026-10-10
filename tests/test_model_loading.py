import sys
import types

import pytest

from backend.app.config import settings
from backend.app.services import semantic


class FakeSentenceTransformer:
    calls = []
    cached = True

    def __init__(self, name, device=None, local_files_only=False):
        FakeSentenceTransformer.calls.append({"name": name, "device": device, "local_files_only": local_files_only})
        if local_files_only and not FakeSentenceTransformer.cached:
            raise OSError("not in local cache")


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(semantic, "_sbert_model", None)
    monkeypatch.setattr(settings, "DEVICE", "cpu")
    FakeSentenceTransformer.calls = []
    FakeSentenceTransformer.cached = True
    return FakeSentenceTransformer


def test_sbert_loads_from_local_cache_only(fake_st):
    semantic.get_sbert_model()
    assert fake_st.calls == [{"name": "all-MiniLM-L6-v2", "device": "cpu", "local_files_only": True}]


def test_sbert_downloads_only_when_not_cached(fake_st):
    fake_st.cached = False
    semantic.get_sbert_model()
    assert [c["local_files_only"] for c in fake_st.calls] == [True, False]


def test_sbert_loaded_once(fake_st):
    first = semantic.get_sbert_model()
    assert semantic.get_sbert_model() is first
    assert len(fake_st.calls) == 1


def test_sbert_uses_configured_device(fake_st, monkeypatch):
    from backend.app.services import device
    monkeypatch.setattr(settings, "DEVICE", "auto")
    monkeypatch.setattr(device, "cuda_available", lambda: True)
    semantic.get_sbert_model()
    assert fake_st.calls[0]["device"] == "cuda"
