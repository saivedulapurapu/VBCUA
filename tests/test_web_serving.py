"""The backend serves the React build from web/dist (no separate frontend server)."""
import pytest
from fastapi.testclient import TestClient

from backend.app import main


@pytest.fixture
def dist(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (tmp_path / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(dist, monkeypatch):
    monkeypatch.setattr(main, "WEB_DIST", dist)
    return TestClient(main.app)  # no `with`: startup (real DB) is not run


def test_root_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "id=root" in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_client_side_routes_fall_back_to_index(client):
    assert "id=root" in client.get("/history/7").text


def test_hashed_assets_are_cached(client):
    r = client.get("/assets/app-abc123.js")
    assert r.text == "console.log(1)"
    assert "immutable" in r.headers["cache-control"]


def test_other_static_files_revalidate(client):
    assert client.get("/favicon.svg").headers["cache-control"] == "no-cache"


def test_api_paths_are_not_swallowed(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/does-not-exist").status_code == 404


def test_path_traversal_returns_index_not_outside_file(client, dist):
    (dist.parent / "secret.txt").write_text("secret", encoding="utf-8")
    r = client.get("/..%2Fsecret.txt")
    assert "secret" not in r.text


def test_resolve_web_file_never_leaves_dist(dist):
    assert main.resolve_web_file(dist, "../secret.txt") == dist / "index.html"
    assert main.resolve_web_file(dist, "assets/app-abc123.js") == (dist / "assets" / "app-abc123.js").resolve()


def test_not_built_gives_clear_404(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "WEB_DIST", tmp_path / "missing")
    r = TestClient(main.app).get("/")
    assert r.status_code == 404
    assert "npm run build" in r.json()["detail"]
