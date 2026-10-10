import sys
import types

import pytest
from sqlalchemy import create_engine

import run
from backend.app.config import Settings


def test_default_database_is_sqlite():
    assert Settings.model_fields["DATABASE_URL"].default == "sqlite:///./vbcua.db"


@pytest.fixture
def fake_psycopg2(monkeypatch):
    """Records whether run.setup_db tries to connect to PostgreSQL."""
    calls = []
    module = types.ModuleType("psycopg2")

    def connect(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("no server in tests")

    module.connect = connect
    extensions = types.ModuleType("psycopg2.extensions")
    extensions.ISOLATION_LEVEL_AUTOCOMMIT = 0
    monkeypatch.setitem(sys.modules, "psycopg2", module)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", extensions)
    monkeypatch.setattr(run, "load_dotenv", lambda *a, **kw: None)
    return calls


@pytest.mark.parametrize("url", ["sqlite:///./vbcua.db", "sqlite://", "mysql://u:p@h/db"])
def test_setup_db_skips_non_postgres(monkeypatch, fake_psycopg2, url):
    monkeypatch.setenv("DATABASE_URL", url)
    run.setup_db()
    assert fake_psycopg2 == []


def test_setup_db_skips_when_unset(monkeypatch, fake_psycopg2):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    run.setup_db()
    assert fake_psycopg2 == []


@pytest.mark.parametrize("url", [
    "postgresql+psycopg2://alice:secret@db.local:5433/vbcua_db",
    "postgresql://alice:secret@db.local:5433/vbcua_db",
])
def test_setup_db_runs_for_postgres(monkeypatch, fake_psycopg2, url):
    monkeypatch.setenv("DATABASE_URL", url)
    run.setup_db()  # connection error is caught and logged
    assert fake_psycopg2 == [{"dbname": "postgres", "user": "alice", "password": "secret", "host": "db.local", "port": 5433}]


def test_psycopg2_url_uses_installed_driver():
    # Plain postgresql:// selects psycopg (v3) in SQLAlchemy 2.1, which is not installed
    engine = create_engine("postgresql+psycopg2://u:p@localhost:5432/db")  # does not connect
    assert engine.dialect.driver == "psycopg2"


# --- Web app build check in run.py ---

import os  # noqa: E402
import time  # noqa: E402


def make_web(tmp_path, built=True):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.tsx").write_text("x", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    if built:
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "index.html").write_text("<div>", encoding="utf-8")
    return tmp_path


def test_web_needs_build_when_dist_missing(tmp_path):
    assert run.web_needs_build(str(make_web(tmp_path, built=False))) is True


def test_web_up_to_date_after_build(tmp_path):
    web = make_web(tmp_path)
    old = time.time() - 100
    os.utime(web / "src" / "App.tsx", (old, old))
    os.utime(web / "package.json", (old, old))
    assert run.web_needs_build(str(web)) is False


def test_web_needs_build_when_source_is_newer(tmp_path):
    web = make_web(tmp_path)
    old = time.time() - 100
    os.utime(web / "dist" / "index.html", (old, old))
    assert run.web_needs_build(str(web)) is True


def test_build_web_without_npm_uses_existing_build(monkeypatch, tmp_path):
    web = make_web(tmp_path)
    monkeypatch.setattr(run, "WEB_DIR", str(web))
    monkeypatch.setattr(run, "web_needs_build", lambda *a: True)
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    assert run.build_web() is True


def test_build_web_without_npm_or_build_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "WEB_DIR", str(make_web(tmp_path, built=False)))
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    assert run.build_web() is False
