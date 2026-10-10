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
