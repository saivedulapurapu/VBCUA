import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from backend.app.main import app

ROOT = Path(__file__).resolve().parent.parent
PLANNED_NOT_IMPLEMENTED = ["/register", "/login", "/upload", "/transcribe", "/feedback", "/report"]


def documented_routes(path):
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^\| (GET|POST|PUT|DELETE|PATCH) \| `([^`]+)` \|", text, re.M))


def app_routes():
    return {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }


@pytest.mark.parametrize("doc", ["README.md", "VBCUA_Documents/Scope_and_Changes.md"])
def test_documented_api_matches_app(doc):
    assert documented_routes(ROOT / doc) == app_routes()


def test_readme_does_not_list_planned_endpoints():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for endpoint in PLANNED_NOT_IMPLEMENTED:
        assert f"`{endpoint}`" not in readme
