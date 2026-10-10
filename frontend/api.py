"""
Backend API calls for the Streamlit frontend. Every failure is raised as
BackendError with a short message for the user and the technical detail kept
separately, so the UI never shows raw exception text as the main message.
"""
import os
from typing import Any, Dict, List

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Whisper small on CPU plus Gemini can take several minutes for a long recording,
# and the first run also downloads models.
EVALUATE_TIMEOUT_S = 600


class BackendError(Exception):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


def _request(method: str, path: str, *, timeout: float, **kwargs) -> requests.Response:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as e:
        raise BackendError("Cannot reach the analysis server. Make sure the backend is running.", str(e)) from e
    except requests.exceptions.Timeout as e:
        raise BackendError(
            "The server is taking longer than expected. The result may still be saved: "
            "check History in a minute.",
            str(e),
        ) from e
    except requests.exceptions.RequestException as e:
        raise BackendError("The request to the analysis server failed.", str(e)) from e
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(_status_message(response.status_code), f"HTTP {response.status_code}: {detail}")
    return response


def _status_message(status: int) -> str:
    if status == 404:
        return "The requested item was not found."
    if status in (400, 422):
        return "The server could not process this request. Check the inputs and try again."
    return "The analysis failed on the server. Please try again."


def get_concepts() -> List[Dict[str, Any]]:
    return _request("GET", "/api/concepts", timeout=10).json()


def create_concept(title: str, description: str, keywords: List[str]) -> Dict[str, Any]:
    payload = {"title": title, "description": description, "keywords": keywords}
    return _request("POST", "/api/concepts", json=payload, timeout=10).json()


def evaluate(concept_id: int, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    ext = os.path.splitext(filename)[1].lstrip(".").lower() or "wav"
    mime = {"mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext, f"audio/{ext}")
    return _request(
        "POST", "/api/evaluate",
        files={"file": (filename, audio_bytes, mime)},
        data={"concept_id": concept_id},
        timeout=EVALUATE_TIMEOUT_S,
    ).json()


def get_evaluations() -> List[Dict[str, Any]]:
    return _request("GET", "/api/evaluations", timeout=10).json()


def get_evaluation(eval_id: int) -> Dict[str, Any]:
    return _request("GET", f"/api/evaluations/{int(eval_id)}", timeout=10).json()


def get_pdf(eval_id: int) -> bytes:
    return _request("GET", f"/api/evaluations/{int(eval_id)}/pdf", timeout=30).content
