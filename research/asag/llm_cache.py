"""
Disk cache for research LLM calls, so every Gemini answer is paid for once and runs are reproducible.

Key: SHA-256 of (model, system prompt, prompt, JSON schema, temperature). Each answer is one JSON
file under <cache_dir>/<model>/<key>.json together with the prompt that produced it.
Temperature is 0. Quota / rate-limit / overload errors are retried with growing waits; if they
persist the run stops, and re-running resumes from the cache.
A call budget makes sure a run never makes more calls than were approved.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from backend.app.services import llm_client

logger = logging.getLogger(__name__)

TEMPERATURE = 0.0
BACKOFF_S = (15.0, 30.0, 60.0, 120.0, 240.0)
BACKOFF_STATUS = {429, 500, 503}


class BudgetExceeded(RuntimeError):
    """The run would make more LLM calls than allowed."""


class QuotaStop(RuntimeError):
    """The API kept refusing after every backoff; re-run later to resume from the cache."""


def _status(error: BaseException) -> Optional[int]:
    for e in (error, error.__cause__):
        code = getattr(e, "code", None)
        if isinstance(code, int):
            return code
    return None


def is_daily_quota(error: BaseException) -> bool:
    """A 429 for a per-day quota (e.g. GenerateRequestsPerDayPerProjectPerModel): waiting minutes will not help."""
    text = f"{error} {error.__cause__ or ''}".lower()
    return _status(error) == 429 and "perday" in text.replace("_", "").replace("-", "").replace(" ", "")


class CachedLLM:
    def __init__(
        self,
        cache_dir: Path,
        model: Optional[str] = None,
        max_calls: int = 0,
        min_interval_s: float = 0.0,
        call: Optional[Callable[..., Dict[str, Any]]] = None,
        backoff_s: Sequence[float] = BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """
        max_calls: how many uncached prompts this run may send (0 = cache only, no API calls).
        min_interval_s: pause between API calls, to stay under a requests-per-minute limit.
        call: replaces llm_client.generate_json (tests, offline runs).
        """
        self.cache_dir = Path(cache_dir)
        self.model = model or llm_client.model_name()
        self.max_calls = max_calls
        self.min_interval_s = min_interval_s
        self._call = call or llm_client.generate_json
        self.backoff_s = tuple(backoff_s)
        self._sleep = sleep
        self.calls = 0
        self.hits = 0
        self.retries = 0
        self.usage: list = []  # token counts of this run's new calls (empty dicts when unknown)
        self._last_call = 0.0

    def tokens(self) -> Dict[str, int]:
        """Summed token counts of this run's new calls."""
        fields = ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")
        return {f: sum((u.get(f) or 0) for u in self.usage) for f in fields}

    def key(self, prompt: str, schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None) -> str:
        payload = json.dumps(
            {"model": self.model, "system": system, "prompt": prompt, "schema": schema, "temperature": TEMPERATURE},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / self.model / f"{key}.json"

    def is_cached(self, prompt: str, schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None) -> bool:
        return self._path(self.key(prompt, schema, system)).exists()

    def generate_json(self, prompt: str, schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None) -> Dict[str, Any]:
        key = self.key(prompt, schema, system)
        path = self._path(key)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"Call budget of {self.max_calls} reached; not calling {self.model}")
        llm_client.last_usage = None
        response = self._call_with_backoff(prompt, schema, system)
        usage = llm_client.last_usage if self._call is llm_client.generate_json else None
        self.usage.append(usage or {})
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "key": key, "temperature": TEMPERATURE, "system": system,
                  "prompt": prompt, "schema": schema, "response": response, "usage": usage}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return response

    def _call_with_backoff(self, prompt, schema, system) -> Dict[str, Any]:
        """The budget counts prompts sent (cache misses); refused retries of the same prompt are counted in self.retries."""
        self.calls += 1
        for attempt in range(len(self.backoff_s) + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if self._last_call and wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self._call(prompt, schema=schema, system=system, temperature=TEMPERATURE, model=self.model)
            except Exception as e:
                if _status(e) not in BACKOFF_STATUS:
                    raise
                if is_daily_quota(e):
                    raise QuotaStop(f"{self.model}: daily quota used up; run again tomorrow to continue") from e
                if attempt == len(self.backoff_s):
                    raise QuotaStop(f"{self.model} still refusing after {attempt} retries: {e}") from e
                self.retries += 1
                delay = self.backoff_s[attempt]
                logger.warning(f"{self.model} returned {_status(e)}; waiting {delay:.0f}s (retry {attempt + 1}/{len(self.backoff_s)})")
                self._sleep(delay)
        raise AssertionError("unreachable")
