import pytest


@pytest.fixture(autouse=True)
def _no_run_cache(monkeypatch):
    # Fake-LLM runs must never be read from or written to the real simulation cache.
    monkeypatch.setenv("RUN_CACHE", "0")
