"""Shared test configuration for the whole suite.

Probe tuning knobs (``MAX_PROBE_QUESTIONS``, ``PROBE_BATCH_SIZE``) are read
by ``graphs.probe.nodes`` at import time. A developer's local ``.env``
(loaded via ``load_dotenv()`` in ``llm.py`` during test imports) must not
change test behavior, so the code defaults are pinned *before* any test
module is imported — same strategy as ``DATABASE_URL`` in
``tests/routers/conftest.py``.
"""

from __future__ import annotations

import os

os.environ["MAX_PROBE_QUESTIONS"] = "10"
os.environ["PROBE_BATCH_SIZE"] = "3"
