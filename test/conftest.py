"""Test isolation: never wait on (or hold) the real job lock — a live translation run holds it."""

import os
import tempfile

os.environ.setdefault("DEV_TITLES_LLM_LOCK", os.path.join(tempfile.mkdtemp(prefix="dt-test-"), "llm.lock"))
