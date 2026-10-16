"""dev-titles packaging shim.

All metadata lives in pyproject.toml; this file exists only for tooling that still
expects a setup.py (e.g. legacy editable installs).
"""

from setuptools import setup

setup()
