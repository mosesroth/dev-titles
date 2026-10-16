"""Sphinx configuration."""

import datetime
import os
import shutil

try:
    from importlib.metadata import version as _pkg_version

    _release = _pkg_version("dev-titles")
except Exception:  # not installed (e.g. building docs from a bare checkout)
    _release = "0.1.0"


def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../src/")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = []

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# Sphinx configuration below.
project = "dev-titles"
release = _release
version = ".".join(_release.split(".")[:2])
copyright = "{}, Moses Roth".format(datetime.datetime.now().year)

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser"
]

source_suffix = ".md"
master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "haiku"
htmlhelp_basename = "dev-titles-doc"

napoleon_use_rtype = False

# Suppress warnings about orphaned documents
suppress_warnings = ["toc.not_readable"]
