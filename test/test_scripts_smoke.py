"""Smoke test: every script under scripts/ at least imports and parses --help (catches latent
breakage in rarely-run scripts — check_vendor_review.py crashed on its first run after a refactor).
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    p
    for p in (ROOT / "scripts").rglob("*.py")
    if p.parent.name != "hooks" and "argparse" in p.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_script_help(script):
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr[-800:]


def test_spot_ab_cache_key_follows_code_content(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "spot_ab.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("spot_ab_t", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    (tmp_path / "src").mkdir()
    (tmp_path / "glossary").mkdir()
    f = tmp_path / "src" / "a.py"
    f.write_text("x = 1")
    h1 = m._code_hash(tmp_path)
    f.write_text("x = 2")
    assert m._code_hash(tmp_path) != h1


def test_localize_arm_refuses_the_live_repo():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "scripts/eval/localize_arm.py"), str(root), "x.vtt", "out"],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "live repo" in r.stderr
