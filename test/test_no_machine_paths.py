"""Code must not depend on files in shared temp dirs or on one developer's machine.

Reading a fixed path like /tmp/x.py is a security issue (anyone on the machine can plant it)
and breaks on every other machine. Comments and docstrings may mention such paths; code may not.
Use tempfile.mkstemp / TemporaryDirectory for temp files, and paths inside the project otherwise.
"""

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD = re.compile(r"""^['"](/tmp/|/var/tmp/|/private/tmp/|/Users/|/home/[a-z])""")


def _string_literals(path):
    src = path.read_text(encoding="utf-8")
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, t in enumerate(toks):
        if t.type != tokenize.STRING:
            continue
        prev = next(
            (
                p
                for p in reversed(toks[:i])
                if p.type
                not in (
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.COMMENT,
                )
            ),
            None,
        )
        if (
            prev is None
            or (prev.type == tokenize.OP and prev.string == ":")
            and t.string.startswith(('"""', "'''"))
        ):
            continue  # module/function/class docstring
        if t.string.startswith(('"""', "'''")) and (
            prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT)
        ):
            continue
        yield t.start[0], t.string.lstrip("rbfuRBFU")


def test_no_hardcoded_temp_or_home_paths():
    offenders = []
    for base in ("src", "scripts"):
        for p in (ROOT / base).rglob("*.py"):
            for line, s in _string_literals(p):
                if BAD.match(s):
                    offenders.append(f"{p.relative_to(ROOT)}:{line}: {s[:60]}")
    assert not offenders, "hard-coded temp/home paths:\n" + "\n".join(offenders)
