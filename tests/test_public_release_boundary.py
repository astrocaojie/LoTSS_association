from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PERSONAL_ABSOLUTE_PATH = re.compile(
    rb"/(?:Users|home|shared|scratch)/[^\x00\r\n\t \"']+",
    flags=re.IGNORECASE,
)


def _release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        relative
        for raw in result.stdout.split(b"\0")
        if raw
        for relative in [Path(raw.decode("utf-8"))]
        if (PROJECT_ROOT / relative).is_file()
    ]


def test_release_text_excludes_personal_paths() -> None:
    path_violations: list[str] = []
    for relative in _release_files():
        data = (PROJECT_ROOT / relative).read_bytes()
        if PERSONAL_ABSOLUTE_PATH.search(data):
            path_violations.append(relative.as_posix())
    assert not path_violations, f"personal absolute paths found: {path_violations}"
