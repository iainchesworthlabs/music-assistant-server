"""
Copy the `_iclforge_player@v1` role package into the sendspin provider.

The package is developed in the iclforge repository (tools/sendspin/ma_role) and is
self-contained, so the fork only carries a verbatim copy plus a note of where it came from.

Usage: python scripts/vendor_ma_role.py <path to tools/sendspin/ma_role>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FILES = ("__init__.py", "eac3.py", "role.py")
TARGET = (
    Path(__file__).resolve().parents[1] / "music_assistant" / "providers" / "sendspin" / "ma_role"
)


def main() -> int:
    """Copy the files and write VENDORED.txt."""
    if len(sys.argv) != 2:
        sys.stdout.write(__doc__ or "")
        return 2
    source = Path(sys.argv[1]).resolve()
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, TARGET / name)
    try:
        commit = subprocess.run(  # noqa: S603
            ["git", "-C", str(source), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        commit = "unknown"
    (TARGET / "VENDORED.txt").write_text(
        "Verbatim copy of tools/sendspin/ma_role/{"
        + ",".join(FILES)
        + "} from the iclforge repository,\n"
        f"commit {commit}. Do not edit here: change it there and run scripts/vendor_ma_role.py.\n",
        encoding="utf-8",
    )
    sys.stdout.write(f"copied {len(FILES)} files from {source} (commit {commit}) to {TARGET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
