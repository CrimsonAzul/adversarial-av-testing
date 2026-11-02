"""Append-only episode records and small file helpers shared by every job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomic publish with retries for Windows sharing locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    for attempt in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class JsonlLog:
    """One JSON object per line, flushed immediately; a crash leaves a readable prefix."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tail_checked = False

    def _drop_torn_tail(self) -> None:
        """A hard kill can leave a partial last line; appending after it would hide every later row from read()."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            self.path.write_bytes(data[:data.rfind(b"\n") + 1])

    def append(self, row: dict) -> None:
        if not self._tail_checked:
            self._drop_torn_tail()
            self._tail_checked = True
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.append(row)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:   # torn final line after a hard kill
                    break
        return rows


def control_state(folder: Path) -> str | None:
    """'stop' (abort now) or 'pause' (finish the current unit, then exit) from control files."""
    if (Path(folder) / "STOP").exists():
        return "stop"
    if (Path(folder) / "PAUSE").exists():
        return "pause"
    return None
