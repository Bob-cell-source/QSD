import ast
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def iter_json_records(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Read Amazon json-lines files.

    Some legacy metadata files use Python dict literals instead of strict JSON,
    so we fall back to ast.literal_eval for those lines.
    """

    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

