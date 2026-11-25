"""Pretty JSON writer that keeps scalar arrays on one line."""

from __future__ import annotations

import json
from pathlib import Path


def write_compact_json(path: Path, snapshot: dict) -> None:
    """Pretty-print ``snapshot``; lists of scalars stay on a single line."""

    def _is_scalar(value: object) -> bool:
        return not isinstance(value, (dict, list, tuple))

    def _encode(obj: object, indent: int = 0) -> str:
        pad = "  " * indent
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = ["{"]
            items = list(obj.items())
            for index, (key, value) in enumerate(items):
                comma = "," if index < len(items) - 1 else ""
                lines.append(
                    f'{pad}  {json.dumps(key)}: {_encode(value, indent + 1)}{comma}'
                )
            lines.append(f"{pad}}}")
            return "\n".join(lines)
        if isinstance(obj, (list, tuple)):
            if all(_is_scalar(item) for item in obj):
                inner = ", ".join(json.dumps(item, default=str) for item in obj)
                return f"[{inner}]"
            if not obj:
                return "[]"
            lines = ["["]
            for index, item in enumerate(obj):
                comma = "," if index < len(obj) - 1 else ""
                lines.append(f"{pad}  {_encode(item, indent + 1)}{comma}")
            lines.append(f"{pad}]")
            return "\n".join(lines)
        return json.dumps(obj, default=str)

    path.write_text(_encode(snapshot) + "\n", encoding="utf-8")
