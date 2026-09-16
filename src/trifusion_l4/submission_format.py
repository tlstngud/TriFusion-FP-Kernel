"""DACON CSV contract and file mapping, independent of CUDA/model imports.

Official columns are *_PROB. Unsuffixed aliases remain accepted for the older
local benchmark fixtures; emitted headers always preserve the input template.
"""

from pathlib import Path
import csv
import math

OUTPUTS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def discover_inputs(root):
    candidates = []
    for name in ("data", "open"):
        directory = Path(root) / name
        if directory.exists():
            candidates.extend(directory.rglob("sample_submission.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one official sample_submission.csv; found {len(candidates)}")
    template = candidates[0]
    with template.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns or not rows:
        raise ValueError("Empty submission template")
    if len({name.casefold() for name in columns}) != len(columns):
        raise ValueError("Duplicate template columns")
    aliases = {}
    for name in OUTPUTS:
        matches = [column for column in columns if column.casefold() in (name, name + "_prob")]
        if len(matches) != 1:
            raise ValueError(f"Expected one output column for {name}; found {matches}")
        aliases[name] = matches[0]
    id_columns = [name for name in columns if name not in aliases.values()]
    if len(id_columns) != 1 or id_columns[0].casefold() != "id":
        raise ValueError("Template must have exactly one ID column")
    id_column = id_columns[0]
    for row in rows:
        value = row.get(id_column)
        if value is None or not value.strip():
            raise ValueError("Empty template ID")
        row[id_column] = value.strip()
    ids = [row[id_column] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate template IDs")
    data_root = next(
        directory
        for directory in (Path(root) / "data", Path(root) / "open")
        if directory in template.parents
    )
    lookup = {}
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            for key in {path.name, path.stem, str(path.relative_to(data_root))}:
                lookup.setdefault(key, set()).add(path)
    paths = []
    for value in ids:
        matches = lookup.get(value, set())
        if len(matches) != 1:
            raise ValueError(f"Expected one audio file for ID {value!r}; found {len(matches)}")
        paths.append(next(iter(matches)))
    return columns, rows, aliases, paths


def write_submission(path, columns, rows, aliases, predictions):
    predictions = [tuple(float(value) for value in row) for row in predictions]
    if (
        len(predictions) != len(rows)
        or any(len(row) != 5 for row in predictions)
        or any(
            not math.isfinite(value) or not 0 <= value <= 1 for row in predictions for value in row
        )
    ):
        raise ValueError("Predictions must be finite [0,1] values for every template row")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row, values in zip(rows, predictions):
            writer.writerow(
                {
                    **row,
                    **{
                        aliases[name]: format(float(value), ".9g")
                        for name, value in zip(OUTPUTS, values)
                    },
                }
            )
    temporary.replace(path)
