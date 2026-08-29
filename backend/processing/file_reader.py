"""Plain-text file ingestion for wizard content."""

from __future__ import annotations

from pathlib import Path

from backend.config import (
    SUPPORTED_FILE_EXTENSIONS,
    TEXT_FILE_ENCODING,
    TEXT_FILE_JOIN_SEPARATOR,
)


def read_text_files(file_paths: list[str]) -> str:
    """Read supported text files exactly and merge them in caller order."""

    if not isinstance(file_paths, list):
        raise TypeError("file_paths must be a list of strings")

    paths: list[Path] = []
    for file_path in file_paths:
        if not isinstance(file_path, str):
            raise TypeError("each file path must be a string")
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension not in SUPPORTED_FILE_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
            raise ValueError(
                f"Unsupported file extension {path.suffix or '<none>'!r}; "
                f"expected one of: {allowed}"
            )
        paths.append(path)

    # Complete extension validation before any filesystem access or read.
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise IsADirectoryError(path)

    contents = [path.read_text(encoding=TEXT_FILE_ENCODING) for path in paths]
    return TEXT_FILE_JOIN_SEPARATOR.join(contents)
