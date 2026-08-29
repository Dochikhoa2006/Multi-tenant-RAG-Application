from __future__ import annotations

from pathlib import Path

import pytest

from backend.processing import file_reader
from backend.processing.file_reader import read_text_files


def test_txt_content_is_preserved_exactly(tmp_path: Path) -> None:
    text_file = tmp_path / "facts.txt"
    content = "  Leading spaces\nMiddle line\nTrailing spaces  \n"
    text_file.write_text(content, encoding="utf-8")

    assert read_text_files([str(text_file)]) == content


def test_markdown_content_is_preserved_exactly(tmp_path: Path) -> None:
    markdown_file = tmp_path / "notes.md"
    content = "# Heading\n\n- First item\n- Second item\n\n"
    markdown_file.write_text(content, encoding="utf-8")

    assert read_text_files([str(markdown_file)]) == content


def test_multiple_files_preserve_caller_order_and_edges(tmp_path: Path) -> None:
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.md"
    first_content = " first\n"
    second_content = "\nsecond "
    first_file.write_text(first_content, encoding="utf-8")
    second_file.write_text(second_content, encoding="utf-8")

    assert read_text_files([str(first_file), str(second_file)]) == (
        first_content + "\n\n" + second_content
    )
    assert read_text_files([str(second_file), str(first_file)]) == (
        second_content + "\n\n" + first_content
    )


def test_join_separator_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text("first", encoding="utf-8")
    second_file.write_text("second", encoding="utf-8")
    monkeypatch.setattr(file_reader, "TEXT_FILE_JOIN_SEPARATOR", "<JOIN>")

    assert read_text_files([str(first_file), str(second_file)]) == (
        "first<JOIN>second"
    )


def test_uppercase_supported_extension_is_accepted(tmp_path: Path) -> None:
    text_file = tmp_path / "NOTES.MD"
    text_file.write_text("Uppercase extension.", encoding="utf-8")

    assert read_text_files([str(text_file)]) == "Uppercase extension."


def test_empty_list_and_empty_file_are_preserved(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.md"
    whitespace_file = tmp_path / "whitespace.txt"
    empty_file.write_text("", encoding="utf-8")
    whitespace_file.write_text(" \n\t ", encoding="utf-8")

    assert read_text_files([]) == ""
    assert read_text_files([str(empty_file)]) == ""
    assert read_text_files([str(whitespace_file)]) == " \n\t "


def test_invalid_extension_is_rejected(tmp_path: Path) -> None:
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"not really a pdf")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_text_files([str(pdf_file)])


def test_all_extensions_are_validated_before_missing_paths(tmp_path: Path) -> None:
    missing_text = tmp_path / "missing.txt"
    invalid_file = tmp_path / "invalid.pdf"

    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_text_files([str(missing_text), str(invalid_file)])


def test_nonexistent_supported_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text_files([str(tmp_path / "missing.txt")])
