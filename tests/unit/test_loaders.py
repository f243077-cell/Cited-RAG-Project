import hashlib

import pytest

from citerag.errors import FileTooLargeError, InvalidFileTypeError
from citerag.ingest.loaders import compute_doc_id, load_pages, sanitize_name, validate_file
from tests.pdf_fixture import make_pdf

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def test_doc_id_is_first_16_hex_of_sha256():
    assert compute_doc_id(b"abc") == hashlib.sha256(b"abc").hexdigest()[:16]


@pytest.mark.parametrize(
    ("name", "data", "kind"),
    [
        ("a.pdf", make_pdf(["hello"]), "pdf"),
        ("A.PDF", make_pdf(["hello"]), "pdf"),
        ("a.md", b"# title", "text"),
        ("a.markdown", b"# title", "text"),
        ("a.txt", "caf\u00e9".encode(), "text"),
        ("a.png", PNG, "image"),
        ("a.jpg", JPEG, "image"),
        ("a.jpeg", JPEG, "image"),
    ],
)
def test_allowed_files(name, data, kind):
    assert validate_file(name, data, max_mb=25) == kind


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("setup.exe", b"MZ\x90\x00"),
        ("renamed.pdf", b"MZ\x90\x00 this is an exe"),  # F-ING-1: magic bytes, not the extension
        ("fake.png", b"not a png"),
        ("fake.jpg", PNG),
        ("empty.txt", b""),
        ("blank.md", b"  \n\t "),
        ("binary.txt", b"abc\x00def"),
        ("latin.txt", b"caf\xe9"),
        ("noext", b"hello"),
    ],
)
def test_rejected_files(name, data):
    with pytest.raises(InvalidFileTypeError):
        validate_file(name, data, max_mb=25)


def test_size_limit_uses_mb_and_allows_the_exact_limit():
    validate_file("a.txt", b"x" * (1024 * 1024), max_mb=1)
    with pytest.raises(FileTooLargeError):
        validate_file("a.txt", b"x" * (1024 * 1024 + 1), max_mb=1)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\file.md", "file.md"),
        ("we<ird>:na|me?.txt", "we_ird__na_me_.txt"),
        ("...hidden.txt", "hidden.txt"),
        ("", "upload"),
    ],
)
def test_sanitize_name(raw, clean):
    assert sanitize_name(raw) == clean


def test_pdf_pages_are_one_based_and_empty_pages_are_skipped():
    loaded = load_pages("pdf", make_pdf(["First page text", "", "Third page text"]))
    assert loaded.page_count == 3
    assert [(p.page, p.text) for p in loaded.pages] == [(1, "First page text"), (3, "Third page text")]


def test_text_file_is_one_page_without_a_number():
    loaded = load_pages("text", b"# Title\n\nBody")
    assert loaded.page_count is None
    assert [(p.page, p.text) for p in loaded.pages] == [(None, "# Title\n\nBody")]


def test_image_file_has_no_text_pages():
    assert load_pages("image", PNG).pages == []


def test_corrupt_pdf_is_a_typed_error():
    with pytest.raises(InvalidFileTypeError):
        load_pages("pdf", b"%PDF-1.4 then garbage")


def test_pdf_without_extractable_text_is_a_typed_error():
    with pytest.raises(InvalidFileTypeError, match="no extractable text"):
        load_pages("pdf", make_pdf(["", ""]))
