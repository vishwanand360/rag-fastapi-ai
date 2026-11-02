import pymupdf
import pytest

from backend.ingestion import parse_pdf, parse_pdfs


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_parse_pdf_keeps_source_page_and_chunk_metadata():
    pdf = make_pdf(["Alpha " * 100, "Beta " * 100])
    chunks = parse_pdf(pdf, "report.pdf", chunk_size=80, overlap=10)
    assert chunks
    assert {chunk["page"] for chunk in chunks} == {1, 2}
    assert all(chunk["source"] == "report.pdf" for chunk in chunks)
    assert all(chunk["doc_id"] == chunks[0]["doc_id"] for chunk in chunks)
    assert chunks[0]["metadata"]["page_count"] == 2


@pytest.mark.parametrize("filename,data", [("file.txt", b"%PDF-x"), ("file.pdf", b"not a pdf")])
def test_parse_pdf_rejects_invalid_input(filename, data):
    with pytest.raises(ValueError):
        parse_pdf(data, filename)


def test_parse_pdfs_supports_multiple_files():
    chunks = parse_pdfs([("one.pdf", make_pdf(["One document."])), ("two.pdf", make_pdf(["Second document."]))])
    assert {chunk["source"] for chunk in chunks} == {"one.pdf", "two.pdf"}
    assert len({chunk["doc_id"] for chunk in chunks}) == 2
