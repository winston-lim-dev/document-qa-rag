from __future__ import annotations

from io import BytesIO

import pytest

from document_qa.ingestion import IngestionError, ingest_pdf


def _pdf_with_pages(pages: list[str | None]) -> bytes:
    """Build a tiny PDF containing one simple text stream per non-empty page."""
    objects: list[bytes] = []
    page_ids = [3 + index * 2 for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())

    for index, text in enumerate(pages):
        page_id = page_ids[index]
        content_id = page_id + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
                f"/BaseFont /Helvetica >> >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        escaped = (text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1") if text else b""
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(output)


def test_extracts_multiple_pages_with_provenance() -> None:
    chunks = ingest_pdf(_pdf_with_pages(["First page", "Second page"]), "notes.pdf")

    assert [chunk.text for chunk in chunks] == ["First page", "Second page"]
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert {chunk.filename for chunk in chunks} == {"notes.pdf"}


def test_document_id_is_deterministic_and_content_based() -> None:
    original = _pdf_with_pages(["Same content"])
    changed = _pdf_with_pages(["Different content"])

    first_id = ingest_pdf(original, "one.pdf")[0].document_id
    renamed_id = ingest_pdf(BytesIO(original), "renamed.pdf")[0].document_id
    changed_id = ingest_pdf(changed, "one.pdf")[0].document_id

    assert first_id == renamed_id
    assert first_id != changed_id


def test_chunk_ids_are_deterministic_and_unique() -> None:
    pdf = _pdf_with_pages(["alpha beta gamma delta epsilon zeta eta theta"])
    first = ingest_pdf(pdf, "chunks.pdf", chunk_size=12, chunk_overlap=2)
    second = ingest_pdf(pdf, "chunks.pdf", chunk_size=12, chunk_overlap=2)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert len(first) > 1


def test_empty_pages_do_not_produce_chunks() -> None:
    chunks = ingest_pdf(_pdf_with_pages([None, "Usable", None]), "mixed.pdf")

    assert [(chunk.page, chunk.text) for chunk in chunks] == [(2, "Usable")]


def test_document_without_extractable_text_is_rejected() -> None:
    with pytest.raises(IngestionError, match="no extractable text"):
        ingest_pdf(_pdf_with_pages([None, None]), "empty.pdf")


def test_custom_chunk_settings_and_order_are_deterministic() -> None:
    pdf = _pdf_with_pages([
        "one two three four five six",
        "seven eight nine ten eleven twelve",
    ])

    first = ingest_pdf(pdf, "ordered.pdf", chunk_size=10, chunk_overlap=0)
    second = ingest_pdf(pdf, "ordered.pdf", chunk_size=10, chunk_overlap=0)

    assert len(first) > 2
    assert [chunk.page for chunk in first] == sorted(chunk.page for chunk in first)
    assert first == second
