from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

from app.core.errors import (
    DependencyUnavailableError,
    IngestionValidationError,
    UnsupportedDocumentTypeError,
)
from app.rag.models import ParsedDocument


class DocumentParser(Protocol):
    parser_version: str

    def supports(self, *, filename: str, content_type: str | None) -> bool:
        ...

    def parse(self, *, filename: str, content: bytes, content_type: str | None) -> ParsedDocument:
        ...


class TextDocumentParser:
    parser_version = "text-parser-v1"

    def supports(self, *, filename: str, content_type: str | None) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix in {"", ".txt", ".md"} or (content_type or "").startswith("text/")

    def parse(self, *, filename: str, content: bytes, content_type: str | None) -> ParsedDocument:
        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            raise IngestionValidationError("document text is empty")
        return ParsedDocument(
            text=text,
            content_type=content_type or "text/plain",
            parser_version=self.parser_version,
            metadata={"filename": filename},
        )


class DocxDocumentParser:
    parser_version = "docx-zipxml-parser-v1"

    def supports(self, *, filename: str, content_type: str | None) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix == ".docx" or content_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/docx",
        }

    def parse(self, *, filename: str, content: bytes, content_type: str | None) -> ParsedDocument:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                xml_payload = archive.read("word/document.xml")
        except (KeyError, zipfile.BadZipFile) as exc:
            raise IngestionValidationError("invalid DOCX payload") from exc

        root = ElementTree.fromstring(xml_payload)
        paragraphs: list[str] = []
        for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            parts = [
                node.text
                for node in paragraph.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                )
                if node.text
            ]
            if parts:
                paragraphs.append("".join(parts))

        text = "\n".join(paragraphs).strip()
        if not text:
            raise IngestionValidationError("DOCX contained no extractable text")

        return ParsedDocument(
            text=text,
            content_type=(
                content_type
                or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            parser_version=self.parser_version,
            metadata={"filename": filename, "paragraph_count": len(paragraphs)},
        )


class PdfDocumentParser:
    parser_version = "pypdf-parser-v1"

    def supports(self, *, filename: str, content_type: str | None) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix == ".pdf" or content_type == "application/pdf"

    def parse(self, *, filename: str, content: bytes, content_type: str | None) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError as exc:
            raise DependencyUnavailableError(
                "PDF parsing requires the optional pypdf dependency. "
                'Install with `pip install -e ".[platform]"`.'
            ) from exc

        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page.strip() for page in pages if page.strip()).strip()
        if not text:
            raise IngestionValidationError("PDF contained no extractable text")

        return ParsedDocument(
            text=text,
            content_type=content_type or "application/pdf",
            parser_version=self.parser_version,
            metadata={"filename": filename, "page_count": len(reader.pages)},
        )


class DocumentParserRouter:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def parse(self, *, filename: str, content: bytes, content_type: str | None) -> ParsedDocument:
        for parser in self._parsers:
            if parser.supports(filename=filename, content_type=content_type):
                return parser.parse(filename=filename, content=content, content_type=content_type)
        raise UnsupportedDocumentTypeError(f"unsupported document type for {filename}")


def default_parser_router() -> DocumentParserRouter:
    return DocumentParserRouter(
        parsers=[
            PdfDocumentParser(),
            DocxDocumentParser(),
            TextDocumentParser(),
        ]
    )
