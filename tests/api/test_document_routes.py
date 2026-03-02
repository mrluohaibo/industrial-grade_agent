"""
Integration tests for document API routes.

Author: RAG Team
Created: 2026-03-02
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def test_markdown_file():
    """Create a test Markdown file."""
    content = """# Test Document

This is a test document for RAG processing.

## Section 1

This is the first section with some content.

## Section 2

This is the second section with more content.

The quick brown fox jumps over the lazy dog.
"""
    return io.BytesIO(content.encode("utf-8")), "test_document.md"


@pytest.fixture
def test_txt_file():
    """Create a test plain text file."""
    content = """This is a plain text document.

It contains multiple sentences.
The document is meant for RAG processing.
Content goes here and there.
"""
    return io.BytesIO(content.encode("utf-8")), "test_document.txt"


@pytest.fixture
def test_pdf_file():
    """Create a test PDF file (using a small valid PDF)."""
    # Minimal valid PDF header
    content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/Contents 5 0 R
>>
endobj
4 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
5 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
50 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
0000000270 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
355
%%EOF
"""
    return io.BytesIO(content), "test_document.pdf"


class TestDocumentRoutes:
    """Test cases for document routes."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_upload_markdown(self, client, test_markdown_file):
        """Test uploading a Markdown document."""
        file_content, filename = test_markdown_file

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, file_content, "text/markdown")},
            data={
                "split_strategy": "recursive",
                "chunk_size": "200",
                "chunk_overlap": "20",
                "enable_refinement": "false",  # Disable refinement for faster test
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["message"] == "Document uploaded and processed successfully"
        assert "data" in data
        assert "document_id" in data["data"]
        assert data["data"]["status"] in ["success", "partial"]

        # Store document_id for cleanup
        return data["data"]["document_id"]

    def test_upload_txt(self, client, test_txt_file):
        """Test uploading a plain text document."""
        file_content, filename = test_txt_file

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, file_content, "text/plain")},
            data={
                "split_strategy": "recursive",
                "chunk_size": "200",
                "chunk_overlap": "20",
                "enable_refinement": "false",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        return data["data"]["document_id"]

    def test_upload_pdf(self, client, test_pdf_file):
        """Test uploading a PDF document."""
        file_content, filename = test_pdf_file

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, file_content, "application/pdf")},
            data={
                "split_strategy": "recursive",
                "chunk_size": "200",
                "chunk_overlap": "20",
                "enable_refinement": "false",
            }
        )

        # PDF parsing might fail for minimal PDF
        # We'll accept either success or partial status
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        return data["data"]["document_id"]

    def test_upload_unsupported_file_type(self, client):
        """Test uploading an unsupported file type."""
        content = b"This is not a supported file format"
        filename = "test.xyz"

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
            data={
                "split_strategy": "recursive",
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == 400
        assert "Unsupported file type" in data["detail"]

    def test_upload_large_file(self, client):
        """Test uploading a file that exceeds size limit."""
        # Create a file larger than the limit (10MB)
        content = b"x" * 11 * 1024 * 1024  # 11MB
        filename = "large_file.txt"

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, io.BytesIO(content), "text/plain")},
            data={
                "split_strategy": "recursive",
            }
        )

        assert response.status_code == 413
        data = response.json()
        assert data["code"] == 413
        assert "too large" in data["detail"].lower()

    def test_get_document(self, client, test_txt_file):
        """Test getting document information."""
        # First upload a document
        file_content, filename = test_txt_file

        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, file_content, "text/plain")},
            data={
                "split_strategy": "recursive",
                "chunk_size": "200",
                "chunk_overlap": "20",
                "enable_refinement": "false",
            }
        )

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        document_id = upload_data["data"]["document_id"]

        # Get document info
        response = client.get(f"/api/v1/documents/{document_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["document_id"] == document_id
        assert data["data"]["filename"] == filename
        assert "chunk_count" in data["data"]

        return document_id

    def test_get_document_not_found(self, client):
        """Test getting a non-existent document."""
        fake_id = "9999999999999999999"

        response = client.get(f"/api/v1/documents/{fake_id}")

        assert response.status_code == 404

    def test_get_document_chunks(self, client, test_txt_file):
        """Test getting document chunks."""
        # First upload a document
        file_content, filename = test_txt_file

        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, file_content, "text/plain")},
            data={
                "split_strategy": "recursive",
                "chunk_size": "200",
                "chunk_overlap": "20",
                "enable_refinement": "false",
            }
        )

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        document_id = upload_data["data"]["document_id"]

        # Get document chunks
        response = client.get(f"/api/v1/documents/{document_id}/chunks")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "chunks" in data["data"]
        assert "total" in data["data"]
        assert data["data"]["document_id"] == document_id

        return document_id

    def test_delete_document(self, client, test_txt_file):
        """Test deleting a document."""
        # First upload a document
        file_content, filename = test_txt_file

        upload_response = client.post(
            "/api/v1/documents/upload",
            files={"file": (filename, file_content, "text/plain")},
            data={
                "split_strategy": "recursive",
                "chunk_size": "200",
                "chunk_overlap": "20",
                "enable_refinement": "false",
            }
        )

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        document_id = upload_data["data"]["document_id"]

        # Delete the document
        response = client.delete(f"/api/v1/documents/{document_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["document_id"] == document_id

        # Verify document is deleted
        get_response = client.get(f"/api/v1/documents/{document_id}")
        assert get_response.status_code == 404

    def test_list_documents(self, client):
        """Test listing documents."""
        response = client.get("/api/v1/documents/")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "total" in data["data"]

    def test_upload_with_different_strategies(self, client, test_txt_file):
        """Test uploading with different splitting strategies."""
        file_content, filename = test_txt_file

        strategies = ["recursive", "markdown_header", "hybrid"]

        for strategy in strategies:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": (f"{strategy}_{filename}", file_content, "text/plain")},
                data={
                    "split_strategy": strategy,
                    "chunk_size": "200",
                    "chunk_overlap": "20",
                    "enable_refinement": "false",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
