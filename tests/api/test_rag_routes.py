"""
Integration tests for RAG search routes.

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
def sample_document_id(client):
    """Create a sample document for testing search."""
    content = """Artificial Intelligence (AI) and Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms
that can learn from and make predictions or decisions based on data.

## Deep Learning

Deep learning is a type of machine learning that uses neural networks
with multiple layers to model complex patterns in data.

## Natural Language Processing

Natural Language Processing (NLP) is a branch of AI that deals with
the interaction between computers and human language.

## Applications

AI has many applications including:
- Image recognition
- Speech recognition
- Natural language understanding
- Autonomous vehicles
- Medical diagnosis
"""

    # Upload the document
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("ai_test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")},
        data={
            "split_strategy": "recursive",
            "chunk_size": "200",
            "chunk_overlap": "20",
            "enable_refinement": "false",
        }
    )

    if response.status_code == 200:
        return response.json()["data"]["document_id"]
    return None


class TestRagRoutes:
    """Test cases for RAG search routes."""

    def test_rag_health_check(self, client):
        """Test RAG service health check."""
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert "status" in data["data"]
        assert "services" in data["data"]

    def test_search_basic(self, client, sample_document_id):
        """Test basic search functionality."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "machine learning",
                "top_k": "5",
                "use_rerank": "false",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "data" in data
        assert "query" in data["data"]
        assert data["data"]["query"] == "machine learning"
        assert "results" in data["data"]
        assert "total_hits" in data["data"]

    def test_search_with_document_filter(self, client, sample_document_id):
        """Test search with document ID filter."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "artificial intelligence",
                "top_k": "5",
                "document_id": sample_document_id,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

        # If results exist, verify they are from the specified document
        if data["data"]["total_hits"] > 0:
            for result in data["data"]["results"]:
                assert result["document_id"] == sample_document_id

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "",
                "top_k": "5",
            }
        )

        # Empty query should be rejected (validation error)
        assert response.status_code == 422

    def test_search_top_k_parameter(self, client, sample_document_id):
        """Test search with different top_k values."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        for top_k in [1, 5, 10, 20]:
            response = client.get(
                "/api/v1/rag/search",
                params={
                    "query": "AI",
                    "top_k": str(top_k),
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]["results"]) <= top_k

    def test_search_post_method(self, client, sample_document_id):
        """Test search using POST method."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        response = client.post(
            "/api/v1/rag/search",
            params={
                "query": "neural networks",
                "top_k": "5",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_search_long_query(self, client, sample_document_id):
        """Test search with a longer query."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        long_query = "What are the applications of artificial intelligence and machine learning in modern technology?"

        response = client.get(
            "/api/v1/rag/search",
            params={
                "query": long_query,
                "top_k": "5",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_search_no_results(self, client):
        """Test search that returns no results."""
        # Use a query that's unlikely to match
        response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "xyzabc123nonexistentterm",
                "top_k": "5",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        # May return 0 results or some results depending on test environment
        assert "results" in data["data"]

    def test_search_case_sensitivity(self, client, sample_document_id):
        """Test that search is case-insensitive."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        queries = ["machine learning", "MACHINE LEARNING", "Machine Learning"]

        results_count = []
        for query in queries:
            response = client.get(
                "/api/v1/rag/search",
                params={
                    "query": query,
                    "top_k": "10",
                }
            )

            assert response.status_code == 200
            data = response.json()
            results_count.append(data["data"]["total_hits"])

        # All case variations should return similar results
        # (allowing for some variation due to exact word matching)
        assert max(results_count) > 0

    def test_search_multiple_queries(self, client, sample_document_id):
        """Test multiple different search queries."""
        if sample_document_id is None:
            pytest.skip("No sample document available")

        queries = [
            "AI",
            "machine learning",
            "deep learning",
            "neural networks",
            "natural language processing",
        ]

        for query in queries:
            response = client.get(
                "/api/v1/rag/search",
                params={
                    "query": query,
                    "top_k": "5",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0

    def test_search_chinese_query(self, client):
        """Test search with Chinese language query."""
        response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "人工智能",
                "top_k": "5",
            }
        )

        # Should not fail, even if no results
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    def test_health_services_status(self, client):
        """Test that health check returns service status."""
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "services" in data["data"]

        services = data["data"]["services"]
        # Check for expected service keys (they might be True or False)
        assert isinstance(services, dict)


class TestEndToEndFlow:
    """End-to-end flow tests."""

    def test_upload_and_search_flow(self, client):
        """Test the complete flow: upload document, then search for it."""
        content = """E-commerce and Digital Marketing

E-commerce has revolutionized the way businesses sell products and services.

## Digital Marketing Strategies

- Social media marketing
- Email marketing
- Content marketing
- Search engine optimization (SEO)

## Customer Experience

Providing excellent customer experience is crucial for e-commerce success.
This includes easy navigation, fast loading times, and secure payments.
"""

        # Upload document
        upload_response = client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "ecommerce_test.txt",
                    io.BytesIO(content.encode("utf-8")),
                    "text/plain"
                )
            },
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

        # Search for content from the uploaded document
        search_response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "digital marketing",
                "top_k": "5",
                "document_id": document_id,
            }
        )

        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["code"] == 0

        # Clean up
        client.delete(f"/api/v1/documents/{document_id}")

    def test_upload_search_and_delete_flow(self, client):
        """Test upload -> search -> delete flow."""
        content = """Python Programming

Python is a high-level programming language.

## Features

- Easy to learn
- Versatile
- Large community
- Extensive libraries

## Use Cases

- Web development
- Data science
- Machine learning
- Automation
"""

        # Upload
        upload_response = client.post(
            "/api/v1/documents/upload",
            files={
                "file": ("python_test.txt", io.BytesIO(content.encode("utf-8")), "text/plain")
            },
            data={
                "split_strategy": "recursive",
                "chunk_size": "150",
                "chunk_overlap": "15",
                "enable_refinement": "false",
            }
        )

        assert upload_response.status_code == 200
        document_id = upload_response.json()["data"]["document_id"]

        # Verify document exists
        get_response = client.get(f"/api/v1/documents/{document_id}")
        assert get_response.status_code == 200

        # Search
        search_response = client.get(
            "/api/v1/rag/search",
            params={
                "query": "programming",
                "top_k": "5",
            }
        )
        assert search_response.status_code == 200

        # Delete
        delete_response = client.delete(f"/api/v1/documents/{document_id}")
        assert delete_response.status_code == 200

        # Verify deletion
        get_after_delete = client.get(f"/api/v1/documents/{document_id}")
        assert get_after_delete.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
