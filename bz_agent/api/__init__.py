"""
API module for RAG Document Processing.

This module contains the API routes and schemas for the document processing service.

Author: RAG Team
Created: 2026-03-02
"""

from .prompt_api import router as prompt_router
from .document_routes import document_router
from .rag_routes import rag_router

__all__ = [
    "prompt_router",
    "document_router",
    "rag_router",
]
