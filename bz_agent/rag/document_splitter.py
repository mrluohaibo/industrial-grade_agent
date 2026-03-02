"""
Enterprise-grade Document Splitter for RAG

This module provides comprehensive document splitting capabilities for RAG systems,
supporting multiple splitting strategies optimized for Chinese and English content.

Author: RAG Team
Created: 2026-03-02
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer


class SplitStrategy(str, Enum):
    """Available splitting strategies."""

    RECURSIVE = "recursive"  # Recursive character splitting (recommended)
    MARKDOWN_HEADER = "markdown_header"  # Split by Markdown headers
    SEMANTIC = "semantic"  # Semantic-based splitting
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class SplitResult:
    """Result of a document splitting operation."""

    text: str
    metadata: Dict[str, Any]
    index: int
    chunk_id: str


class DocumentSplitter:
    """
    Enterprise-grade document splitter with multiple strategies.

    Supports:
    - Recursive character splitting with Chinese-aware delimiters
    - Markdown header-based splitting
    - Semantic-based splitting using embeddings
    - Hybrid strategies combining multiple approaches
    """

    # Chinese punctuation marks for better splitting
    CHINESE_PUNCTUATION = ["。", "！", "？", "；", "：", "、", "，"]
    # English punctuation marks
    ENGLISH_PUNCTUATION = [".", "!", "?", ";", ":", ","]

    def __init__(
        self,
        strategy: Union[SplitStrategy, str] = SplitStrategy.RECURSIVE,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        embedding_model: Optional[str] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the document splitter.

        Args:
            strategy: Splitting strategy to use
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for recursive splitting
            embedding_model: Model name for semantic splitting
            keep_separator: Whether to keep separators in the chunks
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.strategy = SplitStrategy(strategy)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
        self.embedding_model = embedding_model

        # Setup separators with Chinese support
        self.separators = separators or self._get_default_separators()

        # Initialize semantic splitter if needed
        self._semantic_model = None
        if self.strategy == SplitStrategy.SEMANTIC or self.strategy == SplitStrategy.HYBRID:
            if embedding_model is None:
                embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self._semantic_model = SentenceTransformer(embedding_model)

    def _get_default_separators(self) -> List[str]:
        """Get default separators optimized for Chinese and English."""
        return [
            "\n\n\n",  # Triple newlines (major section breaks)
            "\n\n",  # Double newlines (paragraph breaks)
            "\n",  # Single newline
            "。",  # Chinese period
            "！",  # Chinese exclamation
            "？",  # Chinese question mark
            "；",  # Chinese semicolon
            "，",  # Chinese comma
            ".",  # English period
            "!",  # English exclamation
            "?",  # English question mark
            ";",  # English semicolon
            ",",  # English comma
            " ",  # Space
            "",  # Fallback: split by character
        ]

    def split_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SplitResult]:
        """
        Split text into chunks using the configured strategy.

        Args:
            text: Input text to split
            document_id: Optional document ID for metadata
            metadata: Additional metadata to include in results

        Returns:
            List of SplitResult objects containing chunks and metadata
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        if document_id:
            metadata["document_id"] = document_id

        if self.strategy == SplitStrategy.RECURSIVE:
            return self._split_recursive(text, metadata)
        elif self.strategy == SplitStrategy.MARKDOWN_HEADER:
            return self._split_markdown(text, metadata)
        elif self.strategy == SplitStrategy.SEMANTIC:
            return self._split_semantic(text, metadata)
        elif self.strategy == SplitStrategy.HYBRID:
            return self._split_hybrid(text, metadata)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _split_recursive(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[SplitResult]:
        """Split text using recursive character splitting."""
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            keep_separator=self.keep_separator,
            strip_whitespace=self.strip_whitespace,
            length_function=len,  # Use character count for Chinese
        )

        chunks = splitter.split_text(text)
        results = []

        for idx, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "split_strategy": SplitStrategy.RECURSIVE.value,
                "chunk_index": idx,
                "chunk_size": len(chunk),
            })
            results.append(SplitResult(
                text=chunk,
                metadata=chunk_metadata,
                index=idx,
                chunk_id=f"{metadata.get('document_id', 'doc')}_{idx}"
            ))

        return results

    def _split_markdown(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[SplitResult]:
        """Split Markdown text by headers."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        # First split by headers
        md_chunks = splitter.split_text(text)
        results = []

        for md_chunk in md_chunks:
            # Further split long chunks using recursive splitting
            chunk_text = md_chunk.page_content
            if len(chunk_text) > self.chunk_size:
                sub_splitter = RecursiveCharacterTextSplitter(
                    separators=self.separators,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    keep_separator=self.keep_separator,
                    strip_whitespace=self.strip_whitespace,
                    length_function=len,
                )
                sub_chunks = sub_splitter.split_text(chunk_text)
                sub_metadata = md_chunk.metadata.copy()
                sub_metadata.update(metadata)

                for idx, sub_chunk in enumerate(sub_chunks):
                    final_metadata = sub_metadata.copy()
                    final_metadata.update({
                        "split_strategy": SplitStrategy.MARKDOWN_HEADER.value,
                        "chunk_index": idx,
                        "chunk_size": len(sub_chunk),
                    })
                    results.append(SplitResult(
                        text=sub_chunk,
                        metadata=final_metadata,
                        index=len(results),
                        chunk_id=f"{metadata.get('document_id', 'doc')}_{len(results)}"
                    ))
            else:
                chunk_metadata = metadata.copy()
                chunk_metadata.update(md_chunk.metadata)
                chunk_metadata.update({
                    "split_strategy": SplitStrategy.MARKDOWN_HEADER.value,
                    "chunk_index": len(results),
                    "chunk_size": len(chunk_text),
                })
                results.append(SplitResult(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    index=len(results),
                    chunk_id=f"{metadata.get('document_id', 'doc')}_{len(results)}"
                ))

        return results

    def _split_semantic(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[SplitResult]:
        """
        Split text using semantic boundaries based on embedding similarity.

        This method calculates embeddings for sentences and splits where
        semantic similarity drops below a threshold, indicating a topic change.
        """
        if self._semantic_model is None:
            raise ValueError("Semantic splitter requires embedding model")

        # Split into sentences first
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return self._split_recursive(text, metadata)

        # Calculate embeddings
        embeddings = self._semantic_model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(embeddings, sentences)

        # Create chunks based on boundaries
        chunks = self._create_chunks_from_boundaries(
            sentences, boundaries, metadata
        )

        # Ensure chunks don't exceed max size
        results = []
        chunk_idx = 0

        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                # Further split large chunks
                sub_splitter = RecursiveCharacterTextSplitter(
                    separators=self.separators,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    keep_separator=self.keep_separator,
                    strip_whitespace=self.strip_whitespace,
                    length_function=len,
                )
                sub_chunks = sub_splitter.split_text(chunk)

                for sub_chunk in sub_chunks:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "split_strategy": SplitStrategy.SEMANTIC.value,
                        "chunk_index": chunk_idx,
                        "chunk_size": len(sub_chunk),
                    })
                    results.append(SplitResult(
                        text=sub_chunk,
                        metadata=chunk_metadata,
                        index=len(results),
                        chunk_id=f"{metadata.get('document_id', 'doc')}_{len(results)}"
                    ))
                    chunk_idx += 1
            else:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "split_strategy": SplitStrategy.SEMANTIC.value,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk),
                })
                results.append(SplitResult(
                    text=chunk,
                    metadata=chunk_metadata,
                    index=len(results),
                    chunk_id=f"{metadata.get('document_id', 'doc')}_{len(results)}"
                ))
                chunk_idx += 1

        return results

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Chinese and English delimiters."""
        # Pattern for Chinese and English sentence delimiters
        pattern = r'([。！？.!?])'
        parts = re.split(pattern, text)

        sentences = []
        current = ""

        for i in range(0, len(parts), 2):
            sentence = parts[i]
            delimiter = parts[i + 1] if i + 1 < len(parts) else ""

            if sentence.strip():
                current += sentence + delimiter
                if delimiter:
                    sentences.append(current.strip())
                    current = ""

        if current.strip():
            sentences.append(current.strip())

        return [s for s in sentences if s.strip()]

    def _find_semantic_boundaries(
        self, embeddings: np.ndarray, sentences: List[str]
    ) -> List[int]:
        """
        Find semantic boundaries based on embedding similarity.

        Args:
            embeddings: Sentence embeddings
            sentences: Original sentences (for size consideration)

        Returns:
            List of boundary indices (end indices of chunks)
        """
        if len(embeddings) < 2:
            return [len(sentences)]

        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Calculate mean and standard deviation of similarities
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Set threshold for boundary detection
        # Boundary when similarity drops significantly below mean
        threshold = mean_sim - std_sim * 1.5

        # Find boundaries
        boundaries = []
        current_chunk_size = 0

        for idx, (sentence, sim) in enumerate(zip(sentences, similarities)):
            current_chunk_size += len(sentence)

            # Check for boundary conditions
            is_boundary = (
                sim < threshold or
                current_chunk_size >= self.chunk_size * 0.8
            )

            if is_boundary and idx > 0:
                boundaries.append(idx + 1)
                current_chunk_size = 0

        # Add final boundary if not already included
        if not boundaries or boundaries[-1] != len(sentences):
            boundaries.append(len(sentences))

        return boundaries

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Create text chunks from sentence boundaries."""
        chunks = []
        prev_boundary = 0

        for boundary in boundaries:
            chunk_sentences = sentences[prev_boundary:boundary]
            chunk = "".join(chunk_sentences)
            if chunk.strip():
                chunks.append(chunk.strip())
            prev_boundary = boundary

        return chunks

    def _split_hybrid(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[SplitResult]:
        """
        Hybrid splitting strategy combining multiple approaches.

        This strategy:
        1. First tries Markdown header splitting if applicable
        2. Falls back to recursive splitting for large chunks
        3. Uses semantic splitting as a final refinement
        """
        # Check if text has Markdown structure
        has_markdown = bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))

        if has_markdown:
            # Try Markdown splitting first
            markdown_chunks = self._split_markdown(text, metadata)
            if self._are_chunks_sized_well(markdown_chunks):
                return markdown_chunks

        # Fall back to recursive splitting
        recursive_chunks = self._split_recursive(text, metadata)

        # If semantic model is available, refine using semantic boundaries
        if self._semantic_model is not None:
            return self._refine_with_semantic(recursive_chunks)

        return recursive_chunks

    def _are_chunks_sized_well(self, chunks: List[SplitResult]) -> bool:
        """Check if chunks are well-sized (not too many very small or large chunks)."""
        if not chunks:
            return False

        sizes = [len(chunk.text) for chunk in chunks]

        # Check if we have good distribution
        too_small = sum(1 for s in sizes if s < self.chunk_size * 0.2)
        too_large = sum(1 for s in sizes if s > self.chunk_size * 1.5)

        # Accept if less than 20% of chunks are problematic
        total = len(chunks)
        return (too_small / total < 0.2) and (too_large / total < 0.2)

    def _refine_with_semantic(
        self, chunks: List[SplitResult]
    ) -> List[SplitResult]:
        """Refine chunks using semantic boundaries."""
        # For now, return as-is
        # Could implement more sophisticated refinement here
        return chunks


class DocumentSplitterFactory:
    """Factory for creating configured document splitters."""

    @staticmethod
    def create_chinese_optimized(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> DocumentSplitter:
        """Create a splitter optimized for Chinese text."""
        return DocumentSplitter(
            strategy=SplitStrategy.RECURSIVE,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "，",
                " ",
                "",
            ],
        )

    @staticmethod
    def create_english_optimized(
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> DocumentSplitter:
        """Create a splitter optimized for English text."""
        return DocumentSplitter(
            strategy=SplitStrategy.RECURSIVE,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ";",
                ",",
                " ",
                "",
            ],
        )

    @staticmethod
    def create_markdown_aware(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> DocumentSplitter:
        """Create a splitter that respects Markdown structure."""
        return DocumentSplitter(
            strategy=SplitStrategy.MARKDOWN_HEADER,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def create_semantic_aware(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: Optional[str] = None,
    ) -> DocumentSplitter:
        """Create a splitter that uses semantic boundaries."""
        return DocumentSplitter(
            strategy=SplitStrategy.SEMANTIC,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )

    @staticmethod
    def create_hybrid(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: Optional[str] = None,
    ) -> DocumentSplitter:
        """Create a hybrid splitter that combines multiple strategies."""
        return DocumentSplitter(
            strategy=SplitStrategy.HYBRID,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )


# Convenience functions for common use cases
def split_text_chinese(
    text: str,
    document_id: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[SplitResult]:
    """Convenience function for splitting Chinese text."""
    splitter = DocumentSplitterFactory.create_chinese_optimized(
        chunk_size, chunk_overlap
    )
    return splitter.split_text(text, document_id=document_id)


def split_text_markdown(
    text: str,
    document_id: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[SplitResult]:
    """Convenience function for splitting Markdown text."""
    splitter = DocumentSplitterFactory.create_markdown_aware(
        chunk_size, chunk_overlap
    )
    return splitter.split_text(text, document_id=document_id)


def split_text_semantic(
    text: str,
    document_id: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: Optional[str] = None,
) -> List[SplitResult]:
    """Convenience function for splitting text using semantic boundaries."""
    splitter = DocumentSplitterFactory.create_semantic_aware(
        chunk_size, chunk_overlap, embedding_model
    )
    return splitter.split_text(text, document_id=document_id)


# Export main classes and functions
__all__ = [
    "SplitStrategy",
    "SplitResult",
    "DocumentSplitter",
    "DocumentSplitterFactory",
    "split_text_chinese",
    "split_text_markdown",
    "split_text_semantic",
]
