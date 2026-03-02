"""
Semantic refiner for document chunks.

Uses LLM to refine chunk content, extract keywords, and identify entities.

Author: RAG Team
Created: 2026-03-02
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from utils.config_init import application_conf
from utils.logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class RefinementConfig:
    """Configuration for semantic refinement."""

    enabled: bool = True
    llm_model: str = "basic"
    max_keywords: int = 10
    max_entities: int = 5
    summary_max_length: int = 200


class RefinementError(Exception):
    """Base exception for refinement errors."""

    pass


@dataclass
class SemanticRefinementResult:
    """Result of semantic refinement for a chunk."""

    chunk_id: str
    refined_summary: str
    keywords: List[str]
    entities: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class SemanticRefiner:
    """
    Refines document chunks using LLM.

    Performs:
    - Content summarization
    - Keyword extraction
    - Entity extraction
    """

    # Prompt template for refinement
    REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的文档内容分析助手。你的任务是对给定的文本片段进行语义精炼。

请对以下内容进行处理：
1. 总结：用简洁的语言概括文本的核心内容（不超过 {summary_max_length} 字）
2. 关键词：提取 {max_keywords} 个最重要的关键词
3. 实体：识别 {max_entities} 个重要的实体（人名、地名、组织名、时间等）

请以JSON格式返回结果，格式如下：
{{
    "summary": "简洁的摘要",
    "keywords": ["关键词1", "关键词2", ...],
    "entities": ["实体1", "实体2", ...]
}}"""),
        ("user", "{content}")
    ])

    def __init__(self, config: Optional[RefinementConfig] = None):
        """
        Initialize the semantic refiner.

        Args:
            config: Refinement configuration. If None, loads from application config.
        """
        if config is None:
            config = self._load_config_from_file()

        self.config = config
        self._llm: Optional[Runnable] = None

        if self.config.enabled:
            self._init_llm()

    def _load_config_from_file(self) -> RefinementConfig:
        """Load configuration from application.yaml."""
        try:
            enabled = application_conf.get_properties("semantic_refinement.enabled", True)
            llm_model = application_conf.get_properties("semantic_refinement.llm_model", "basic")
            max_keywords = application_conf.get_properties("semantic_refinement.max_keywords", 10)
            max_entities = application_conf.get_properties("semantic_refinement.max_entities", 5)
            summary_max_length = application_conf.get_properties("semantic_refinement.summary_max_length", 200)

            return RefinementConfig(
                enabled=enabled,
                llm_model=llm_model,
                max_keywords=max_keywords,
                max_entities=max_entities,
                summary_max_length=summary_max_length,
            )
        except Exception as e:
            logger.warning(f"Failed to load refinement config, using defaults: {e}")
            return RefinementConfig()

    def _init_llm(self):
        """Initialize the LLM based on configuration."""
        try:
            from bz_agent.agents.llm import get_llm_by_type

            self._llm = get_llm_by_type(self.config.llm_model)
            logger.info(f"Initialized LLM for semantic refinement: {self.config.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.config.enabled = False

    def refine_chunk(self, chunk_id: str, content: str) -> SemanticRefinementResult:
        """
        Refine a single chunk of content.

        Args:
            chunk_id: Unique identifier for the chunk
            content: The chunk content to refine

        Returns:
            SemanticRefinementResult with refined data

        Raises:
            RefinementError: If refinement fails
        """
        if not self.config.enabled:
            # Return original content as summary if refinement is disabled
            return SemanticRefinementResult(
                chunk_id=chunk_id,
                refined_summary=content[:self.config.summary_max_length],
                keywords=[],
                entities=[],
                success=True,
                error="Refinement disabled"
            )

        try:
            # Prepare the prompt with configuration
            prompt = self.REFINEMENT_PROMPT.format(
                content=content,
                summary_max_length=self.config.summary_max_length,
                max_keywords=self.config.max_keywords,
                max_entities=self.config.max_entities
            )

            # Invoke LLM
            logger.debug(f"Refining chunk {chunk_id}, content length: {len(content)}")
            response = self._llm.invoke(prompt)
            result_text = response.content

            # Parse JSON response
            parsed = self._parse_refinement_result(result_text)

            return SemanticRefinementResult(
                chunk_id=chunk_id,
                refined_summary=parsed.get("summary", content[:self.config.summary_max_length]),
                keywords=parsed.get("keywords", []),
                entities=parsed.get("entities", []),
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to refine chunk {chunk_id}: {e}")
            # Return fallback result
            return SemanticRefinementResult(
                chunk_id=chunk_id,
                refined_summary=content[:self.config.summary_max_length],
                keywords=[],
                entities=[],
                success=False,
                error=str(e)
            )

    def refine_chunks(self, chunks: List[tuple]) -> List[SemanticRefinementResult]:
        """
        Refine multiple chunks.

        Args:
            chunks: List of (chunk_id, content) tuples

        Returns:
            List of SemanticRefinementResult
        """
        results = []

        for chunk_id, content in chunks:
            result = self.refine_chunk(chunk_id, content)
            results.append(result)

        return results

    def _parse_refinement_result(self, result_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured data.

        Args:
            result_text: Raw LLM response

        Returns:
            Parsed dictionary with summary, keywords, entities

        Raises:
            RefinementError: If parsing fails
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*"summary"[^{}]*"keywords"[^{}]*"entities"[^{}]*\}', result_text, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from LLM response: {e}")

        # Fallback: try to parse the entire response as JSON
        try:
            parsed = json.loads(result_text)
            return parsed
        except json.JSONDecodeError:
            pass

        # Final fallback: extract from text
        return self._extract_from_text(result_text)

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from text when JSON parsing fails.

        Args:
            text: The LLM response text

        Returns:
            Dictionary with extracted data
        """
        result = {
            "summary": "",
            "keywords": [],
            "entities": []
        }

        # Extract summary (look for summary section)
        summary_patterns = [
            r'(?:摘要|总结|summary)[:：]\s*(.*?)(?=\n\s*(?:关键词|实体|keywords|entities)|$)',
            r'["\']summary["\']\s*:\s*["\']([^"\']+)["\']',
        ]

        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["summary"] = match.group(1).strip()[:self.config.summary_max_length]
                break

        # Extract keywords
        keywords_patterns = [
            r'(?:关键词|keywords)[:：]\s*\[(.*?)\]',
            r'["\']keywords["\']\s*:\s*\[(.*?)\]',
            r'(?:关键词|keywords)[:：]\s*([^\n]+)',
        ]

        for pattern in keywords_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                if "[" in keywords_text:
                    keywords = [k.strip().strip('"\'') for k in keywords_text.split(",")]
                else:
                    keywords = [k.strip() for k in keywords_text.split(",")]
                result["keywords"] = keywords[:self.config.max_keywords]
                break

        # Extract entities
        entities_patterns = [
            r'(?:实体|entities)[:：]\s*\[(.*?)\]',
            r'["\']entities["\']\s*:\s*\[(.*?)\]',
            r'(?:实体|entities)[:：]\s*([^\n]+)',
        ]

        for pattern in entities_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities_text = match.group(1)
                if "[" in entities_text:
                    entities = [e.strip().strip('"\'') for e in entities_text.split(",")]
                else:
                    entities = [e.strip() for e in entities_text.split(",")]
                result["entities"] = entities[:self.config.max_entities]
                break

        return result


# ============================================================================
# Convenience functions
# ============================================================================


def refine_chunk(
    chunk_id: str,
    content: str,
    config: Optional[RefinementConfig] = None
) -> SemanticRefinementResult:
    """
    Convenience function to refine a single chunk.

    Args:
        chunk_id: Unique identifier for the chunk
        content: The chunk content to refine
        config: Refinement configuration

    Returns:
        SemanticRefinementResult with refined data
    """
    refiner = SemanticRefiner(config)
    return refiner.refine_chunk(chunk_id, content)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "RefinementConfig",
    "RefinementError",
    "SemanticRefinementResult",
    "SemanticRefiner",
    "refine_chunk",
]
