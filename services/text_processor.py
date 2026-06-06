"""
Text processing service: cleaning, normalization, and token-aware chunking.
Ensures LLM inputs stay within context limits while preserving semantic coherence.
"""

import re
import logging
from typing import Iterator

from config import config

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles text cleaning and chunking for LLM consumption.

    Responsibilities:
    - Normalize and clean raw input text
    - Split text into token-safe chunks with configurable overlap
    - Provide chunk metadata for downstream merging
    """

    def __init__(self) -> None:
        self._max_chunk_chars: int = int(
            config.chunking.max_chunk_tokens * config.chunking.chars_per_token
        )
        self._overlap_chars: int = int(
            config.chunking.overlap_tokens * config.chunking.chars_per_token
        )

    # ------------------------------------------------------------------
    # Public API
    # ----
    #--------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Normalize raw text for LLM consumption.

        Steps:
        1. Collapse whitespace
        2. Remove control characters
        3. Normalize punctuation
        4. Strip excessive blank lines

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        text = self._remove_control_characters(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_punctuation(text)
        text = text.strip()
        logger.debug("Cleaned text length: %d chars", len(text))
        return text

    def chunk(self, text: str) -> list[str]:
        """
        Split text into overlapping, token-safe chunks.

        Uses sentence-boundary-aware splitting to avoid cutting mid-sentence.
        Applies a sliding overlap to preserve cross-chunk context.

        Args:
            text: Cleaned input text.

        Returns:
            List of text chunks, each within the configured token limit.
        """
        if len(text) <= self._max_chunk_chars:
            logger.debug("Text fits in a single chunk.")
            return [text]

        sentences = self._split_into_sentences(text)
        chunks = list(self._build_chunks(sentences))
        logger.info("Text split into %d chunk(s).", len(chunks))
        return chunks

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using character-based heuristic.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return max(1, int(len(text) / config.chunking.chars_per_token))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """Strip non-printable control characters except newlines and tabs."""
        return re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces/tabs; reduce multiple blank lines to one."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        """Normalize curly quotes and dashes to ASCII equivalents."""
        replacements = {
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-",
            "\u2026": "...",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences using punctuation heuristics.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        # Split on sentence-ending punctuation followed by whitespace
        raw = re.split(r"(?<=[.!?])\s+", text)
        # Further split very long "sentences" at paragraph breaks
        sentences: list[str] = []
        for seg in raw:
            if "\n\n" in seg:
                sentences.extend(p.strip() for p in seg.split("\n\n") if p.strip())
            else:
                sentences.append(seg.strip())
        return [s for s in sentences if s]

    def _build_chunks(self, sentences: list[str]) -> Iterator[str]:
        """
        Greedily pack sentences into chunks respecting the character limit.
        Applies a trailing overlap from the previous chunk for context continuity.

        Args:
            sentences: List of sentence strings.

        Yields:
            Text chunk strings.
        """
        current: list[str] = []
        current_len: int = 0
        overlap_buffer: str = ""

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for space

            if current_len + sentence_len > self._max_chunk_chars and current:
                chunk_text = overlap_buffer + " ".join(current)
                yield chunk_text.strip()
                # Prepare overlap: last N chars of current chunk
                joined = " ".join(current)
                overlap_buffer = joined[-self._overlap_chars:] + " " if self._overlap_chars else ""
                current = []
                current_len = 0

            current.append(sentence)
            current_len += sentence_len

        if current:
            yield (overlap_buffer + " ".join(current)).strip()