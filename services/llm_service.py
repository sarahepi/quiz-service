"""
LLM service: orchestrates Groq API calls to generate MCQs from text chunks.
Handles prompt engineering, retry logic, and multi-chunk result merging.
"""

import logging
import math
import time
from typing import Literal

from groq import Groq, APIError, RateLimitError, APITimeoutError

from config import config
from services.text_processor import TextProcessor
from utils.json_parser import parse_llm_response, deduplicate_mcqs

logger = logging.getLogger(__name__)

DifficultyLevel = Literal["easy", "medium", "hard"]

_DIFFICULTY_INSTRUCTIONS: dict[str, str] = {
    "easy": (
        "Questions should test basic recall and comprehension. "
        "Use simple, direct language. Distractors should be clearly wrong but plausible."
    ),
    "medium": (
        "Questions should test understanding and application of concepts. "
        "Distractors must be subtly wrong and require careful reading to eliminate."
    ),
    "hard": (
        "Questions should test analysis, synthesis, and evaluation. "
        "Distractors must be highly plausible and require deep domain knowledge to eliminate. "
        "Avoid trivially obvious wrong answers."
    ),
}


def _build_prompt(text: str, num_questions: int, difficulty: DifficultyLevel) -> str:
    """
    Construct a strict, hallucination-resistant prompt for MCQ generation.

    Args:
        text: Educational source text.
        num_questions: Number of MCQs to generate.
        difficulty: Desired difficulty level.

    Returns:
        Formatted prompt string.
    """
    difficulty_instruction = _DIFFICULTY_INSTRUCTIONS[difficulty]

    return f"""You are an expert educational assessment designer. Your task is to generate exactly {num_questions} high-quality multiple-choice questions (MCQs) from the provided text.

STRICT RULES:
1. All questions MUST be directly derived from the provided text. Do NOT invent facts.
2. Each question must have exactly 4 options: A, B, C, D.
3. Exactly one option must be correct. The answer key must be one of: A, B, C, D.
4. Distractors (wrong options) must be plausible but clearly incorrect upon careful reading.
5. Do NOT repeat similar questions or use overlapping concepts across questions.
6. Questions must be self-contained; avoid references like "according to the passage."
7. Do NOT add explanations, commentary, or any text outside the JSON.

DIFFICULTY: {difficulty.upper()}
{difficulty_instruction}

OUTPUT FORMAT (strict JSON, no markdown, no extra text):
{{
  "mcqs": [
    {{
      "question": "<question text>",
      "A": "<option A>",
      "B": "<option B>",
      "C": "<option C>",
      "D": "<option D>",
      "answer": "<A|B|C|D>"
    }}
  ]
}}

SOURCE TEXT:
\"\"\"
{text}
\"\"\"

Generate exactly {num_questions} MCQs now:"""


class LLMService:
    """
    Orchestrates LLM-based MCQ generation with chunking, retries, and deduplication.

    Responsibilities:
    - Delegate text preprocessing to TextProcessor
    - Build and send prompts to Groq API
    - Retry on transient errors with exponential backoff
    - Merge and deduplicate results across chunks
    - Cap output to requested question count
    """

    def __init__(self) -> None:
        self._client = Groq(api_key=config.llm.api_key)
        self._processor = TextProcessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_mcqs(
        self,
        text: str,
        num_questions: int,
        difficulty: DifficultyLevel = "medium",
    ) -> list[dict]:
        """
        Generate MCQs from educational text.

        Handles long documents by chunking and distributing the question
        quota proportionally across chunks, then deduplicating results.

        Args:
            text: Raw educational text (any length).
            num_questions: Total number of MCQs to return.
            difficulty: Desired difficulty level.

        Returns:
            List of validated MCQ dictionaries (length == num_questions).

        Raises:
            ValueError: If the text is empty or generation fails completely.
            RuntimeError: If the LLM API is persistently unavailable.
        """
        cleaned = self._processor.clean(text)
        if not cleaned:
            raise ValueError("Input text is empty after cleaning.")

        chunks = self._processor.chunk(cleaned)
        logger.info(
            "Processing %d chunk(s) for %d question(s) at %s difficulty.",
            len(chunks),
            num_questions,
            difficulty,
        )

        questions_per_chunk = self._distribute_questions(num_questions, len(chunks))
        all_mcqs: list[dict] = []

        for idx, (chunk, q_count) in enumerate(zip(chunks, questions_per_chunk)):
            if q_count == 0:
                continue
            logger.info("Generating %d MCQ(s) from chunk %d/%d.", q_count, idx + 1, len(chunks))
            mcqs = self._generate_from_chunk(chunk, q_count, difficulty)
            all_mcqs.extend(mcqs)

        unique_mcqs = deduplicate_mcqs(all_mcqs)
        logger.info(
            "Total MCQs before dedup: %d, after dedup: %d.", len(all_mcqs), len(unique_mcqs)
        )

        if not unique_mcqs:
            raise RuntimeError("MCQ generation produced zero valid questions.")

        return unique_mcqs[:num_questions]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_from_chunk(
        self,
        chunk: str,
        num_questions: int,
        difficulty: DifficultyLevel,
    ) -> list[dict]:
        """
        Call the LLM for a single chunk with retry logic.

        Args:
            chunk: Text chunk to generate questions from.
            num_questions: Questions to request for this chunk.
            difficulty: Difficulty level.

        Returns:
            List of valid MCQ dicts (may be fewer than requested on partial failures).
        """
        prompt = _build_prompt(chunk, num_questions, difficulty)
        last_error: Exception | None = None

        for attempt in range(1, config.llm.max_retries + 1):
            try:
                logger.debug("LLM call attempt %d/%d.", attempt, config.llm.max_retries)
                response = self._client.chat.completions.create(
                    model=config.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                )
                raw_content = response.choices[0].message.content or ""
                mcqs = parse_llm_response(raw_content)
                logger.debug("Parsed %d MCQ(s) from chunk.", len(mcqs))
                return mcqs

            except (RateLimitError, APITimeoutError) as exc:
                wait = config.llm.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient LLM error on attempt %d: %s. Retrying in %.1fs.",
                    attempt,
                    exc,
                    wait,
                )
                last_error = exc
                time.sleep(wait)

            except APIError as exc:
                logger.error("Non-retryable API error: %s", exc)
                raise RuntimeError(f"Groq API error: {exc}") from exc

            except ValueError as exc:
                logger.warning(
                    "Parse error on attempt %d: %s. Retrying.", attempt, exc
                )
                last_error = exc
                time.sleep(config.llm.retry_delay)

        raise RuntimeError(
            f"LLM generation failed after {config.llm.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _distribute_questions(total: int, num_chunks: int) -> list[int]:
        """
        Distribute question quota across chunks as evenly as possible.

        Args:
            total: Total questions requested.
            num_chunks: Number of text chunks.

        Returns:
            List of per-chunk question counts summing to >= total
            (last chunk absorbs the remainder).
        """
        if num_chunks == 1:
            return [total]

        base = math.floor(total / num_chunks)
        remainder = total - base * num_chunks
        distribution = [base] * num_chunks

        # Distribute extra questions to the first chunks
        for i in range(remainder):
            distribution[i] += 1

        # Request a small surplus per chunk to compensate for dedup losses
        surplus_factor = 1.3
        return [max(1, math.ceil(q * surplus_factor)) for q in distribution]