"""
AI Quiz Generator — FastAPI application layer.
Exposes the POST /generate endpoint and handles request validation,
error translation, and structured response formatting.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import config
from services.llm_service import LLMService

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: initialize/teardown shared resources
# ---------------------------------------------------------------------------

_llm_service: Optional[LLMService] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared services on startup; clean up on shutdown."""
    global _llm_service
    logger.info("Initializing LLM service...")
    _llm_service = LLMService()
    logger.info("AI Quiz Generator ready.")
    yield
    logger.info("Shutting down AI Quiz Generator.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Quiz Generator",
    description="Generate high-quality multiple-choice questions from educational text using LLMs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Validated request payload for /generate."""

    text: Annotated[str, Field(min_length=50, description="Educational source text.")]
    num_questions: Annotated[
        int,
        Field(
            ge=config.min_questions_per_request,
            le=config.max_questions_per_request,
            description=f"Number of MCQs to generate ({config.min_questions_per_request}–{config.max_questions_per_request}).",
        ),
    ] = 5
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v


class MCQItem(BaseModel):
    """Single MCQ entry in the response."""

    question: str
    A: str
    B: str
    C: str
    D: str
    answer: Literal["A", "B", "C", "D"]


class GenerateResponse(BaseModel):
    """Response envelope for /generate."""

    mcqs: list[MCQItem]


class ErrorResponse(BaseModel):
    """Structured error envelope."""

    detail: str
    code: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "ai-quiz-generator"}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate MCQs from educational text",
    tags=["Quiz"],
)
async def generate_mcqs(request: GenerateRequest) -> GenerateResponse:
    """
    Generate multiple-choice questions from the provided educational text.

    - Accepts plain text; PDF/DOCX extraction is the caller's responsibility.
    - Handles arbitrarily long documents via internal chunking.
    - Returns exactly `num_questions` validated MCQs (or fewer if the text
      does not contain enough distinct information).

    Args:
        request: Validated GenerateRequest payload.

    Returns:
        GenerateResponse containing the list of MCQs.
    """
    if _llm_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is not initialized.",
        )

    logger.info(
        "Received /generate request: num_questions=%d, difficulty=%s, text_length=%d",
        request.num_questions,
        request.difficulty,
        len(request.text),
    )

    try:
        mcqs = _llm_service.generate_mcqs(
            text=request.text,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
        )
    except ValueError as exc:
        logger.warning("Validation error during generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("Runtime error during generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during generation.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc

    logger.info("Successfully generated %d MCQ(s).", len(mcqs))
    return GenerateResponse(mcqs=[MCQItem(**mcq) for mcq in mcqs])


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Catch-all handler to prevent stack trace leakage."""
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "code": "INTERNAL_ERROR"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )