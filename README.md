# AI Quiz Generator Microservice

Production-ready FastAPI microservice that generates high-quality MCQs from educational text using Groq (LLaMA 3).

## Architecture

```
quiz_service/
├── app.py                    # API layer (FastAPI)
├── config.py                 # Environment config (singleton)
├── services/
│   ├── llm_service.py        # LLM orchestration, chunking, retries
│   └── text_processor.py     # Text cleaning + token-aware chunking
├── utils/
│   └── json_parser.py        # Safe LLM output parsing + validation
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY
python app.py
```

## API

### POST /generate

**Request:**
```json
{
  "text": "Educational content here...",
  "num_questions": 5,
  "difficulty": "medium"
}
```

**Response:**
```json
{
  "mcqs": [
    {
      "question": "What is ...?",
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D",
      "answer": "B"
    }
  ]
}
```

### GET /health

```json
{"status": "ok", "service": "ai-quiz-generator"}
```
