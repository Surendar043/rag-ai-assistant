# RAG AI Assistant

An AI-powered question-answering application that answers user questions using
retrieved document context rather than relying only on general model knowledge.

## Features

- **Document pipeline**: chunking, embeddings, vector storage, and retrieval
- **LangGraph agent**: selects and calls tools (retrieve documents) to answer questions
- **Structured evaluation**: retrieval quality (recall@k, MRR) measured separately from answer quality
- **Edge-case tests**: out-of-scope queries, empty input, nonsense text
- **Clean separation**: API layer, retrieval logic, and model components are decoupled
- **FastAPI REST API**: `/ingest` to index documents, `/ask` to ask questions

## Repository Layout

```
rag-ai-assistant/
├── config/           # central configuration
├── data/             # raw documents + processed index (gitignored)
├── eval/             # evaluation dataset + metrics
├── src/
│   ├── api/          # FastAPI layer (HTTP only)
│   ├── agent/        # LangGraph agent (tool selection + calling)
│   ├── models/       # LLM + embedding wrappers
│   ├── pipeline/     # chunking, embeddings, vector store, retrieval, generation
│   └── utils/        # logging, helpers
├── tests/            # unit tests
├── main.py           # local CLI runner
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd rag-ai-assistant

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your settings (e.g., LLM API keys)
```

## Usage

### Start the API

```bash
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Index documents

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"text": "The refund policy allows returns within 30 days with a receipt.", "source": "policy.pdf"},
      {"text": "Reset your password at Settings > Security > Forgot Password.", "source": "user_guide.pdf"}
    ]
  }'
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

### Local CLI

```bash
python main.py
```

## Evaluation

Run the evaluation suite to measure retrieval and answer quality separately:

```bash
python eval/evaluate.py
```

Sample output:
```
Retrieval Recall@5: 0.800
Answer Correctness: 1.000
```

## Testing

```bash
pytest
```

## Project Structure by Concern

| Layer | Responsibility | Files |
|-------|---------------|-------|
| API | HTTP request/response, validation | `src/api/` |
| Pipeline | chunking, embeddings, retrieval, generation | `src/pipeline/` |
| Agent | LangGraph tool selection & calling | `src/agent/` |
| Models | LLM + embedding wrappers | `src/models/` |
| Eval | dataset + metrics (retrieval vs. answer) | `eval/` |

## License

MIT
