# EU AI Act RAG System

RAG system for querying the EU AI Act regulation.

## Architecture

```
PDF → Parser → Chunker → Embeddings → FAISS Index
                                          ↓
Query → Dense Search + BM25 Search → Fusion → Reranking → Top Chunks
                                                              ↓
                                          Context + Query → LLM → Answer + Citations
                                                              ↓
                                          Ground Truth → Metrics → Excel/JSON Reports
```

**Why This Architecture:**

- Hybrid retrieval (dense + sparse + reranking) for better accuracy
- Multi-provider LLM support for flexibility
- Comprehensive metrics for evaluation
- Environment-driven configuration

## Prerequisites

- Python 3.10+
- 8GB RAM minimum
- API key for at least one LLM provider (WatsonX AI, OpenAI, Anthropic, or Groq)

## Installation

```bash
# Setup environment
cd assessments
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### 1. Process Documents

```bash
# Step 1: Parse PDF
python scripts/process_pdf_enhanced.py

# Step 2: Build indexes
python scripts/build_embeddings.py
```

### 2. Run API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access at: `http://localhost:8000/docs`

### 3. Run Evaluations

**Interactive (single query):**

```bash
python scripts/live_evaluation.py
```

**Full evaluation (all ground truth queries):**

```bash
python scripts/live_evaluation.py --full
```

## Ground Truth Format

Create `data/ground_truth.json` using `data/ground_truth.example.json` as template:

```json
{
  "test_queries": [
    {
      "query_id": "unique_id",
      "query": "Your question",
      "query_type": "fact|reasoning|comparative|abstract",
      "expected_answer": "Correct answer",
      "relevant_chunk_ids": [],
      "expected_citations": ["Article 3(1)"],
      "difficulty": "easy|medium|hard",
      "category": "category_name"
    }
  ]
}
```

**Required fields:** query_id, query, query_type, expected_answer, expected_citations

## Evaluation Reports

**Excel Report** (3 sheets):

- Query Results: All queries with answers and metrics
- Metrics Summary: Average metrics across queries
- Retrieved Chunks: Detailed chunk information

**JSON Report**: Complete evaluation data for analysis

**Output location:** `results/live_evaluation_YYYYMMDD_HHMMSS.xlsx|json`

## Metrics

**Retrieval:** Precision@K, Recall@K, MRR, NDCG, MAP  
**Generation:** BLEU, ROUGE-1/2/L, BERTScore  
**Citations:** Precision, Recall, F1  
**Quality:** Answer length, chunk utilization, response time

## API Endpoints

- `GET /health` - System status and configuration check
- `POST /api/qa` - Question answering with comprehensive metrics
- `POST /api/chat` - Conversational interface with chat history
- `POST /api/retrieve` - Retrieve relevant chunks without generation

## Scripts

- `process_pdf_enhanced.py` - Parse PDF with page numbers and section detection
- `build_embeddings.py` - Generate embeddings and build FAISS/BM25 indexes
- `live_evaluation.py` - Interactive evaluation with Excel/JSON reports
- `run_20_questions.py` - Run 20 questions without ground truth comparison
- `test_retrieval.py` - Test hybrid retrieval system
- `test_query.py` - Test single query end-to-end

## Configuration

All settings in `.env`:

- LLM provider and API keys
- Chunk size and overlap
- Retrieval parameters (top_k, weights)
- Model names and parameters

## Project Structure

```
assessments/
├── api/                    # REST API
├── data/                   # Documents, indexes, ground truth
├── src/                    # Source code
│   ├── embeddings/         # Embedding generation
│   ├── evaluation/         # Metrics and evaluation
│   ├── generation/         # LLM generation
│   ├── ingestion/          # Document processing
│   ├── retrieval/          # Hybrid retrieval
│   └── utils/              # Utilities
├── scripts/                # Execution scripts
├── notebooks/              # Jupyter notebooks
├── results/                # Evaluation results
└── logs/                   # System logs
```
