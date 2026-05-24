# 📄 Financial Document Intelligence — RAG Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets financial analysts query thousands of PDF reports via natural language. Built with LangChain, OpenAI GPT, FAISS, and deployed as a FastAPI endpoint on AWS.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Lambda-yellow?logo=amazon-aws)

---

## 🚀 Features

- **RAG Pipeline** — Chunks, embeds, and indexes 3,000+ PDFs into a FAISS vector store
- **Natural Language Q&A** — Analysts ask questions in plain English; GPT-4o-mini answers from retrieved context
- **FastAPI Endpoint** — REST API with `/query` and `/rebuild-index` routes
- **AWS S3 Integration** — Auto-downloads PDFs from S3 on first run
- **Persistent Index** — FAISS index saved to disk; no re-embedding on restart
- **Source Attribution** — Every answer includes the source document(s)

---

## 🏗️ Architecture

```
PDFs (AWS S3)
     │
     ▼
PyPDFLoader → RecursiveCharacterTextSplitter → OpenAI Embeddings
                                                      │
                                                      ▼
                                               FAISS Vector Store
                                                      │
                User Query ──────────────────► Retriever (top-k)
                                                      │
                                                      ▼
                                              GPT-4o-mini (LLM)
                                                      │
                                                      ▼
                                                Final Answer
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/omgupta/financial-doc-intelligence.git
cd financial-doc-intelligence
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your OPENAI_API_KEY and AWS credentials
```

### 3. Ingest local PDFs (optional, instead of S3)

```bash
mkdir -p data/pdfs
# Copy your PDFs into data/pdfs/
python ingest.py --pdf-dir ./data/pdfs
```

### 4. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 📡 API Usage

### Query endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue growth in Q3 2023?"}'
```

**Response:**
```json
{
  "answer": "Revenue grew by 12.4% YoY in Q3 2023, driven by...",
  "sources": ["Q3_2023_Annual_Report.pdf", "Investor_Presentation_2023.pdf"]
}
```

### Rebuild index

```bash
curl -X POST http://localhost:8000/rebuild-index
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Documents indexed | 3,000+ PDFs |
| Analyst research time reduction | ~60% |
| Average query response time | < 3s |
| Embedding model | text-embedding-ada-002 |
| LLM | GPT-4o-mini |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Orchestration | LangChain |
| Vector Store | FAISS |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-ada-002 |
| PDF Parsing | PyPDF |
| API | FastAPI + Uvicorn |
| Storage | AWS S3 |

---

## 📁 Project Structure

```
financial-doc-intelligence/
├── app.py              # FastAPI app + RAG chain
├── ingest.py           # PDF ingestion script
├── requirements.txt
├── .env.example
├── data/
│   └── pdfs/           # Place your PDFs here
└── faiss_index/        # Auto-created after first ingest
```

---

## 👤 Author

**Om Gupta** · [omguptabca2020@gmail.com](mailto:omguptabca2020@gmail.com)  
Software Developer — Data & ML @ Bonanza Portfolio Ltd
