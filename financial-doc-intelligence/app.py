"""
Financial Document Intelligence — RAG Q&A System
Stack: LangChain · OpenAI GPT · FAISS · FastAPI · AWS S3
"""

import os
import boto3
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
AWS_BUCKET_NAME  = os.getenv("AWS_BUCKET_NAME", "your-s3-bucket")
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Document Intelligence API",
    description="RAG-based Q&A over financial PDF documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# ── Globals ───────────────────────────────────────────────────────────────────
vectorstore: Optional[FAISS] = None
qa_chain = None


# ── Helper Functions ──────────────────────────────────────────────────────────

def download_pdfs_from_s3(bucket: str, prefix: str = "reports/") -> list[str]:
    """Download PDFs from S3 to a temp directory and return local paths."""
    s3     = boto3.client("s3")
    tmpdir = tempfile.mkdtemp()
    paths  = []

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pdf"):
            local_path = Path(tmpdir) / Path(key).name
            s3.download_file(bucket, key, str(local_path))
            paths.append(str(local_path))
            print(f"  Downloaded: {key}")

    print(f"[S3] {len(paths)} PDFs downloaded.")
    return paths


def build_vectorstore(pdf_paths: list[str]) -> FAISS:
    """Load, chunk, embed, and index all PDFs into FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages  = loader.load()
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)
        print(f"  Indexed: {Path(path).name} → {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store      = FAISS.from_documents(all_docs, embeddings)
    store.save_local(FAISS_INDEX_PATH)
    print(f"[FAISS] Vector store saved with {len(all_docs)} total chunks.")
    return store


def load_or_build_vectorstore() -> FAISS:
    """Load existing FAISS index or build from S3 PDFs."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    if Path(FAISS_INDEX_PATH).exists():
        print("[FAISS] Loading existing index …")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[FAISS] Index not found — building from S3 …")
        pdf_paths = download_pdfs_from_s3(AWS_BUCKET_NAME)
        return build_vectorstore(pdf_paths)


def build_qa_chain(store: FAISS) -> RetrievalQA:
    """Build a RetrievalQA chain with a custom finance-focused prompt."""
    prompt_template = """You are a financial analyst assistant. Use the context below
to answer the question accurately. If the answer is not in the context, say
"I don't have enough information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global vectorstore, qa_chain
    print("Loading vector store …")
    vectorstore = load_or_build_vectorstore()
    qa_chain    = build_qa_chain(vectorstore)
    print("✅ RAG system ready.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": vectorstore is not None}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="System not ready yet.")

    result  = qa_chain({"query": req.question})
    answer  = result["result"]
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    })
    return QueryResponse(answer=answer, sources=sources)


@app.post("/rebuild-index")
async def rebuild_index():
    """Force-rebuild the FAISS index from S3."""
    global vectorstore, qa_chain
    pdf_paths   = download_pdfs_from_s3(AWS_BUCKET_NAME)
    vectorstore = build_vectorstore(pdf_paths)
    qa_chain    = build_qa_chain(vectorstore)
    return {"status": "Index rebuilt successfully."}
