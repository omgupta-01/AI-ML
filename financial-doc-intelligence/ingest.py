"""
ingest.py — One-time script to build / refresh the FAISS index from local PDFs
Usage: python ingest.py --pdf-dir ./data/pdfs
"""

import argparse
from pathlib import Path
from app import build_vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS index.")
    parser.add_argument("--pdf-dir", default="./data/pdfs", help="Directory with PDF files")
    args = parser.parse_args()

    pdf_dir   = Path(args.pdf_dir)
    pdf_paths = [str(p) for p in pdf_dir.glob("*.pdf")]

    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        exit(1)

    print(f"Found {len(pdf_paths)} PDFs. Building index …")
    build_vectorstore(pdf_paths)
    print("Done! Run: uvicorn app:app --reload")
