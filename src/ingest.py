#!/usr/bin/env python3
"""
Robust PDF ingestion script with diagnostics.

Usage:
  python src/ingest.py
  python src/ingest.py --folder data --out outputs/chunks.json --chunk_size 1200 --overlap 200 --recursive

This script:
 - Searches for PDF files (data/pdfs and data by default)
 - Extracts text with pdfplumber
 - Reports per-file stats (pages, chars, empty pages)
 - Saves non-empty overlapping chunks into outputs/chunks.json
"""
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import pdfplumber
import sys

def extract_text_from_pdf(pdf_path: Path):
    """
    Extract text from each page and return list of page texts.
    """
    page_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                try:
                    text = p.extract_text()
                except Exception as e:
                    # page extraction error
                    text = ""
                page_texts.append(text if text is not None else "")
    except Exception as e:
        print(f"[ERROR] Could not open {pdf_path}: {e}")
        return []
    return page_texts

def chunk_text_full(text: str, chunk_size: int = 1200, overlap: int = 200):
    chunks = []
    i = 0
    L = len(text)
    if L == 0:
        return []
    while i < L:
        chunk = text[i : i + chunk_size]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks

def find_pdfs(base_folder: Path, recursive: bool = False):
    found = []
    if recursive:
        found = list(base_folder.rglob("*.pdf"))
    else:
        found += list(base_folder.glob("*.pdf"))
    # keep unique, preserve order
    unique = []
    seen = set()
    for p in found:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def ingest_folder(pdf_folder: str = "data", out_path: str = "outputs/chunks.json", chunk_size:int=1200, overlap:int=200, recursive:bool=False):
    base = Path(pdf_folder)
    if not base.exists():
        print(f"[ERROR] Folder does not exist: {base.resolve()}")
        return []

    pdfs = find_pdfs(base, recursive=recursive)
    if not pdfs:
        print(f"[WARNING] No PDF files found under {base.resolve()} or {base.resolve()/ 'pdfs'}.")
        print("Place PDFs in 'data/' or 'data/pdfs/' or run with --recursive to search subfolders.")
        return []

    print(f"Found {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(" -", p.name)

    all_chunks = []
    file_summaries = []

    for pdf in tqdm(pdfs, desc="Processing PDFs"):
        page_texts = extract_text_from_pdf(pdf)
        num_pages = len(page_texts)
        chars_per_page = [len(t) for t in page_texts]
        total_chars = sum(chars_per_page)
        empty_pages = sum(1 for c in chars_per_page if c == 0)
        avg_chars = (total_chars / num_pages) if num_pages>0 else 0

        file_summaries.append({
            "file": str(pdf),
            "pages": num_pages,
            "total_chars": total_chars,
            "empty_pages": empty_pages,
            "avg_chars_per_page": int(avg_chars)
        })

        # Quick heuristic: scanned pdf likely if many pages but almost no chars
        if num_pages >= 1 and avg_chars < 50:
            print(f"{pdf.name} appears to be scanned/contains images (avg {avg_chars:.1f} chars/page)")

        # combine pages & chunk
        full_text = "\n\n".join([t for t in page_texts if t and t.strip()])
        if not full_text or len(full_text.strip())==0:
            # No machine-readable text found in this PDF
            print(f"[INFO] No extractable text in {pdf.name} — skipping.")
            continue

        chunks = chunk_text_full(full_text, chunk_size=chunk_size, overlap=overlap)
        for idx, c in enumerate(chunks):
            all_chunks.append({
                "id": f"{pdf.stem}_{idx}",
                "source": pdf.name,
                "text": c
            })

    # save chunks
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n=== INGEST SUMMARY ===")
    for s in file_summaries:
        print(f"{Path(s['file']).name}: pages={s['pages']}, chars={s['total_chars']}, empty_pages={s['empty_pages']}, avg_chars/page={s['avg_chars_per_page']}")

    print(f"\nSaved {len(all_chunks)} chunks -> {outp.resolve()}")
    return all_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs -> chunks.json (diagnostic version)")
    parser.add_argument("--folder", type=str, default="data", help="Folder with PDFs (default: data). Will also check data/pdfs.")
    parser.add_argument("--out", type=str, default="outputs/chunks.json", help="Output JSON path")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--recursive", action="store_true", help="Search recursively under folder")
    args = parser.parse_args()

    ingest_folder(pdf_folder=args.folder, out_path=args.out, chunk_size=args.chunk_size, overlap=args.overlap, recursive=args.recursive)
