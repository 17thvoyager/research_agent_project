"""
backfill_docs.py — Run this ONCE to register existing PDFs in data/ 
for a specific user. Use this if you uploaded papers before the 
per-user ownership system was added.

Usage:
    python backfill_docs.py aswin
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import db
from pathlib import Path

DATA_DIR = "./data"

def backfill(username: str):
    pdfs = list(Path(DATA_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}/")
        return

    print(f"Registering {len(pdfs)} file(s) for user '{username}':")
    for pdf in pdfs:
        db.register_document(username, pdf.name)
        print(f"  ✅  {pdf.name}")
    print("\nDone! Restart the API and log in — your papers will appear.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backfill_docs.py <username>")
        sys.exit(1)
    backfill(sys.argv[1].lower().strip())
