#!/usr/bin/env python3
"""
RAG Database Setup Script
=========================

This script helps you populate the RAG (Retrieval-Augmented Generation) database
with your course materials. The AI will use these materials when testing quiz
questions in "With RAG" mode.

Usage:
    python scripts/setup_rag.py --add-file lecture_notes.txt
    python scripts/setup_rag.py --add-folder ./course_materials/
    python scripts/setup_rag.py --status
    python scripts/setup_rag.py --clear

Supported file types: .txt, .md, .pdf (requires pypdf), .docx (requires python-docx)
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("ERROR: chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME


def get_collection():
    """Get or create the RAG collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    return collection


def read_text_file(filepath):
    """Read a text file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def read_pdf_file(filepath):
    """Read a PDF file (requires pypdf)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"WARNING: pypdf not installed. Skipping {filepath}")
        print("  Install with: pip install pypdf")
        return None

    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def read_docx_file(filepath):
    """Read a Word document (requires python-docx)."""
    try:
        from docx import Document
    except ImportError:
        print(f"WARNING: python-docx not installed. Skipping {filepath}")
        print("  Install with: pip install python-docx")
        return None

    doc = Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def read_file(filepath):
    """Read a file based on its extension."""
    ext = Path(filepath).suffix.lower()

    if ext in ['.txt', '.md']:
        return read_text_file(filepath)
    elif ext == '.pdf':
        return read_pdf_file(filepath)
    elif ext == '.docx':
        return read_docx_file(filepath)
    else:
        print(f"WARNING: Unsupported file type: {ext}. Skipping {filepath}")
        return None


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def add_file(filepath, collection):
    """Add a single file to the RAG database."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return 0

    print(f"Processing: {filepath.name}")

    text = read_file(filepath)
    if text is None:
        return 0

    chunks = chunk_text(text)
    if not chunks:
        print(f"  No content to add")
        return 0

    # Generate unique IDs for each chunk
    base_id = filepath.stem.replace(" ", "_")[:50]
    ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]

    # Add to collection
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": str(filepath), "chunk": i} for i in range(len(chunks))]
    )

    print(f"  Added {len(chunks)} chunks")
    return len(chunks)


def add_folder(folderpath, collection):
    """Add all supported files from a folder."""
    folderpath = Path(folderpath)
    if not folderpath.exists():
        print(f"ERROR: Folder not found: {folderpath}")
        return 0

    supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
    total_chunks = 0

    for filepath in folderpath.rglob('*'):
        if filepath.suffix.lower() in supported_extensions:
            total_chunks += add_file(filepath, collection)

    return total_chunks


def show_status(collection):
    """Show the current status of the RAG database."""
    count = collection.count()
    print(f"\nRAG Database Status")
    print(f"{'='*40}")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Location: {CHROMA_DB_PATH}")
    print(f"Total chunks: {count}")

    if count > 0:
        # Show a sample of sources
        results = collection.peek(limit=10)
        sources = set()
        for metadata in results.get('metadatas', []):
            if metadata and 'source' in metadata:
                sources.add(Path(metadata['source']).name)

        print(f"\nSample sources:")
        for source in list(sources)[:5]:
            print(f"  - {source}")


def clear_database(collection):
    """Clear all documents from the RAG database."""
    count = collection.count()
    if count == 0:
        print("Database is already empty.")
        return

    response = input(f"Are you sure you want to delete all {count} chunks? (yes/no): ")
    if response.lower() == 'yes':
        # Get all IDs and delete
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
        print(f"Deleted {count} chunks.")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(description="Manage the RAG database for course materials")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--add-file', metavar='FILE', help='Add a single file to the database')
    group.add_argument('--add-folder', metavar='FOLDER', help='Add all supported files from a folder')
    group.add_argument('--status', action='store_true', help='Show database status')
    group.add_argument('--clear', action='store_true', help='Clear all documents from the database')

    args = parser.parse_args()

    collection = get_collection()

    if args.add_file:
        chunks = add_file(args.add_file, collection)
        print(f"\nTotal chunks added: {chunks}")
        show_status(collection)

    elif args.add_folder:
        chunks = add_folder(args.add_folder, collection)
        print(f"\nTotal chunks added: {chunks}")
        show_status(collection)

    elif args.status:
        show_status(collection)

    elif args.clear:
        clear_database(collection)


if __name__ == '__main__':
    main()
