"""
Course material (RAG) helper functions.

Extracted from App.py — no logic changes.
"""

import os
import streamlit as st
from config import CHROMA_DB_PATH, RAG_COLLECTION_PREFIX, get_display_name, get_rag_collection_name


def get_all_courses():
    """Get all course material collections from the database."""
    courses = []
    try:
        import chromadb
        if os.path.exists(str(CHROMA_DB_PATH)):
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            for coll in client.list_collections():
                if coll.name.startswith(RAG_COLLECTION_PREFIX):
                    display_name = get_display_name(coll.name)
                    # Get file count (unique source files)
                    files = get_course_files(coll.name)
                    file_count = len(files)
                    segment_count = coll.count()
                    courses.append({
                        'name': coll.name,
                        'display_name': display_name,
                        'file_count': file_count,
                        'segment_count': segment_count
                    })
    except:
        pass
    return courses


# Keep old function name as alias for compatibility
get_all_rag_collections = get_all_courses


def get_course_files(collection_name):
    """Get list of source files in a course with their segment counts."""
    files = {}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        coll = client.get_collection(collection_name)
        results = coll.get(include=['metadatas'])
        for meta in results.get('metadatas', []):
            source = meta.get('source', 'Unknown')
            files[source] = files.get(source, 0) + 1
    except:
        pass
    return files


# Keep old function name as alias for compatibility
get_collection_files = get_course_files


def delete_file_from_course(collection_name, filename):
    """Delete all segments from a specific file in a course."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        coll = client.get_collection(collection_name)
        # Get all IDs where source matches the filename
        results = coll.get(include=['metadatas'])
        ids_to_delete = []
        for i, meta in enumerate(results.get('metadatas', [])):
            if meta.get('source') == filename:
                ids_to_delete.append(results['ids'][i])
        if ids_to_delete:
            coll.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        return 0
    except Exception as e:
        return -1


def clear_course_materials(collection_name):
    """Delete all materials from a course."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        # Delete and recreate the collection to clear it
        client.delete_collection(collection_name)
        client.get_or_create_collection(name=collection_name)
        return True
    except Exception as e:
        return False


def process_uploaded_files(uploaded_files, collection_name, progress_container):
    """Process uploaded files with progress feedback. Returns total segments added."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    coll = client.get_or_create_collection(name=collection_name)

    total_segments = 0
    total_files = len(uploaded_files)

    for file_idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress_container.markdown(f"**Processing file {file_idx + 1} of {total_files}:** {uploaded_file.name}")

        content = ""
        if uploaded_file.name.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                import io
                pdf_bytes = uploaded_file.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                total_pages = len(reader.pages)

                page_texts = []
                for page_idx, page in enumerate(reader.pages):
                    progress_container.markdown(f"**Processing file {file_idx + 1} of {total_files}:** {uploaded_file.name}  \nPage {page_idx + 1} of {total_pages}")
                    page_texts.append(page.extract_text() or "")
                content = "\n".join(page_texts)
            except ImportError:
                progress_container.warning(f"Skipped {uploaded_file.name} - PDF support requires: pip install pypdf")
                continue
        else:
            content = uploaded_file.read().decode('utf-8', errors='ignore')

        if not content.strip():
            continue

        # Chunk content
        chunk_size, overlap = 1000, 200
        chunks = []
        start = 0
        while start < len(content):
            chunk = content[start:start + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap

        if chunks:
            display_name = get_display_name(collection_name)
            base_id = f"{display_name}_{uploaded_file.name}".replace(" ", "_")[:50]
            coll.add(
                documents=chunks,
                ids=[f"{base_id}_chunk_{i}" for i in range(len(chunks))],
                metadatas=[{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]
            )
            total_segments += len(chunks)

    return total_segments


def check_rag_available():
    """Check if RAG database is available"""
    try:
        import chromadb
        if os.path.exists('./chroma_db'):
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("unit_materials")
            return True
        return False
    except:
        return False
