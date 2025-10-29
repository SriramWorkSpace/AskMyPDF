import os
import uuid
from typing import List
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    # Allow import-time errors to surface at runtime with a clear message
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore


app = Flask(__name__)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def get_text_from_pdfs(files: List) -> str:
    text = ""
    for f in files:
        try:
            reader = PdfReader(f.stream)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            # Continue other files; report which one failed
            filename = getattr(f, 'filename', 'uploaded.pdf')
            text += f"\n[Error reading {filename}: {e}]\n"
    return text


def chunk_text(text: str) -> List[str]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def _list_index_names(pc) -> set:
    names = set()
    try:
        res = pc.list_indexes()
        # Try names() helper first
        maybe_names = getattr(res, 'names', None)
        if callable(maybe_names):
            try:
                for n in maybe_names() or []:
                    names.add(n)
            except Exception:
                pass
        # Fallbacks
        items = getattr(res, 'indexes', None) or res
        if isinstance(items, (list, tuple)):
            for it in items:
                if isinstance(it, str):
                    names.add(it)
                elif isinstance(it, dict):
                    n = it.get('name')
                    if n:
                        names.add(n)
                else:
                    n = getattr(it, 'name', None)
                    if n:
                        names.add(n)
    except Exception:
        # As a last resort, attempt direct list
        try:
            for n in pc.list_indexes() or []:
                if isinstance(n, str):
                    names.add(n)
        except Exception:
            pass
    return names


def ensure_pinecone_index(pc, index_name: str, dim: int = 768) -> None:
    existing = _list_index_names(pc)
    if index_name not in existing:
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


@app.route('/', methods=['GET', 'POST'])
def ingest():
    try:
        if request.method == 'GET':
            return jsonify({
                "message": "Use POST with multipart/form-data. Field name: 'files' (one or more PDFs). Optional: 'session_id'",
                "health": "/health"
            })
        # Validate environment
        require_env("GOOGLE_API_KEY")
        pinecone_api_key = require_env("PINECONE_API_KEY")
        index_name = require_env("PINECONE_INDEX_NAME")

        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files uploaded. Use 'files' form field with one or more PDFs."}), 400

        # Extract text from PDFs
        raw_text = get_text_from_pdfs(files)
        if not raw_text.strip():
            return jsonify({"error": "No text could be extracted from the uploaded PDFs."}), 400

        # Chunking
        chunks = chunk_text(raw_text)
        if not chunks:
            return jsonify({"error": "No chunks produced from text."}), 400

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectors = embeddings.embed_documents(chunks)

        # Vector DB upsert (namespace per session)
        session_id = request.form.get('session_id') or str(uuid.uuid4())

        if Pinecone is None:
            return jsonify({"error": "pinecone-client not available on server."}), 500

        pc = Pinecone(api_key=pinecone_api_key)
        ensure_pinecone_index(pc, index_name, dim=len(vectors[0]))
        index = pc.Index(index_name)

        # Prepare records in small batches
        upserts = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            upserts.append({
                "id": f"{session_id}-{i}",
                "values": vec,
                "metadata": {
                    "text": chunk[:1500],  # trim metadata size
                }
            })

        # Batch upload to avoid payload size limits
        batch_size = 100
        for start in range(0, len(upserts), batch_size):
            index.upsert(vectors=upserts[start:start + batch_size], namespace=session_id)

        return jsonify({
            "session_id": session_id,
            "chunks": len(chunks),
            "message": "Ingestion complete"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health check for quick verification
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})
