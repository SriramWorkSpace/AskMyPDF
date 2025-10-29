import os
from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None  # type: ignore


app = Flask(__name__)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


@app.route('/', methods=['POST'])
def ask():
    try:
        require_env("GOOGLE_API_KEY")
        pinecone_api_key = require_env("PINECONE_API_KEY")
        index_name = require_env("PINECONE_INDEX_NAME")

        payload = request.get_json(silent=True) or {}
        question = (payload.get('question') or '').strip()
        session_id = (payload.get('session_id') or '').strip()
        top_k = int(payload.get('top_k') or 5)

        if not question:
            return jsonify({"error": "Missing 'question' in JSON body."}), 400
        if not session_id:
            return jsonify({"error": "Missing 'session_id' in JSON body (from ingest step)."}), 400

        # Embed query
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        qvec = embeddings.embed_query(question)

        if Pinecone is None:
            return jsonify({"error": "pinecone-client not available on server."}), 500

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)

        results = index.query(vector=qvec, top_k=top_k, include_metadata=True, namespace=session_id)

        contexts = []
        sources = []
        for m in results.get('matches', []) or []:
            meta = m.get('metadata', {}) or {}
            text = meta.get('text', '')
            if text:
                contexts.append(text)
                sources.append({
                    "id": m.get('id'),
                    "score": m.get('score'),
                    "text": text[:300]
                })

        context_block = "\n\n".join(contexts[:top_k]) if contexts else ""

        prompt = (
            "You are a helpful assistant answering questions about provided documents.\n"
            "Use the context to answer concisely. If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        answer = llm.invoke(prompt).content

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})
