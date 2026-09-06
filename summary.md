# AskMyPDF — Project Summary

---

## 1. Overview

AskMyPDF is a Retrieval-Augmented Generation (RAG) application that turns static
PDF files into a conversational knowledge base. A user uploads one or more PDFs;
the application extracts the text, splits it into overlapping chunks, converts
each chunk into a numeric embedding vector, and stores those vectors in a vector
database. When a question is asked, the question is embedded with the same model,
the nearest chunks are retrieved by cosine similarity, and those chunks are
injected as context into a Google Gemini prompt. The model answers **only from
the retrieved context**, which is what prevents the hallucination problem of
asking an LLM about a document it has never seen.

The repository contains **two independent implementations of the same RAG
pipeline**, because the two runtimes have incompatible constraints:

| | Local / Streamlit app | Serverless / Vercel app |
|---|---|---|
| Entry point | `app.py` | `index.html` + `api/ingest.py` + `api/ask.py` |
| UI | Streamlit (server-rendered Python) | Static HTML + vanilla JS `fetch` |
| Web layer | Streamlit's own server | Flask WSGI apps on `@vercel/python` |
| Vector store | FAISS (in-process, in-memory) | Pinecone (managed, serverless) |
| Conversation memory | `ConversationBufferMemory` (per session) | None — each request is stateless |
| Chain | LangChain `ConversationalRetrievalChain` | Hand-rolled: embed → query → prompt |
| State handling | `st.session_state` holds chain + messages | `session_id` UUID used as a Pinecone namespace |
| Isolation | Process-local; store dies with the session | Namespace-per-session inside a shared index |
| Best for | Development, demos, local iteration | Public deployment, multi-user, zero-ops |

Both paths share the same four building blocks: **PyPDF2** for extraction,
**LangChain `CharacterTextSplitter`** (chunk 1000 / overlap 200) for chunking,
**Google `models/embedding-001`** (768-dimensional) for embeddings, and
**`gemini-1.5-flash-latest`** for generation.

**Configuration surface**

| Variable | Required by | Default | Purpose |
|---|---|---|---|
| `GOOGLE_API_KEY` | both paths | — | Gemini embeddings + chat |
| `PINECONE_API_KEY` | serverless only | — | Vector DB auth |
| `PINECONE_INDEX_NAME` | serverless only | — | Index to create/use |
| `PINECONE_CLOUD` | serverless only | `aws` | Serverless spec |
| `PINECONE_REGION` | serverless only | `us-east-1` | Serverless spec |

Both API handlers guard these through a `require_env()` helper that raises a
`RuntimeError` naming the missing variable, which the outer `try/except` converts
into a `500` with a readable JSON body.

---

## 2. Working — step by step

### 2.1 Local Streamlit path (`app.py`)

**Startup**
1. `load_dotenv()` reads `.env`; `load_css()` injects the dark theme via
   `st.markdown(..., unsafe_allow_html=True)`.
2. If `GOOGLE_API_KEY` is absent, the app renders an error and calls `st.stop()`.
3. Three keys are initialised in `st.session_state`: `conversation` (the chain),
   `messages` (chat transcript), `processing_done` (gates the chat input).

**Ingestion — triggered by "Process Documents"**
1. **Extract** — `get_pdf_text()` iterates the uploaded files and for each page
   appends `page.extract_text() or ""`. A per-file `try/except` means one corrupt
   PDF surfaces an error but does not abort the rest.
2. **Chunk** — `get_text_chunks()` uses `CharacterTextSplitter(separator="\n",
   chunk_size=1000, chunk_overlap=200)`. The 200-character overlap is what stops
   a sentence straddling a chunk boundary from becoming unretrievable.
3. **Embed + index** — `get_vectorstore()` calls `FAISS.from_texts()` with
   `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`. FAISS builds the
   index entirely in process memory.
4. **Chain** — `get_conversation_chain()` wires
   `ConversationalRetrievalChain.from_llm()` with `ChatGoogleGenerativeAI(
   model="gemini-1.5-flash-latest", temperature=0.7,
   convert_system_message_to_human=True)`, `vectorstore.as_retriever()`,
   a `ConversationBufferMemory(memory_key='chat_history', output_key='answer')`,
   and `return_source_documents=True`.
5. The chain is stored in `st.session_state.conversation`, `messages` is cleared,
   and `processing_done` flips to `True`, enabling the chat box.

**Question answering**
1. `st.chat_input` yields a prompt; it is appended to `messages` and echoed.
2. `st.session_state.conversation({'question': prompt})` runs the chain, which
   internally: (a) condenses the new question plus `chat_history` into a
   standalone question, (b) retrieves similar chunks from FAISS, (c) stuffs them
   into the prompt, (d) calls Gemini.
3. The `answer` is rendered; `source_documents` are shown (first 300 chars each)
   inside a "View Sources" expander; the answer is appended to `messages`.
4. Any exception is caught and displayed rather than crashing the app.

> **Why memory matters here:** because `ConversationBufferMemory` feeds prior
> turns back into the chain, a follow-up like *"and what about the second one?"*
> is resolved against the earlier turns before retrieval happens.

### 2.2 Serverless path (`index.html` + `api/*`)

**`POST /api/ingest`** — `multipart/form-data`, field `files`, optional `session_id`
1. Validate `GOOGLE_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.
2. `get_text_from_pdfs()` reads each `f.stream` with `PdfReader`. A failure is
   written *into* the text as `[Error reading <name>: <e>]` so the rest proceeds.
3. Empty text → `400`. Otherwise chunk with the same 1000/200 splitter.
4. `embeddings.embed_documents(chunks)` — one batched call for all chunks.
5. `session_id = form value or str(uuid.uuid4())`.
6. `ensure_pinecone_index()` lists existing indexes (via `_list_index_names()`,
   which defensively handles the several shapes the Pinecone client has returned
   across versions) and creates the index with `metric="cosine"` and
   `dimension=len(vectors[0])` if missing — so the dimension is never hard-coded
   wrong.
7. Records `{id: f"{session_id}-{i}", values: vec, metadata: {text: chunk[:1500]}}`
   are upserted in batches of 100 under `namespace=session_id`, keeping each
   request under Pinecone's payload limit.
8. Responds `{session_id, chunks, message}`.

**`POST /api/ask`** — JSON `{question, session_id, top_k?}`
1. Validate env and both required body fields (`400` if missing).
2. `embeddings.embed_query(question)` → 768-dim query vector.
3. `index.query(vector=qvec, top_k=top_k or 5, include_metadata=True,
   namespace=session_id)` — the namespace confines retrieval to *this* user's
   documents inside a shared index.
4. Matches are unwrapped with dict-or-attribute fallbacks (again, client-version
   tolerance); their `metadata.text` is joined with blank lines into a context block.
5. The prompt instructs the model to answer from context and to say it doesn't
   know otherwise. `llm.invoke(prompt)` is called; if `content` comes back as a
   list of parts, the parts are concatenated into a string.
6. Responds `{answer, sources: [{id, score, text[:300]}]}`.

**Browser orchestration (`index.html`)**
- Ingest click → `FormData` with every selected file → `fetch('/api/ingest')` →
  stores `data.session_id` in a module-scoped variable, prints the chunk count,
  and enables the Ask button.
- Ask click → appends the user message to the chat list → `fetch('/api/ask')`
  with the question and stored `session_id` → appends the assistant answer.
- Both handlers disable their button for the duration and re-enable it in a
  `finally`, rendering errors as messages rather than throwing.

> **Key structural difference:** the serverless path has **no conversation
> memory**. Every `/api/ask` call is independent, because a serverless function
> keeps no state between invocations.

### 2.3 Request/response contract

```
POST /api/ingest   (multipart/form-data)
  files: <pdf>[, <pdf>...]         session_id?: string
  -> 200 { "session_id": "uuid", "chunks": 42, "message": "Ingestion complete" }
  -> 400 { "error": "No files uploaded..." | "No text could be extracted..." }
  -> 500 { "error": "Missing environment variable: PINECONE_API_KEY" }

POST /api/ask      (application/json)
  { "question": string, "session_id": string, "top_k"?: number = 5 }
  -> 200 { "answer": string, "sources": [{ "id", "score", "text" }] }
  -> 400 { "error": "Missing 'question' in JSON body." }
  -> 500 { "error": "<message>" }

GET  /api/ingest | /api/ask               -> usage hint JSON
GET  /api/ingest/health | /api/ask/health -> { "status": "ok" }
```

---

## 3. Architecture

### 3.1 System architecture — both paths side by side

```
+------------------------------------------------------------------------------+
|                              A S K M Y P D F                                  |
+------------------------------------------------------------------------------+

  PATH A - LOCAL / STREAMLIT                 PATH B - SERVERLESS / VERCEL
  ==========================                 ============================

  +------------------------+                 +----------------------------+
  |  Browser :8501         |                 |  Browser                   |
  |  Streamlit-rendered UI |                 |  index.html (static)       |
  |  - sidebar uploader    |                 |  - <input type=file>       |
  |  - st.chat_input       |                 |  - fetch() calls           |
  |  - source expander     |                 |  - sessionId in JS var     |
  +-----------+------------+                 +------+--------------+------+
              | WebSocket                           | multipart    | JSON
              | (stateful)                          |              |
  +-----------v------------+            +-----------v---+  +-------v--------+
  |  app.py  (Streamlit)   |            | /api/ingest   |  |  /api/ask      |
  |                        |            | Flask @Vercel |  |  Flask @Vercel |
  |  st.session_state:     |            |  (stateless)  |  |  (stateless)   |
  |   - conversation chain |            +-------+-------+  +-------+--------+
  |   - messages[]         |                    |                  |
  |   - processing_done    |                    |                  |
  +-----------+------------+                    |                  |
              |                                 |                  |
  +-----------v------------+            +-------v------------------v-------+
  |  LangChain             |            |  Shared processing primitives    |
  |  ConversationalRetriev-|            |  PyPDF2 -> CharacterTextSplitter |
  |  alChain + Buffer      |            |  (chunk 1000 / overlap 200)      |
  |  Memory                |            +-------+------------------+-------+
  +-----------+------------+                    |                  |
              |                                 |                  |
  +-----------v------------+            +-------v-------+  +-------v--------+
  |  FAISS  (in-process)   |            | Pinecone      |  | Pinecone       |
  |  in-memory index       |            | upsert        |  | query top_k=5  |
  |  dies with the session |            | namespace=sid |  | namespace=sid  |
  +-----------+------------+            +-------+-------+  +-------+--------+
              |                                 |                  |
              +----------------+----------------+------------------+
                               |
                 +-------------v----------------------+
                 |   Google Generative AI  (Gemini)   |
                 |   - models/embedding-001  (768-d)  |
                 |   - gemini-1.5-flash-latest        |
                 +------------------------------------+
```

### 3.2 Ingestion pipeline (identical logic, different sink)

```
  PDF file(s)
      |
      v
+-----------------+   PdfReader(...).pages -> page.extract_text()
|  1. EXTRACT     |   per-file try/except: one bad PDF != total failure
|     PyPDF2      |
+--------+--------+
         |  raw_text : str
         v
+-----------------+   CharacterTextSplitter(
|  2. CHUNK       |       separator="\n",
|   LangChain     |       chunk_size=1000,
|                 |       chunk_overlap=200)   <- overlap preserves
+--------+--------+                               boundary-straddling sentences
         |  chunks : List[str]
         v
+-----------------+   GoogleGenerativeAIEmbeddings("models/embedding-001")
|  3. EMBED       |   .embed_documents(chunks)  ->  N x 768 floats
|    Gemini       |
+--------+--------+
         |  vectors : List[List[float]]
         v
+--------------------------------+--------------------------------+
|  4a. STORE - FAISS  (local)    |  4b. STORE - Pinecone (cloud)   |
|  FAISS.from_texts(...)         |  ensure_index(dim=len(v[0]),    |
|  held in st.session_state      |               metric=cosine)    |
|  lifetime = browser session    |  upsert(batch=100,              |
|                                |         namespace=session_id)   |
+--------------------------------+--------------------------------+
```

### 3.3 Query pipeline

```
   User question
        |
        +--------------- PATH A (Streamlit, has memory) ----------------+
        |                                                              |
        v                                                              |
  +------------------------------+                                     |
  | CONDENSE                     |  chat_history + new question        |
  | ConversationalRetrievalChain |  -> one standalone question         |
  +--------------+---------------+  (resolves "it", "the second one")  |
                 |                                                     |
        +--------+                                                     |
        |                                                              |
        +--------------- PATH B (serverless, stateless) ---------------+
        |                 question used verbatim                       |
        v                                                              |
  +------------------------------+                                     |
  | EMBED QUERY                  |  embed_query(q) -> 768-d vector     |
  +--------------+---------------+                                     |
                 v                                                     |
  +------------------------------+                                     |
  | RETRIEVE  (cosine similarity)|  FAISS .as_retriever()              |
  |                              |  -- or --                           |
  |                              |  Pinecone .query(top_k=5,           |
  |                              |            namespace=session_id)    |
  +--------------+---------------+                                     |
                 v                                                     |
  +------------------------------+                                     |
  | AUGMENT                      |  "Use the context to answer.        |
  | context block + question     |   If not in the context, say        |
  | -> grounded prompt           |   you don't know."                  |
  +--------------+---------------+                                     |
                 v                                                     |
  +------------------------------+                                     |
  | GENERATE                     |  gemini-1.5-flash-latest            |
  |                              |  temperature = 0.7                  |
  +--------------+---------------+                                     |
                 v                                                     |
  +------------------------------+                                     |
  | RESPOND                      |  answer + sources (300-char shown)  |
  +--------------+---------------+                                     |
                 |                                                     |
                 +-----> memory.save(question, answer) ----------------+
                         (Path A only)
```

### 3.4 Session isolation in the shared Pinecone index

```
                  Pinecone index  "askmypdf"   (cosine, 768-d)
   +----------------------------------------------------------------+
   |  namespace: 3f2a...-user-A     namespace: 9c17...-user-B        |
   |  +--------------------------+  +--------------------------+     |
   |  | 3f2a-0  vec  meta.text   |  | 9c17-0  vec  meta.text   |     |
   |  | 3f2a-1  vec  meta.text   |  | 9c17-1  vec  meta.text   |     |
   |  | 3f2a-2  ...              |  | 9c17-2  ...              |     |
   |  +--------------------------+  +--------------------------+     |
   |            ^                              ^                     |
   |            | query(namespace=3f2a...)     | query(ns=9c17...)   |
   +------------+------------------------------+---------------------+
                |                              |
        user A's /api/ask              user B's /api/ask
        -- cannot see B's vectors, and vice versa --
```

### 3.5 Vercel routing and build

```
   vercel.json
   |- builds:  api/*.py  ->  @vercel/python   (each file = one function)
   +- routes:
        /             ->  /index.html      (static, CDN-served)
        /api/ingest   ->  /api/ingest.py   (Flask WSGI app)
        /api/ask      ->  /api/ask.py      (Flask WSGI app)
```

---

## 4. Framework chosen, and why over the alternatives

### 4.1 The whole stack at a glance

| Layer | Chosen | Main alternatives | Deciding reason |
|---|---|---|---|
| Local UI | **Streamlit** | Gradio, React + FastAPI, Flask + Jinja | UI and RAG logic in one Python file; `st.session_state` is the entire state layer |
| Serverless API | **Flask on `@vercel/python`** | FastAPI, `BaseHTTPRequestHandler`, hosted Streamlit | Module-level WSGI `app` is auto-detected — no adapter code; handles multipart + JSON out of the box |
| RAG orchestration | **LangChain** | Raw Gemini SDK, LlamaIndex, Haystack | `ConversationalRetrievalChain` gives history-condensation, retrieval, stuffing and sources for free |
| Embeddings + LLM | **Google Gemini** | OpenAI GPT-4, Ollama/Llama, HuggingFace | One API key covers both roles; Flash is cheap, fast, large-context |
| Vector store (local) | **FAISS** | Chroma, Pinecone everywhere | In-process, zero setup, zero cost; the store is *meant* to die with the session |
| Vector store (cloud) | **Pinecone** | pgvector, self-hosted Qdrant/Weaviate | Managed + persistent + namespaced — multi-user isolation with no infrastructure |
| Frontend (cloud) | **Vanilla HTML/JS** | React, Vue, Svelte | Two `fetch` calls and a message list; a build step would add cost without benefit |
| PDF extraction | **PyPDF2** | pdfplumber, PyMuPDF, OCR (Tesseract) | Pure-Python, small wheel, no system deps — critical for serverless bundle size |

### 4.2 Streamlit — the local UI

| | |
|---|---|
| **Chosen because** | The entire interface (uploader, chat transcript, status stepper, source expander) is expressed in Python in the same file as the RAG logic. No API layer, no client bundle, no state-sync code. |
| **Key APIs used** | `st.session_state`, `st.file_uploader`, `st.chat_input`, `st.chat_message`, `st.status`, `st.expander` |
| **Trade-off accepted** | Re-runs the whole script on every interaction, so expensive objects **must** live in `st.session_state` or PDFs would be re-embedded constantly. Scales poorly to concurrent users — which is exactly why Path B exists. |

| Alternative | What it offers | Why it lost |
|---|---|---|
| **React/Next.js + FastAPI** | Full control over UI, production-grade | Two languages, two build systems, an HTTP contract and CORS to maintain — for a form and a message list. Correct for a product, over-engineered here. |
| **Gradio** | Even faster for a single input/output box | Closest competitor. Loses on multi-widget layout: the sidebar, `st.status` stepper and `st.expander` composition are what this UI is built from. |
| **Flask + Jinja templates** | No framework magic, fully explicit | You hand-write routes, templates, form handling, polling and session plumbing — everything Streamlit gives away. |

### 4.3 Flask on `@vercel/python` — the serverless API

| | |
|---|---|
| **Chosen because** | `@vercel/python` detects a module-level WSGI `app` and serves it directly, so each file is a normal Flask app locally *and* a serverless function in production, with zero adapter code. |
| **Key APIs used** | `request.files.getlist('files')` (multipart), `request.get_json(silent=True)` (JSON), `jsonify` |
| **Trade-off accepted** | No async, no schema validation — both handlers validate fields by hand and wrap the body in `try/except`. |

| Alternative | What it offers | Why it lost |
|---|---|---|
| **Deploying Streamlit itself** | One codebase for both targets | Streamlit needs a long-lived stateful WebSocket process; serverless is short-lived and stateless — architecturally incompatible. Would require a paid always-on container host. |
| **FastAPI** | Async, Pydantic validation, OpenAPI docs | Genuinely better for a larger API. For two endpoints with a handful of fields, the extra dependency weight buys nothing and inflates cold-start bundle size. |
| **Raw `BaseHTTPRequestHandler`** | Zero dependencies, smallest bundle | You parse multipart form data by hand. Flask earns its size on that alone. |

### 4.4 LangChain — the RAG orchestration

| | |
|---|---|
| **Chosen because** | Supplies exactly the pieces this project would otherwise hand-write: overlap-aware chunking, a uniform provider interface, and a conversational chain that implements history-condensation, retrieval, context stuffing and source return in one object. |
| **Key APIs used** | `CharacterTextSplitter`, `GoogleGenerativeAIEmbeddings`, `ChatGoogleGenerativeAI`, `ConversationalRetrievalChain`, `ConversationBufferMemory` |
| **Used asymmetrically** | Path A uses the **full chain abstraction** (it needs memory). Path B uses LangChain as a **thin client wrapper** only and builds the prompt by hand — keeping the cold-start dependency graph small and the behaviour explicit. |

| Alternative | What it offers | Why it lost |
|---|---|---|
| **Raw `google-generativeai` SDK** | Fewer dependencies, full transparency | Perfectly viable — `api/ask.py` is essentially this already. But re-implementing conversational history-condensation for Path A is real work for no gain. |
| **LlamaIndex** | Stronger indexing/retrieval abstractions | Wins on exotic index types; this app needs conversational memory far more than it needs those. |
| **Haystack** | Enterprise pipeline DAGs, production search | Heavier and aimed at larger deployments. Overkill for a single retriever and one LLM. |

### 4.5 Google Gemini — embeddings and generation

| | |
|---|---|
| **Chosen because** | `gemini-1.5-flash-latest` is optimised for this exact workload: high-throughput, low-latency, cheap, with a context window that comfortably holds five retrieved 1000-character chunks. |
| **Models used** | `models/embedding-001` (768-dim) for embeddings; `gemini-1.5-flash-latest` (temperature 0.7) for generation |
| **Operational win** | A single `GOOGLE_API_KEY` covers both roles — one credential, one vendor, one bill. |

| Alternative | What it offers | Why it lost |
|---|---|---|
| **OpenAI GPT-4 + `text-embedding-3`** | Excellent quality, mature ecosystem | Materially higher cost per token and no free tier — real friction for a portfolio/demo project. |
| **Local models (Ollama, Llama)** | Zero API cost, full data privacy | Needs GPU-class hardware and **cannot run in a serverless function at all**. |
| **HuggingFace Inference API** | Free tier, model variety | The stale `HUGGINGFACEHUB_API_TOKEN` in `.env` suggests this was explored. Rejected to consolidate on one provider; free endpoints also suffer cold starts and rate limits. |

### 4.6 FAISS locally, Pinecone in production — the two vector stores

This is the sharpest design decision in the project, and the split is deliberate.

| | FAISS (Path A) | Pinecone (Path B) |
|---|---|---|
| **What it is** | In-process C++/Python similarity-search library | Managed serverless vector database |
| **Setup cost** | `FAISS.from_texts()` — index exists in RAM | API key + lazily created index |
| **Persistence** | None; dies with the process | Durable across invocations |
| **Concurrency** | One process only | Shared by every function instance |
| **Multi-user isolation** | N/A (single user) | `namespace=session_id` — a logical slice per user in one physical index |
| **Cost** | Free | Managed-service pricing |
| **Bundle impact** | `faiss-cpu` is a large binary wheel | Thin HTTP client |
| **Why not the other one** | Cannot survive a serverless invocation, cannot be shared, and bloats the deploy bundle — hence excluded from `requirements.txt` | Requires network access and consumes quota on every local experiment |

| Alternative | What it offers | Why it lost |
|---|---|---|
| **Chroma / Qdrant / Weaviate (self-hosted)** | Open source, no vendor lock-in | Requires a server you operate and pay for around the clock — defeats the purpose of a serverless deployment. |
| **pgvector on Postgres** | Vectors and relational data in one store | Strong choice *if* the app already had a database. It doesn't — adding Postgres purely for vectors is a heavier operational commitment than a managed service. |
| **Pinecone everywhere (local too)** | One code path instead of two | Unifies the code but makes local development require network access and burn quota on every experiment. |
