📄 Ask Your PDF

An AI-driven Streamlit web app that lets you upload PDF files and interact with them using natural language. It harnesses the power of Google Gemini models and FAISS vector stores to interpret and respond to questions based on your PDF content.

---

🚀 Key Features

- Support for multiple PDF uploads
- Natural language chat powered by Gemini 1.5 Flash
- Intelligent responses via semantic search
- Remembers previous queries with chat history
- High-speed vector retrieval using FAISS
- Built using Streamlit and LangChain

Note: For Vercel deployment, this repo also includes a lightweight static frontend (`index.html`) and Python serverless APIs (`api/ingest.py`, `api/ask.py`) that use Pinecone as a managed vector database instead of FAISS.

---

🧠 How It Works

1. Upload PDFs from the sidebar

2. Behind the scenes:
   - Text is extracted using PyPDF2
   - Content is chunked into smaller pieces
   - Chunks are converted into embeddings using GoogleGenerativeAIEmbeddings
   - Embeddings are stored in a FAISS vector database
   - A ConversationalRetrievalChain is created with ChatGoogleGenerativeAI for interactive Q&A

---

📦 Setup Instructions

Install dependencies:
pip install -r requirements.txt

Add your API key to a .env file:
GOOGLE_API_KEY=your_google_api_key_here

Run the app:
streamlit run app.py

---

🚀 Deploy to Vercel (Serverless APIs + Static UI)

What you'll get:
- A static web UI at `/` (index.html)
- Python serverless APIs under `/api/ingest` and `/api/ask`
- Managed vector storage in Pinecone (namespace per session)

1) Set required environment variables in Vercel Project Settings → Environment Variables:

- GOOGLE_API_KEY = your Google Generative AI (Gemini) API key
- PINECONE_API_KEY = your Pinecone API key
- PINECONE_INDEX_NAME = askmypdf (or any name you prefer)
- PINECONE_CLOUD = aws (default; optional)
- PINECONE_REGION = us-east-1 (default; optional)

2) Deploy using Vercel CLI (PowerShell on Windows):

```powershell
# Install vercel CLI (if needed)
npm install -g vercel

# From the project root (this folder)
vercel login
vercel init  # optional: or link to an existing project with `vercel link`
vercel       # first deploy (preview)
vercel --prod
```

3) Use the app:
- Open the deployed URL
- Step 1: Upload PDFs and click "Process Documents" (ingests to Pinecone)
- Step 2: Ask questions (retrieval + Gemini answer)

Notes and limits:
- Serverless functions are stateless; ingestion uses a generated session ID to isolate your vectors in Pinecone (namespace).
- Timeouts may occur for very large PDFs; prefer smaller files or split uploads.
- The original Streamlit app (`app.py`) remains for local usage; Vercel serves the static UI + APIs path.
