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

🔑 Rotate/Change your Google API key (Gemini)

You can provide the key in one of these ways (in priority order):

1) Streamlit secrets (recommended for deployment)
    - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
    - Put your key:
       GOOGLE_API_KEY = "your_new_google_api_key_here"

2) .env file (for local dev)
    - Ensure a `.env` file exists next to `app.py` with:
       GOOGLE_API_KEY=your_new_google_api_key_here

3) Windows PowerShell environment variable (current session only)
    - In the same shell where you run Streamlit:
       $env:GOOGLE_API_KEY = "your_new_google_api_key_here"

4) Persist environment variable for your user (requires new shell)
    - setx GOOGLE_API_KEY "your_new_google_api_key_here"
    - Close and reopen PowerShell for it to take effect.

Notes
- The app supports a fallback demo mode without the key (uses local sentence-transformers).
- For the full Google-powered experience, install the original packages:
   pip install langchain langchain-community langchain-google-genai faiss-cpu
   and keep the entries in `requirements.txt` up-to-date for your platform.
