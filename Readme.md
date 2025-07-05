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
