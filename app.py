import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from typing import List

# --- CUSTOM CSS STYLING ---
def load_css():
    st.markdown("""
    <style>
    /* Modern CSS Reset and Base Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Custom Font and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Dark Theme Base */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
        color: #e0e0e0;
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #0099cc, #00d4ff);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
        background: linear-gradient(135deg, #00d4ff, #0099cc, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    .main-header p {
        color: #b0b0b0;
        font-size: 1.2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s ease-in-out infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .sidebar-header h3 {
        font-weight: 600;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .sidebar-header p {
        position: relative;
        z-index: 1;
        opacity: 0.9;
    }
    
    /* File Upload Styling */
    .stFileUploader {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed #00d4ff;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stFileUploader:hover {
        border-color: #0099cc;
        background: rgba(26, 26, 46, 0.9);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4);
    }
    
    /* Chat Container Styling */
    .chat-container {
        background: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 60vh;
        position: relative;
    }
    
    .chat-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    }
    
    /* Message Styling */
    .stChatMessage {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        transform: translateX(5px);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    .stChatMessage[data-testid="chatMessage"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 204, 0.1));
        border-left: 4px solid #00d4ff;
    }
    
    /* Chat Input Styling */
    .stChatInput {
        background: rgba(26, 26, 46, 0.95);
        border-radius: 15px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        padding: 1.2rem;
        margin-top: 1.5rem;
        backdrop-filter: blur(10px);
        color: #e0e0e0;
    }
    
    .stChatInput:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
        background: rgba(26, 26, 46, 0.98);
    }
    
    /* Status and Info Styling */
    .stStatus {
        background: rgba(26, 26, 46, 0.95);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 204, 0.1));
        border: 1px solid #00d4ff;
        border-radius: 12px;
        padding: 1.2rem;
        color: #00d4ff;
        backdrop-filter: blur(10px);
    }
    
    /* Warning and Error Styling */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
        border: 1px solid #ffc107;
        border-radius: 12px;
        padding: 1.2rem;
        color: #ffc107;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(229, 57, 53, 0.1));
        border: 1px solid #f44336;
        border-radius: 12px;
        padding: 1.2rem;
        color: #f44336;
        backdrop-filter: blur(10px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        font-weight: 600;
        color: #00d4ff;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner Styling */
    .stSpinner {
        color: #00d4ff;
    }
    
    /* Success Message Styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(67, 160, 71, 0.1));
        border: 1px solid #4caf50;
        border-radius: 12px;
        padding: 1.2rem;
        color: #4caf50;
        backdrop-filter: blur(10px);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 46, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0099cc, #00d4ff);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        
        .chat-container {
            padding: 1.5rem;
        }
        
        .sidebar-header {
            padding: 1.5rem;
        }
    }
    
    /* Additional Dark Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stTextInput > div > div > input {
        background: rgba(26, 26, 46, 0.95);
        color: #e0e0e0;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1);
    }
    
    .stSelectbox > div > div > div {
        background: rgba(26, 26, 46, 0.95);
        color: #e0e0e0;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .stMarkdown {
        color: #e0e0e0;
    }
    
    .stDivider {
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_pdf_text(pdf_docs: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits long text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks: List[str]):
    """Creates a FAISS vector store using Google embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not text_chunks:
        return None
    try:
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_conversation_chain(vectorstore):
    """Creates the conversational retrieval chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

# --- Main app ---

def main():
    load_dotenv()
    load_css()  # Load custom CSS
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🚨 Google API Key not found. Please set it in your .env file or Streamlit Secrets.")
        st.stop()

    st.set_page_config(
        page_title="Ask Your PDFs",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/ask-your-pdfs',
            'Report a bug': 'https://github.com/your-repo/ask-your-pdfs/issues',
            'About': '# Ask Your PDFs\n\nAn intelligent PDF chat application powered by AI.'
        }
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    # --- SIDEBAR FILE UPLOAD ---
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>📄 Document Hub</h3>
            <p>Upload & Process Your PDFs</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Choose your PDF files",
            accept_multiple_files=True,
            type="pdf",
            help="Select one or more PDF files to analyze"
        )

        if st.button("🚀 Process Documents", use_container_width=True):
            if pdf_docs:
                with st.status("🔄 Processing documents...", expanded=True) as status:
                    st.write("📖 **Step 1:** Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    status.update(label="✅ Text extracted successfully")

                    if raw_text:
                        st.write("✂️ **Step 2:** Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        status.update(label="✅ Text chunked successfully")

                        st.write("🧠 **Step 3:** Creating vector store...")
                        vectorstore = get_vectorstore(text_chunks)
                        status.update(label="✅ Vector store created successfully")

                        if vectorstore:
                            st.write("🔗 **Step 4:** Building conversation chain...")
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.processing_done = True
                            st.session_state.messages = []
                            status.update(label="🎉 Processing Complete!", state="complete", expanded=False)
                            st.success(f"✅ Successfully processed {len(pdf_docs)} document(s)!")
                        else:
                            status.update(label="❌ Error in vector store creation.", state="error")
                    else:
                        status.update(label="❌ No text could be extracted from the PDFs.", state="error")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    # --- MAIN CHAT INTERFACE ---
    st.markdown("""
    <div class="main-header">
        <h1>🚀Ask Your PDFs</h1>
        <p>Transform your documents into intelligent conversations. Upload PDFs and get instant answers to your questions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(
            placeholder="Ask a question about your documents... (e.g., What are the main topics discussed in the document?)",
            disabled=not st.session_state.processing_done
        ):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = {}
                    try:
                        response = st.session_state.conversation({'question': prompt})
                        answer = response.get('answer', "Sorry, I couldn't find an answer.")
                    except Exception as e:
                        st.error(f"⚠️ Internal Error: {e}")
                        answer = "Error during processing."

                    st.markdown(answer)

                    # Sources expander
                    with st.expander("📚 View Sources", expanded=False):
                        if 'source_documents' in response and response['source_documents']:
                            for i, doc in enumerate(response['source_documents'], 1):
                                st.markdown(f"**Source {i}:**")
                                st.info(f"{doc.page_content[:300]}...")
                                st.divider()
                        else:
                            st.write("No source documents found.")

            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
