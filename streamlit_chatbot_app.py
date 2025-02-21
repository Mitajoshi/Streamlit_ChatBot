
from openai import OpenAI
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from PyPDF2 import PdfReader

st.title("ChatGPT-like Chatbot")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load sentence transformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Store chat history and FAISS index
if "messages" not in st.session_state:
    st.session_state.messages = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_texts" not in st.session_state:
    st.session_state.document_texts = []

# File Upload
uploaded_file = st.file_uploader("Upload a file (TXT, PDF, CSV)", type=["txt", "pdf", "csv"])

# Process Uploaded File
if uploaded_file:
    if uploaded_file.type == "text/plain":  # TXT File
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        text = df.to_string()

    # Split document into chunks
    chunk_size = 500  # Adjust based on document size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    st.session_state.document_texts = chunks

    # Create FAISS Index
    embeddings = embedder.encode(chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    st.session_state.faiss_index = faiss_index

    st.success(f"File processed with {len(chunks)} chunks added to FAISS!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve Relevant Chunks from FAISS
    if st.session_state.faiss_index is not None:
        query_embedding = embedder.encode([prompt])
        D, I = st.session_state.faiss_index.search(np.array(query_embedding, dtype=np.float32), k=3)
        context = " ".join([st.session_state.document_texts[i] for i in I[0] if i < len(st.session_state.document_texts)])
    else:
        context = "No relevant documents found."

    # Generate Response with Retrieved Context
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Use the following retrieved document content: {context}"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})















