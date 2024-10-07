import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import PyPDF2  # For PDF reading
import sentencepiece  # Ensure sentencepiece is imported


# Semantic segmentation model (BigBird)
tokenizer_segmentation = AutoTokenizer.from_pretrained('google/bigbird-roberta-base')
model_segmentation = AutoModel.from_pretrained('google/bigbird-roberta-base')

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Language model for generating answers
tokenizer_llm = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
model_llm = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

# Streamlit UI components
st.title("IntelliSearch: Contextual Retrieval & RAG System")

uploaded_file = st.file_uploader("Upload a Document", type=["txt", "pdf"])
question = st.text_input("Ask a question about the document:")

# Initialize session state for index and chunks
if 'index' not in st.session_state:
    st.session_state['index'] = None
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = None


# Function to read uploaded document
def read_document(file):
    if file.type == "text/plain":
        return file.read().decode('utf-8')
    elif file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        st.error("Unsupported file type.")
        return None

# Semantic Segmentation (Chunking)
def semantic_segmentation(text, max_length=512):
    # Split text into sentences
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(tokenizer_segmentation.tokenize(current_chunk + sentence)) < max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Embedding Chunks
def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Build FAISS Index
def build_faiss_index(embeddings):
    embeddings_np = embeddings.cpu().detach().numpy().astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index

# Retrieve Relevant Chunks
def retrieve_relevant_chunks(question, index, chunks, top_k=5):
    question_embedding = embedding_model.encode([question], convert_to_tensor=True)
    question_embedding_np = question_embedding.cpu().detach().numpy().astype('float32')
    distances, indices = index.search(question_embedding_np, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Generate Answer Using LLM
def generate_answer(question, context):
    # Optimized prompt engineering
    prompt = f"""
You are an expert assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    inputs = tokenizer_llm.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model_llm.generate(inputs, max_length=150, early_stopping=True)
    answer = tokenizer_llm.decode(summary_ids[0], skip_special_tokens=True)
    return answer


# Process Uploaded Document
if uploaded_file is not None:
    with st.spinner('Processing document...'):
        document_text = read_document(uploaded_file)
        if document_text:
            chunks = semantic_segmentation(document_text)
            embeddings = embed_chunks(chunks)
            index = build_faiss_index(embeddings)
            st.session_state['index'] = index
            st.session_state['chunks'] = chunks
            st.success("Document processed and indexed successfully.")


# Handle User Question
if question and st.session_state['index'] is not None:
    with st.spinner('Generating answer...'):
        relevant_chunks = retrieve_relevant_chunks(question, st.session_state['index'], st.session_state['chunks'])
        context = ' '.join(relevant_chunks)
        answer = generate_answer(question, context)
        st.write("### Answer:")
        st.write(answer)