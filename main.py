import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

import google.generativeai as genai

genai.configure(api_key="AIzaSyBplbFnv2FWfaJjqZyWZdnlbm3krd-W_ww")
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Test query")
print(response.text)

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Title
st.set_page_config(page_title="Agentic RAG", layout="centered")
st.title("📄 Agentic RAG QA Bot")
st.write("Ask questions based on the uploaded documents using Gemini-powered Agentic RAG.")

# Upload document
uploaded_file = st.file_uploader("📤 Upload a text file", type=["txt"])

if uploaded_file:
    # Load and split document
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.read())
    loader = TextLoader("temp.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embedding and vectorstore
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever()

    # LLM setup
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Store conversation
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    user_query = st.text_input("💬 Ask your question:")
    if user_query:
        relevant_docs = retriever.get_relevant_documents(user_query)
        response = chain.run(input_documents=relevant_docs, question=user_query)
        st.session_state.history.append((user_query, response))

    # Display chat history
    for q, a in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("ai"):
            st.markdown(a)
