import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import whoami
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ FIXED 

# ✅ Load environment variables
load_dotenv()

# ✅ Get Hugging Face token from .env
token = os.getenv("HF_TOKEN")

if not token:
    st.error("❌ Hugging Face API token not found. Check your .env file.")
    st.stop()

# ✅ Set token in environment (ensures authentication for embeddings)
os.environ["HF_TOKEN"] = token

# ✅ Streamlit UI
st.title("📄 RAG System with Teacher-Student LAB Architecture")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # ✅ Faster Student Model (for embedding-based retrieval)
    embedder = HuggingFaceEmbeddings()

    text_splitter = SemanticChunker(embedder)
    documents = text_splitter.split_documents(docs)

    # ✅ FAISS Vector Store for fast retrieval
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # ✅ Teacher Model (Larger LLM for fallback)
    teacher_llm = Ollama(model="deepseek-r1:1.5b")

    # ✅ Student Model Prompt
    student_prompt = """
    Use the following retrieved context to answer the question.
    Context: {context}
    Question: {question}
    Answer:
    """

    STUDENT_PROMPT = PromptTemplate.from_template(student_prompt)

    student_chain = LLMChain(llm=teacher_llm, prompt=STUDENT_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=student_chain, document_variable_name="context")

    # ✅ RAG Retrieval-Based QA Chain
    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    # ✅ User Input (Query)
    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        try:
            response = qa(user_input)
            if response and response["result"]:
                # ✅ Student Model (FAISS Context Retrieval)
                st.write("🟢 **Response from Student Model (PDF Context):**")
                st.write(response["result"])
            else:
                # ❌ No Relevant Context Found → Query the Teacher Model
                st.write("🔴 No relevant information found in PDF. Querying the **Teacher Model (DeepSeek-R1)**...")
                llm_response = teacher_llm.invoke(user_input)
                st.write("🟠 **Response from Teacher Model (LLM Fallback):**")
                st.write(llm_response)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
