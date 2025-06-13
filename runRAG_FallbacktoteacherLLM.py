import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA


import os
from dotenv import load_dotenv
from huggingface_hub import whoami

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Get token from environment
token = os.getenv("HF_TOKEN")




st.title("📄 RAG System with DeepSeek R1 & Ollama (PDF + LLM Fallback)")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-r1:1.5b")

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        try:
            response = qa(user_input)
            if response and response["result"]:
                st.write("**Response from PDF:**")
                st.write(response["result"])
            else:
                # Fallback to direct LLM query if no context found
                st.write("No relevant information found in PDF. Querying the LLM directly...")
                llm_response = llm.invoke(user_input)
                st.write("**Response from LLM:**")
                st.write(llm_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")