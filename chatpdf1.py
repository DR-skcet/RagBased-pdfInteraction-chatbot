import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

import streamlit as st

def main():
    st.set_page_config(page_title="📄 ChatPDF - RAG Assistant", page_icon="🧠", layout="wide")

    # --- Sidebar ---
    with st.sidebar:
        st.title("📂 Upload & Process")
        st.markdown("Upload one or more PDF documents. Click the button to process them for Q&A.")
        
        pdf_docs = st.file_uploader("📄 Upload PDF Files", type="pdf", accept_multiple_files=True)
        
        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Extracting and indexing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ All documents processed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    # --- Header ---
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: white;'>🧠 ChatPDF  AI Assistant</h1>
            <p style='font-size: 18px; color: gray;'>
                Ask intelligent questions across multiple PDFs using Retrieval-Augmented Generation (RAG)
            </p>
        </div>
        <hr>
    """, unsafe_allow_html=True)

    # --- Main Content ---
    st.markdown("### 💬 Ask a Question")
    st.markdown("*Type any question related to your uploaded PDFs below*")

    user_question = st.text_input("❓ Your Question")

    if user_question:
        with st.spinner("🤖 Generating answer..."):
            user_input(user_question)

if __name__ == "__main__":
    main()
