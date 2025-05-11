import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config(page_title="📄 ChatPDF - RAG Assistant", page_icon="🧠", layout="wide")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user1": {"password": "user123", "role": "user"},
    "user2": {"password": "user456", "role": "user"},
}

def login():
    st.sidebar.header("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")

    if login_btn:
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['role'] = user["role"]
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")

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

def get_vector_store(text_chunks, username):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    os.makedirs(f"faiss_indexes/{username}", exist_ok=True)
    vector_store.save_local(f"faiss_indexes/{username}/index")

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

def user_input(user_question, username, role):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = []

    if role == "admin":
        for user_folder in os.listdir("faiss_indexes"):
            db_path = f"faiss_indexes/{user_folder}/index"
            if os.path.exists(db_path):
                db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                docs.extend(db.similarity_search(user_question))
    else:
        db_path = f"faiss_indexes/{username}/index"
        if os.path.exists(db_path):
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_question)
        else:
            st.warning("⚠️ No documents uploaded yet.")

    if docs:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    else:
        st.warning("❌ No relevant documents found for this question.")

def main():
    # Check if logged_in exists, if not call the login function.
    if "logged_in" not in st.session_state or not st.session_state['logged_in']:
        login()
        return

    username = st.session_state['username']
    role = st.session_state['role']

    # Once logged in, skip login and show the chat interface
    st.sidebar.success(f"✅ Logged in as: {username} ({role})")
    pdf_docs = st.file_uploader("📄 Upload PDF Files", type="pdf", accept_multiple_files=True)

    if st.button("🚀 Submit & Process"):
        if pdf_docs:
            with st.spinner("🔄 Extracting and indexing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, username)
                st.success("✅ Documents processed successfully!")
        else:
            st.warning("⚠️ Upload at least one PDF.")

    if st.button("🔒 Logout"):
        st.session_state.clear()
        st.rerun()

    # Main Content
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='color: #2E8B57;'>🧠 ChatPDF  AI Assistant</h1>
            <p style='font-size: 18px; color: gray;'>
                Ask intelligent questions across multiple PDFs using Retrieval-Augmented Generation (RAG)
            </p>
        </div>
        <hr>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 Ask a Question")
    st.markdown("*Type any question related to your uploaded PDFs below*")

    user_question = st.text_input("❓ Your Question")

    if user_question:
        with st.spinner("🤖 Generating answer..."):
            user_input(user_question, username, role)

if __name__ == "__main__":
    main()
