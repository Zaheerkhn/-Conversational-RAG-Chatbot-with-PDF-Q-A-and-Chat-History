import streamlit as st 
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A ChatBot With Chat History"

# Load Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")

# Sidebar for API Key and Session ID
st.sidebar.title("🔑 API & Session Management")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
session_id = st.sidebar.text_input("Session ID", value="Default_session")

st.sidebar.markdown("---")  # Divider for better UI

# Main Title
st.title("🤖 Conversational RAG with Chat History")
st.markdown("Upload PDFs and ask questions related to their content.")

# Initialize session state store if not exists
if "store" not in st.session_state:
    st.session_state.store = {}

# Upload PDF Section
st.subheader("📂 Upload PDF File")
pdf_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

# Validate API Key
if not api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar.")
    st.stop()

# Load Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Deepseek-R1-Distill-Llama-70b")

if pdf_files:
    documents = []
    with st.spinner("📖 Processing PDFs..."):
        for pdf_file in pdf_files:
            temp_pdf_path = f'./temp_{pdf_file.name}'
            with open(temp_pdf_path, "wb") as file:
                file.write(pdf_file.read())

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temp_pdf_path)  # Cleanup temp file

    st.success(f"✅ Successfully loaded {len(pdf_files)} PDF(s).")

    # Text Splitting and Vector Store Creation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

# Define contextualizing prompt
    contextualize_q_system_prompt =(
            "Given a chat history and the latest user question, "
             "reformulate the question to be standalone if necessary. "
             "Return the question as is if no context is needed."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Define QA system prompt
    qa_system_prompt = (
                "Answer the question based on the context below. "
                "Be concise. If unsure, say 'I don't know'.\n\n"
                "{context}"
         )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Session History Management
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # Add message history to chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Chat Section
    st.subheader("💬 Chat Interface")
    user_input = st.text_input("Ask a question:")

    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        # Display Chat Messages
        st.chat_message("user").markdown(f"**You:** {user_input}")
        st.chat_message("assistant").markdown(f"**Assistant:** {response['answer']}")

        # Display Chat History
        with st.expander("📜 Chat History"):
            for msg in session_history.messages:
                if msg.type == "human":
                    st.markdown(f"**You:** {msg.content}")
                else:
                    st.markdown(f"**Assistant:** {msg.content}")

else:
    st.info("📌 Upload a PDF to begin.")


