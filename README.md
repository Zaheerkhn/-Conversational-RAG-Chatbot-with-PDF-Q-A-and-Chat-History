# 🧠 Conversational RAG Chatbot with PDF Q&A and Chat History  

This project is an advanced **Conversational Retrieval-Augmented Generation (RAG) chatbot** that enables users to upload PDFs and ask questions based on their content. It integrates **document-based Q&A with persistent chat history**, ensuring a seamless and context-aware conversation experience.  

🚀 **This application brings together the best aspects of my previous projects**, combining efficient document retrieval with interactive chat history to enhance user interactions.  

## 🌟 Features  

✔️ **Conversational Q&A** - Ask questions about uploaded PDFs and get intelligent answers.  
✔️ **Persistent Chat History** - Remembers previous messages for better contextual responses.  
✔️ **Multi-PDF Support** - Upload multiple PDFs at once for querying.  
✔️ **Efficient Vector Search** - Uses **FAISS** for fast and accurate document retrieval.  
✔️ **Secure API Handling** - Requires a **Groq API key** for execution.  
✔️ **Interactive UI** - Built with **Streamlit** for an easy-to-use interface.  

## 🛠️ Tech Stack  

- **Frontend**: Streamlit (for UI & interactivity)  
- **LLM**: Groq (Deepseek-R1-Distill-Llama-70b)  
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)  
- **Vector Store**: FAISS (for document retrieval)  
- **PDF Processing**: PyPDFLoader  
- **Environment Management**: dotenv  

## 📥 Installation  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/Zaheerkhn/-Conversational-RAG-Chatbot-with-PDF-Q-A-and-Chat-History.git
cd -Conversational-RAG-Chatbot-with-PDF-Q-A-and-Chat-History
```

### 2️⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys  

Create a `.env` file in the root directory and add your API keys:  
```sh
HF_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langchain_api_key
```

### 4️⃣ Run the Application  
```sh
streamlit run app.py
```

## 🎯 How to Use  

1. **Enter your Groq API key** in the input field.  
2. **Upload one or multiple PDFs** for analysis.  
3. **Ask questions** related to the document's content.  
4. **View chat history** for context-aware responses.  

## 📌 Example Interaction  

- **User**: _"What is the main topic of this document?"_  
- **Assistant**: _"The document discusses AI advancements in healthcare."_  

## 📜 License  

This project is open-source under the **Apache License 2.0**.  


---

🎯 _Contributions are welcome! Feel free to open issues and submit pull requests._ 🚀  

