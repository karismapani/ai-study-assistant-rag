# AI Study Assistant using RAG

An AI-powered study assistant that answers questions from uploaded study materials using LangChain and RAG.

## 🚀 Features
- Upload any study material (PDF)
- Ask questions from the document
- Get accurate context-aware answers
- Supports multi-document retrieval
- Powered by LangChain, ChromaDB and Google Gemini

## 🛠️ Tech Stack
- Python
- LangChain
- ChromaDB (Vector Store)
- Google Gemini API
- PyPDF

## ⚙️ How It Works
1. Study material PDF is loaded and split into chunks
2. Chunks are stored in ChromaDB vector store
3. User asks a question
4. Relevant chunks are retrieved and passed to LLM
5. LLM generates accurate answer based on context

## 📦 Installation
pip install langchain-google-genai chromadb pypdf langchain-text-splitters langchain-core langchain-community

## 🔑 Setup
Add your Gemini API key:
export GOOGLE_API_KEY="your-api-key"

## 📌 Status
🚧 Under Development# ai-study-assistant-rag
AI-powered study assistant using LangChain and RAG
