import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set your API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# Load multiple PDFs
def load_documents(pdf_paths):
    all_chunks = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
    print(f"✅ {len(all_chunks)} chunks created from {len(pdf_paths)} document(s)!")
    return all_chunks

# Create ChromaDB vector store
def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ ChromaDB vector store created!")
    return vectorstore

# Ask question
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful study assistant. Answer the question based on the study material below.
    Be clear, concise and educational in your response.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer

# Main
if __name__ == "__main__":
    print("📚 AI Study Assistant")
    print("---------------------")
    
    pdf_paths = []
    while True:
        path = input("Enter PDF path (or press Enter to continue): ")
        if path == "":
            break
        pdf_paths.append(path)
    
    if not pdf_paths:
        print("❌ No documents provided!")
        exit()
    
    chunks = load_documents(pdf_paths)
    vectorstore = create_vectorstore(chunks)
    
    print("\n🤖 Study Assistant Ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = ask_question(vectorstore, question)
        print(f"Assistant: {answer}\n")
