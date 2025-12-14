import os
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

from langchain_openai import ChatOpenAI  # ✅ IMPORTANT

# --------------------------------------------------
# Ollama Config
# --------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"

llm = ChatOpenAI(
    model=MODEL,
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"   # dummy key, required by interface
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
file_path = r"C:\Users\adity\OllamaRAG\data\swiggy_text_data.txt"
with open(file_path, "r", encoding="utf-8") as f:
    df = f.read()
#df=pd.read_json(file_path)

# Convert rows → LangChain Documents


documents = [
    Document(
        page_content=df,
        metadata={"source": r"C:\Users\adity\OllamaRAG\data\swiggy_text_data.txt"}
    )
]

print("✅ Rows converted to documents")

# --------------------------------------------------
# Chunking
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

embeddings = FastEmbedEmbeddings(
    model_name="thenlper/gte-large"
)

print(f"✅ Chunks created: {len(chunks)}")

# --------------------------------------------------
# Vector Store
# --------------------------------------------------
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="swiggy_chroma_db"
)

retriever = vectordb.as_retriever(search_kwargs={"k": len(chunks)})
print("✅ Vector DB & Retriever ready")

# --------------------------------------------------
# Prompt
# --------------------------------------------------
RAG_TEMPLATE = ChatPromptTemplate.from_template("""
You are a data analyst working on Swiggy order data.

Answer ONLY using the context below.
If the answer is not present, say "Data not available".

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------
# RAG Chain
# --------------------------------------------------
rag_chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
        "question": itemgetter("question")
    }
    | RAG_TEMPLATE
    | llm
    | StrOutputParser()
)

print("\n Swiggy Ollama RAG System Ready!")

# --------------------------------------------------
# Chat Loop
# --------------------------------------------------
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    answer = rag_chain.invoke({"question": query})
    print("\nAnswer:", answer)
