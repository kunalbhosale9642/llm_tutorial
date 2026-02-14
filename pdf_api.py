from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os

# LangChain & RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
# This saves it one folder higher than your web files
UPLOAD_DIR = "../uploaded_files_external"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 1. Initialize Models Globally (Prevents reloading for every request)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="gemma3")

@app.post("/upload-pdf")
async def process_rag_request(
    file: UploadFile = File(...), 
    question: str = Form(...)
):
    # Validate PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 2. Save file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. RAG Processing
        # Load and Split
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Create Vectorstore (Ephemeral for this session)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # 4. Chain Construction
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 5. Invoke and Get Answer
        answer = rag_chain.invoke(question)

        # Clean up vectorstore and file
        vectorstore.delete_collection()
        os.remove(file_path)

        return {
            "answer": answer,
            "filename": file.filename,
            "status": "success"
        }

    except Exception as e:
        # Cleanup file if something crashes
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)