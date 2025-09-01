from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Annotated
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
import re

from dotenv import load_dotenv
app = FastAPI(title='Manual Reader')

load_dotenv()
CHROMA_PATH='chroma'
EMBED_MODEL='text'
LLM_MODEL = 'gpt-4o-mini'

def load_documents(files):
    all_docs = []
    for file in files:
        loader=PyPDFLoader(file)
        documents=loader.load()
        for doc in documents:
            doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        all_docs.extend(documents)
    return all_docs


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    all_docs = []
    for pdf_file in files:
        loader = PyPDFLoader(pdf_file)
        documents =loader.load()
        for doc in documents:
            doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        all_docs.extend(documents)


