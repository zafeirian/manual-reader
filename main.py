from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Annotated
from pydantic import BaseModel
from create_vdb import generate_vdb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import shutil, tempfile, os
from pathlib import Path
from contextlib import asynccontextmanager


CHROMA_PATH='chroma'
EMBED_MODEL='text'
LLM_MODEL = 'gpt-4o-mini'

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    embedding_function = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=api_key,
    chunk_size=64,
    )

    upload_dir = tempfile.mkdtemp(prefix='uploads_')

    app.state.upload_dir = upload_dir
    app.state.embedding_function = embedding_function

    try:
        yield
    finally:
        app.state.embedding_function = None
        shutil.rmtree(upload_dir, ignore_errors=True)


app = FastAPI(title='Manual Reader', lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    tmp_paths: list[str] = []
    documents = generate_vdb(files, embedding_function, CHROMA_PATH)
    if len(documents) > 0:
        return {"VDB created": True}
    else:
        raise HTTPException(status_code=500, detail='VDB not created.')



