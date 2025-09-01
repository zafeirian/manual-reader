from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Annotated
from pydantic import BaseModel
from create_vdb import split_into_chunks, creation_of_chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import shutil, tempfile, os, re
from pathlib import Path
from contextlib import asynccontextmanager
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader


CHROMA_PATH='chroma'
EMBED_MODEL='text-embedding-3-small'
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
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)


app = FastAPI(title='Manual Reader', lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    embedding_function: OpenAIEmbeddings = app.state.embedding_function

    all_docs =[]
    for f in files:
        ext = Path(f.filename).suffix
        safe_name = f"{uuid4().hex}{ext}"
        dest_path = os.path.join(app.state.upload_dir, safe_name)

        with open(dest_path, "wb") as out:
            shutil.copyfileobj(f.file, out)

        loader=PyPDFLoader(dest_path)
        docs = loader.load()
        for doc in docs:
            doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        all_docs.extend(docs)

    chunks = split_into_chunks(all_docs)
    creation_of_chroma(chunks = chunks, embedding_function=embedding_function, persist_directory=CHROMA_PATH)

    return {"Vector Database created": True}
    


