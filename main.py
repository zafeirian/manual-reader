from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Annotated, Optional, List
from pydantic import BaseModel
from create_vdb import split_into_chunks, creation_of_chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import shutil, tempfile, os, re
from pathlib import Path
from contextlib import asynccontextmanager
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate


CHROMA_PATH='chroma'
EMBED_MODEL='text-embedding-3-small'
LLM_MODEL = 'gpt-4o-mini'
PROMPT_TEMPLATE='''
Answer the question based ONLY on the following context:

{context}

---

Answer the question based on the above context: {query}
'''

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

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0
    )

    upload_dir = tempfile.mkdtemp(prefix='uploads_')

    app.state.upload_dir = upload_dir
    app.state.embedding_function = embedding_function
    app.state.llm = llm

    try:
        yield
    finally:
        app.state.embedding_function = None
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        app.state.llm = None

class QueryRequest(BaseModel):
    text: str
    k: int = 3

class Chunk(BaseModel):
    content: str
    score: Optional[float] = None
    source: Optional[str] = None
    page: Optional[int] = None

class QueryResponse(BaseModel):
    response: str
    sources: list[dict]



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
    

@app.post("/ask")
async def ask(req: QueryRequest):
    llm: ChatOpenAI = app.state.llm
    embedding_function: OpenAIEmbeddings = app.state.embedding_function

    db = Chroma(embedding_function=embedding_function, 
                persist_directory=CHROMA_PATH)
    
    try:
        results = db.similarity_search_with_relevance_scores(req.text, k=req.k)
        if len(results)==0 or results[0][1]<0.3:
            return {"Error": "Unable to find relevant content."}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    
    chunks = [Chunk(content=doc.page_content, score=_score, source=doc.metadata['source'], page=doc.metadata['page']) for doc, _score in results]

    context_text = "\n\n---\n\n".join([chunk.content for chunk in chunks])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=req.text)

    response_text = await llm.ainvoke(prompt)
    sources = [{"source": chunk.source, "page": chunk.page} for chunk in chunks]

    return QueryResponse(response=response_text.content, sources=sources)
