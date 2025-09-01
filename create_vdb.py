from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import shutil, os

def load_documents(files):
    all_docs = []
    for file in files:
        loader=PyPDFLoader(file)
        documents=loader.load()
        for doc in documents:
            doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        all_docs.extend(documents)
    return all_docs

def split_into_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def vdb(chunks: list[Document], embedding_function, persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    db = Chroma.from_documents(documents=chunks, 
                              embedding_function=embedding_function, 
                              persist_directory=persist_directory
                              )
    print(f"Saved {len(chunks)} chunks into a Vector DB.")

def generate_vdb(documents: list[Document], embedding_function, persist_directory):
    docs = load_documents(documents)
    chunks = split_into_chunks(docs)
    vdb(chunks, embedding_function, persist_directory)

