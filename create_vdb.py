from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import shutil, os

def split_into_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

def creation_of_chroma(chunks: list[Document], embedding_function, persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    db = Chroma.from_documents(documents=chunks, 
                              embedding=embedding_function, 
                              persist_directory=persist_directory
                              )