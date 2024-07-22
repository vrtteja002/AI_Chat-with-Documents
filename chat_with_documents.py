"""Chat with retrieval and embeddings."""
import logging
import os
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import streamlit as st

from utils import MEMORY, load_document

logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()
api_key = st.secrets["OPENAI_API_KEY"]
# Setup LLM and QA chain; set temperature low to keep hallucinations in check
LLM = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0.5, streaming=True, openai_api_key=api_key
)

def configure_retriever(docs: list[Document]) -> BaseRetriever:
    """Retriever to use."""
    # Split each document documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # Create vectordb with single call to embedding model for texts:
    vectordb = Chroma.from_documents(splits, embeddings)
    return vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 10,
            "include_metadata": True
        },
    )

def configure_chain(retriever: BaseRetriever):
    """Configure chain with a retriever."""
    MEMORY.output_key = 'answer'
    return ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000,
    )

def configure_retrieval_chain(uploaded_files):
    """Read documents, configure retriever, and the chain."""
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
      for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs)
    chain = configure_chain(retriever=retriever)
    return chain

