from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI

import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🗒️",
)


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    retriever = vector_store.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DO NOT make anything up.
            
            ------
            Context: {context}
     
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask question to an AI about your files!        
    
    Upload your files on the sidebar.
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf .md or .docx file",
        type=["txt", "pdf", ".md", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("준비가 다 되었습니다. 궁금하신 점을 질문하세요 :)", "ai", save=False)
    paint_history()
    message = st.chat_input("첨부하신 파일에 대해 궁금하신 점을 입력하세요!")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state["messages"] = []
