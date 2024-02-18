from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌏",
)

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()


st.markdown(
    """
    Ask questions about the content of a website.
    
    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    url = st.text_input(
        "URL 주소를 적어주세요",
        placeholder="https://example.com",
    )


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)
