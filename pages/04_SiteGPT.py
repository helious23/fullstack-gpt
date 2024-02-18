from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st


@st.cache_data(
    show_spinner="웹사이트를 읽고 있습니다. 이 작업은 최초 1회만 진행됩니다."
)
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 1
    loader.requests_kwargs = {"verify": False}
    docs = loader.load()
    return docs


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌏",
)

st.title("SiteGPT")

html2text_tramsformer = Html2TextTransformer()

st.markdown(
    """
    Ask questions about the content of a website.
    
    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    url = st.text_input(
        "URL 주소를 적어주세요",
        placeholder="https://example.com/sitemap.xml",
    )

# if url:
#     loader = AsyncChromiumLoader([url])
#     docs = loader.load()
#     html2text_tramsformer.transform_documents(docs)
#     st.markdown(docs)

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL 을 적어주세요")
    else:
        docs = load_website(url)
        st.write(docs)
