from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(
    show_spinner="웹사이트를 읽고 있습니다. 이 작업은 최초 1회만 진행됩니다."
)
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        restrict_to_same_domain=False,
        filter_urls=[
            r"^(.*\/blog\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
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
