from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st
from bs4 import BeautifulSoup
import html2text

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    #명령문
    당신은 웹사이트 분석기 입니다.
    [context]의 내용만을 기반으로 사용자의 [question]에 대답을 하여야 합니다.
    대답은 [context]에 나와 있는 내용만으로 이루어져야 하며 기존의 학습된 데이터가 포함되면 안됩니다.
    대답을 찾지 못할 경우 지어내지 말고 모르겠다고 대답하여야 합니다.
    
    #제약조건
    대답을 만든 경우 0점에서 5점 사이의 자연수로 대답에 대한 Score를 부여해야 합니다.
    Score는 사용자의 [question]과 관련이 클수록 높고, 관련이 없을수록 낮아야 합니다.
    사용자의 [question]과 전혀 상관이 없는 내용일 경우 Score는 0점입니다.
    모든 대답에 반드시 Score를 부여하세요.
    Context:{context}
    
    #예시
    Question: 달은 지구로부터 얼마나 멀리 떨어져 있나요?
    Answer: 달은 지구로부터 384,000km 떨어져 있습니다.
    Score: 5
    
    Question: 태양은 지구로부터 얼마나 떨어져 있나요?
    Answer: 모르겠습니다.
    Score: 0
    
    시작하세요!
    
    Question:{question}
    
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {
    #             "question": question,
    #             "context": doc.page_content,
    #         }
    #     )
    #     answers.append(result.content)

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            #명령문
            당신은 사용자의 [question]에 대답을 하는 서비스 입니다.
            제공되는 [answers] 만을 사용하여 대답을 하여야 합니다.
            
            #제약조건
            제공되는 [answers] 중 높은 점수를 가진 항목과 가장 최근 [Date] 항목을 사용하여 대답하세요.
            가장 최근의 항목보다 점수가 높은 항목을 우선 사용하여 대답하세요.
            제공되는 [answers] 의 점수가 같은 항목이 있다면, 최근의 항목만 사용하여 대답하세요.
            제공되는 [answers] 의 source 를 함께 보여주세요.
            
            시작하세요!
            Answers: {answers}
         
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']} \n Source:{answer['source']} \n Date:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup: BeautifulSoup):
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
    cache_dir = LocalFileStore(f"./.cache/site_embeddings/{url}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        restrict_to_same_domain=False,
        # filter_urls=[
        #     r"^(.*\/blog\/).*",
        # ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    loader.requests_kwargs = {"verify": False}
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌏",
)

st.title("SiteGPT")

html2text_tramsformer = Html2TextTransformer()


with st.sidebar:
    url = st.text_input(
        "URL 주소를 적어주세요",
        placeholder="https://example.com/sitemap.xml",
    )

# if url:
#     loader = AsyncChromiumLoader([url])
#     docs = loader.load()
#     st.write(docs)
#     html2text_tramsformer.transform_documents(docs)
#     text_maker = html2text.HTML2Text()
#     text = text_maker.handle(
#         docs[0].page_content,
#     )
#     st.write(text)


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL 을 적어주세요")
    else:
        retriever = load_website(url)
        query = st.text_input("웹 사이트에 대해 궁금한 점을 물어보세요")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))

else:
    st.markdown(
        """
        Ask questions about the content of a website.
        
        Start by writing the URL of the website on the sidebar.
        """
    )
