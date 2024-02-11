import json
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema.runnable import RunnableLambda
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        StdOutCallbackHandler(),
    ],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
             #명령문
             당신은 퀴즈 선생님입니다. 
             [context]를 기반으로 하여 반드시 10개의 질문을 만들어야 합니다.
             내용이 적더라도 10개의 질문을 만드는 것이 목표입니다.
             만들어야 하는 질문은 [context]에 관련되어야만 하며 기존의 학습된 데이터가 포함되면 안됩니다. 
             질문의 내용은 사용자에게 [context]의 지식을 확인하기 위함입니다.
             
             #제약조건
             각각의 질문마다 4가지 보기를 제공합니다. 그 중 세 가지는 [context]와 일치하지 않아야 합니다. 나머지 하나는 [context]와 일치하는 사실이어야 합니다.
             [context]와 일치하는 사실은 보기 뒤에 (o) 표시를 하여야 합니다.
             
             #예시
             질문: 바다의 색깔은 무엇인가요?
             보기: 빨강 | 노랑 | 초록 | 파랑(o)
             
             질문: Goergia 의 수도는 어디인가요?
             보기: Baku | Tbilisi(o) | Manila | Beirut
             
             질문: 영화 Avatar 의 출시연도는 언제인가요?
             보기: 2007년 | 2001년 | 2009년(o) | 1998년
             
             질문: Julius Ceasar 는 누구일까요?
             보기: 로마의 황제(o) | 화가 | 배우 | 모델
             
             시작하세요!
             
             [context]:{context}
             """,
        ),
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                #명령문
                당신은 강력한 formatting 알고리즘 입니다.
                당신은 #예시질문 을 JSON format 으로 변환합니다.
                Answers with (o) are the correct ones.
                (o) 표시가 되어 있는 것이 정답입니다.
                
                #예시질문
                질문: 바다의 색깔은 무엇인가요?
                보기: 빨강 | 노랑 | 초록 | 파랑(o)
                
                질문: Goergia 의 수도는 어디인가요?
                보기: Baku | Tbilisi(o) | Manila | Beirut
                
                질문: 영화 Avatar 의 출시연도는 언제인가요?
                보기: 2007년 | 2001년 | 2009년(o) | 1998년

                질문: Julius Ceasar 는 누구일까요?
                보기: 로마의 황제(o) | 화가 | 배우 | 모델
                
                #예시Output
                ```json
                {{ "questions": [
                        {{
                            "question": "바다의 색깔은 무엇인가요?",
                            "answers": [
                                    {{
                                        "answer": "빨강",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "노랑",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "초록",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "파랑",
                                        "correct": true
                                    }},
                            ]
                        }},
                                    {{
                            "question": "Goergia 의 수도는 어디인가요?",
                            "answers": [
                                    {{
                                        "answer": "Baku",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Tbilisi",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Manila",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Beirut",
                                        "correct": false
                                    }},
                            ]
                        }},
                                    {{
                            "question": "영화 Avatar 의 출시연도는 언제인가요?",
                            "answers": [
                                    {{
                                        "answer": "2007년",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2001년",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2009년",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "1998년",
                                        "correct": false
                                    }},
                            ]
                        }},
                        {{
                            "question": "Julius Ceasar 는 누구일까요?",
                            "answers": [
                                    {{
                                        "answer": "로마의 황제",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "화가",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "배우",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "모델",
                                        "correct": false
                                    }},
                            ]
                        }}
                    ]
                }}
                ```
                시작하세요!

                질문: {context}

            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .txt .pdf .md or .docx file",
            type=["txt", "pdf", "md", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(
                top_k_results=3,
                lang="ko",
            )
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT
    
    I will make a quiz from Wikipedia articles or files you upload to test your knoledge and help you study.
    
    Get started by uploading a file or searching on Wikipedia in the sidebar.    
    """
    )
else:

    start = st.button("퀴즈 만들기")

    if start:
        chain = {"context": questions_chain} | formatting_chain | output_parser
        response = chain.invoke(docs)
        st.write(response)
