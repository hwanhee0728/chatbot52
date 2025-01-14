__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from PIL import Image

# .env 파일 로드
load_dotenv()
password_key = os.getenv('KEY')

# 페이지 설정
st.set_page_config(page_title="KMLA Chatbot", page_icon="🤖", layout="wide")

# CSS를 사용하여 여백 제거
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
        }
    </style>
    """, unsafe_allow_html=True)

st.write("KMLA Chatbot 팀")
st.write(":robot_face: KMLA Chatbot - 민사고에 대해 질문해주세요!")

# 인증 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 비밀번호 입력 필드
password = st.text_input(":name_badge: 패스워드를 넣어주세요!", type="password", key="password_input")

if password:
    if password == password_key:
        st.session_state.authenticated = True
        st.success("✅ 인증 성공! 질문을 입력하세요.")
    else:
        st.error("❌ 비밀번호가 올바르지 않습니다. 다시 시도해 주세요.")
        st.session_state.authenticated = False  # 인증 실패 시 상태 초기화

# 인증된 경우 질문 입력 필드 표시
if st.session_state.authenticated:
    # CSS를 적용한 질문 입력 필드
    user_input = st.text_input(
        ":eight_pointed_black_star:민사고에 대해 질문하고 엔터를 눌러주세요!",
        key="user_input",
        placeholder="질문을 입력하세요.",
    )

    if user_input:
        # 질문에 대한 응답 표시
        st.write(':pencil2::pencil2::pencil2: 답변 준비 중입니다 :pencil2::pencil2::pencil2:')

        # 임베딩 모델 및 Chroma DB 초기화
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        try:
            db = Chroma(persist_directory="chromadb_ada4.1", embedding_function=embeddings_model)
        except Exception as e:
            st.error(f"Error initializing database: {e}")
            raise

        # StreamHandler 클래스 정의
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text = initial_text

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                self.container.markdown(self.text)

        # 통합된 QA 체인 초기화
        try:
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)

            # 초기 가이드를 대화 시작에 포함
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name="gpt-4o",
                    temperature=0.3,
                    max_tokens=10000,
                    streaming=True,
                    callbacks=[stream_handler]
                ),
                retriever=db.as_retriever(search_kwargs={"k": 40}),
                return_source_documents=False
            )

            # 질문 처리 및 응답 표시
            initial_prompt = "당신은 한국어를 잘 이해하며, 항상 공손하고 친근하고 따뜻하고 즐거운 태도로 답변하고, 아주 상세하게 답하는 민족사관고등학교 챗봇입니다."
            qa_chain.invoke({"query": f"{initial_prompt}\n{user_input}"})
        except Exception as e:
            st.error(f"Error during QA chain execution: {e}")
            raise
