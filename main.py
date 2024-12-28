__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

# .env 파일 로드
#load_dotenv()
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

st.write("융합프로젝트 2024년 1학기 KMLA Chatbot 팀")
st.write(":robot_face: KMLA Chatbot - 민사고에 대해 질문해주세요!")

# 인증 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 비밀번호 입력 필드
password = st.text_input(":name_badge: 패스워드를 넣어주세요!", type="password")

if password:
    if password == password_key:
        st.session_state.authenticated = True
        st.success("✅ 인증 성공! 질문을 입력하세요.")
    else:
        st.error("❌ 비밀번호가 올바르지 않습니다. 다시 시도해 주세요.")
        st.session_state.authenticated = False  # 인증 실패 시 상태 초기화

# 인증된 경우 질문 입력 필드 표시
if st.session_state.authenticated:
    user_input = st.text_input(":eight_pointed_black_star:민사고에 대해 질문하고 엔터를 눌러주세요!")

    if user_input:
        # 질문에 대한 응답 표시
        st.write(':pencil2::pencil2::pencil2: 답변 준비 중입니다 :pencil2::pencil2::pencil2:')

        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        try:
            db = Chroma(persist_directory="chromadb_ada2.8", embedding_function=embeddings_model)
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

        # Chunk 추출 및 로그 출력
        retriever = db.as_retriever(search_kwargs={"k": 20})  # 상위 n개의 관련 문서 검색

        try:
            # 질문에 대해 검색된 chunk를 가져옴
            relevant_docs = retriever.get_relevant_documents(user_input)
            
            #chunk_count = len(relevant_docs)  # 검색된 chunk 수 계산
            #st.write(f":mag: {chunk_count}개의 관련 문서를 검색했습니다. 이 내용을 바탕으로 답변을 생성합니다.")  # 웹 화면에 알림
            #for idx, doc in enumerate(relevant_docs):
            #    print(f"Chunk {idx + 1}: {doc.page_content}")  # VSCode 터미널에만 출력
            #    print("================================================================================")  # 구분선 추가
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            raise

        # LLM 및 QA 체인 구성
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3, max_tokens=3000, streaming=True, callbacks=[stream_handler])
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

        try:
            qa_chain.invoke({"query": user_input})
        except Exception as e:
            st.error(f"Error during QA chain execution: {e}")
            raise
