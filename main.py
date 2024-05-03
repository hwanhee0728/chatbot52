# Streamlit에 올릴때 필요한 코드 (sql 오류 fix위해)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# langchain 라이브러리를 Import한다.
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

import streamlit as st
from PIL import Image
import pandas as pd
import tempfile
import os
import io
import time

# Stream을 위한 Import
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

import socket
import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter


# 엑셀 파일에 데이터를 기록하는 함수입니다.
def log_question_to_excel(question, ip_address, timestamp):
    filename = 'question_log.xlsx'
    try:
        # 엑셀 파일이 이미 존재하면 로드하고, 그렇지 않으면 새 파일을 생성합니다.
        try:
            workbook = load_workbook(filename)
            sheet = workbook.active
        except FileNotFoundError:
            workbook = Workbook()
            sheet = workbook.active
            sheet['A1'] = 'Question'
            sheet['B1'] = 'IP Address'
            sheet['C1'] = 'Timestamp'

        # 새로운 데이터를 추가합니다.
        new_row = (question, ip_address, timestamp)
        sheet.append(new_row)

        # 파일을 저장합니다.
        workbook.save(filename)
    except Exception as e:
        print(f"Error logging question to Excel: {e}")


# 로컬 IP 주소를 가져오는 함수입니다.
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 이 IP는 실제로 연결되지 않지만, 루프백 주소(127.0.0.1)를 반환하지 않도록 합니다.
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


# 환경변수를 로드한다. (ChatGPT API Key를 .env라는 파일에 넣어야 함. OPENAI_API_KEY=시리얼넘버)
load_dotenv()

password_key = os.getenv('KEY')
admin_key = os.getenv('ADMIN')


# 엑셀 파일 다운로드를 위한 함수
def download_excel():
    filename = 'question_log.xlsx'
    with open(filename, "rb") as file:
        btn = st.download_button(
                label="질문 로그 다운로드",
                data=file,
                file_name=filename,
                mime="application/vnd.ms-excel"
            )

def app():

    col1, col2 = st.columns([6,15])

    with col1:
        st.subheader(':robot_face: Chatbot')
        image = Image.open('robot01.png')
        st.image(image, width=170)

        st.write("")
        st.write(':shamrock: KMLA Chatbot Team')
        st.write(':fire:구환희 전지훈 권휘우')
        st.write("")
        st.write('[ Chroma DB 컨텐츠 ]')
        st.write(':one: homepage_v13')
        st.write(':two: schoolregulation_v13')
        st.write(':three: knowhow_v13')
        st.write(':four: namuwiki_v13')

        st.write("")
        st.write('[ 주요 Specification ]')
        st.write(':zap: LLM : OpenAI GPT-4')
        st.write(':zap: Framework : Langchain')
        st.write(':zap: Vector DB : Chroma')
        st.write(':zap: Embedding : OpenAI')
        st.write(':zap: Chunk : 600, 30')
        st.write(':zap: WWW : Github, Streamlit')

        st.write("")
        st.write("")

        # 엑셀 다운로드
        admin_password = st.text_input(":computer: 관리자", type="password")
        if admin_password:
            if admin_password == admin_key:
                st.success("비밀번호 확인 완료")
                # 엑셀 파일 다운로드 기능
                download_excel()
            else:
                st.error("잘못된 비밀번호입니다.")

    with col2:
        st.write("")

        password = st.text_input(":heavy_check_mark: 민사고 :gun: 패스워드 넣어주세요. :red_circle::red_circle::red_circle::red_circle::red_circle::red_circle::red_circle:", type="password")

        if password:
            if password == password_key:
                # 비밀번호가 맞으면 다운로드 버튼 표시
                st.success("패스워드 확인 완료!")

                st.write("")
                user_input = st.text_input(":eight_pointed_black_star:민사고에 대해 질문하고 엔터를 눌러주세요!")

                st.write(':one: homepage : 수학선생님? 학교시설? 동아리? 5월 귀가는?')
                st.write(':two: schoolregulation : 수강신청? 외출외박 신청? 혼정? 벌점이 3점이상?')
                st.write(':three: knowhow : 비트? 융프? 초아? 라피네? 줄임말 사례를 알려줘')
                st.write(':four: namuwiki : 치킨데이? 바비큐 방법? 한과영교류전? 아침기? 면학실?')

                if user_input:

                    st.write("")
                    st.write(':robot_face: 답변 드립니다!')

                    # 질문을 로깅합니다.
                    log_question_to_excel(user_input, get_local_ip(), datetime.datetime.now())

                    embeddings_model = OpenAIEmbeddings()
                    db = Chroma(persist_directory="chromadb_600", embedding_function=embeddings_model)

                    # Stream구현을(한글자씩 쓰여지는 기능) 위해 Handler 만들기
                    class StreamHandler(BaseCallbackHandler):
                        def __init__(self, container, initial_text=""):
                            self.container = container
                            self.text=initial_text
                        def on_llm_new_token(self, token: str, **kwargs) -> None:
                            self.text += token
                            self.container.markdown(self.text)

                    question = user_input
                    chat_box = st.empty()
                    stream_handler = StreamHandler(chat_box)                
                    llm = ChatOpenAI(model_name="gpt-4-turbo-2024-04-09", temperature=0, streaming=True, callbacks=[stream_handler])
                    qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())         
                    qa_chain.invoke({"query": question})

            else:
                st.error("에러")

if __name__ == "__main__":
    app()