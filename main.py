__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
password_key = os.getenv('KEY')

st.markdown('<a href="https://surveykmla.streamlit.app/" target="_blank">챗봇 사용후 Feedback주실 분 여기 클릭!</a>', unsafe_allow_html=True)
image = Image.open('chatbot21.png')
image = image.resize((600, 56))
st.image(image)

password = st.text_input(":strawberry: 7글자 패스워드를 넣어주세요!", type="password")

if password:
    if password == password_key:

        st.success("패스워드 확인 완료!")
        st.write("")
        user_input = st.text_input(":apple: 민사고에 대해 질문하고 엔터를 눌러주세요!")

        if user_input:
            # 질문하면 아래 내용 보여주기
            st.write("")
            st.write(':tangerine::tangerine::tangerine::tangerine: 답변 드립니다 :tangerine::tangerine::tangerine::tangerine:')

            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            try:
                db = Chroma(persist_directory="chromadb_ada", embedding_function=embeddings_model)
            except Exception as e:
                st.error(f"Error initializing database: {e}")
                raise

            # Stream구현을(한글자씩 쓰여지는 기능) 위해 Handler 만들기
            class StreamHandler(BaseCallbackHandler):
                def __init__(self, container, initial_text=""):
                    self.container = container
                    self.text = initial_text

                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    self.text += token
                    self.container.markdown(self.text)

            question = user_input
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True, callbacks=[stream_handler], max_tokens=2000)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

            try:
                qa_chain.invoke({"query": question})
            except Exception as e:
                st.error(f"Error during QA chain execution: {e}")
                raise
