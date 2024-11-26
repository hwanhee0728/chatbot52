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

#load_dotenv()
password_key = os.getenv('KEY')

image = Image.open('robot01.png')
image = image.resize((150, 150))
st.image(image)

st.header("KMLA Chatbot")

password = st.text_input(":name_badge: 7글자 패스워드를 넣어주세요!", type="password")

if password:
    if password == password_key:
        st.success("패스워드 확인 완료!")
        user_input = st.text_input(":eight_pointed_black_star:민사고에 대해 질문하고 엔터를 눌러주세요!")

        if user_input:
            # 질문하면 아래 내용 보여주기
            st.write(':pencil2::pencil2::pencil2: 답변 드립니다 :pencil2::pencil2::pencil2:')

            embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            try:
                db = Chroma(persist_directory="chromadb_ada2", embedding_function=embeddings_model)
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
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, max_tokens=3000, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

            try:
                qa_chain.invoke({"query": question})
            except Exception as e:
                st.error(f"Error during QA chain execution: {e}")
                raise
