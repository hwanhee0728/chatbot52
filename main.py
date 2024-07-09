__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import os

from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()
password_key = os.getenv('KEY')

image = Image.open('presentation.png')
image = image.resize((528, 120))
st.image(image)
st.write(':star: KMLA Chatbot Team :star: 28기 구환희, 전지훈, 권휘우')

password = st.text_input(":name_badge: 7글자 패스워드를 넣어주세요!", type="password")

if password:
    if password == password_key:

        st.success("패스워드 확인 완료!")
        st.write("")
        user_input = st.text_input(":eight_pointed_black_star:민사고에 대해 질문하고 엔터를 눌러주세요!")

        if user_input:
            # 질문하면 아래 내용 보여주기
            st.write("")
            st.write(':pencil2::pencil2::pencil2: 답변 드립니다 :pencil2::pencil2::pencil2:')

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
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())         
            qa_chain.invoke({"query": question})
                
