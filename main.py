# Streamlit에 올릴때 필요한 코드 (sql 오류 fix위해), 로컬에서는 주석처리 필요
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# langchain 라이브러리 Import
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

# 환경변수, 이미지, 데이터조작, 임시파일, 운영체계관련, 입출력, 시간 관련 라이브러리 Import
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import os

# 한글자씩 답변하기 위한, Stream을 위한 라이브러리ㅡImport
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# 환경변수 로드
load_dotenv()
password_key = os.getenv('KEY')

# 챗봇 메인 함수
def app():

    # 화면 6:15로 분할
    col1, col2 = st.columns([6,15])

    with col1:
        image = Image.open('robot01.png')
        image = image.resize((170, 170))
        st.image(image)
        st.subheader('KMLA Chatbot')
        st.write(':star::star::star::star::star::star::star:')
        st.write('KMLA 28기 (2024-1학기)')
        st.write('구환희, 전지훈, 권휘우')
        st.write("")
        st.write("")
        st.write(':orange_book: LLM 구성')
        st.write(':small_blue_diamond: python, vscode')
        st.write(':small_blue_diamond: langchain F/W')
        st.write(':small_blue_diamond: streamlit')
        st.write(':small_blue_diamond: chroma vector DB')
        st.write(':small_blue_diamond: OpenAI GPT-4o')
        st.write(':small_blue_diamond: OpenAI embedding')
        st.write("")
        st.write("")
        st.write(':closed_book: DB 컨텐츠')
        st.write(':small_blue_diamond: homepage_v40')
        st.write(':small_blue_diamond: regulation_v15')
        st.write(':small_blue_diamond: knowhow_v40')
        st.write(':small_blue_diamond: namuwiki_v16')
        st.write("")



    with col2:
        st.write("")
        # 패스워드 받고, 화면 보여주기위한 텍스트 입력
        password = st.text_input(":key: 안내 받은 7 글자 패스워드를 넣어주세요!", type="password")

        if password:
            if password == password_key:

                # 비밀번호가 맞으면 화면 보여주기
                st.success("패스워드 확인 완료!")
                st.write("")
                user_input = st.text_input(":eight_pointed_black_star:민사고에 대해 질문하고 엔터를 눌러주세요!")

                st.write(':one: homepage : 국어 선생님을 알려줄래? 학교 8월 일정을 알려줄래?')
                st.write(':two: schoolregulation : 외출외박 신청은 어떻게 해? 벌점이 3점이상?')
                st.write(':three: knowhow : 비전트립이 뭐야? 학교에 어떤 동아리가 있어?')
                st.write(':four: namuwiki : 치킨데이가 뭐야? 바베큐 파티를 하려는데 방법은?')
                st.write(':robot_face: Magic 키워드! ~에 대해 상세히 알려줄래? 하면 더 잘 알려줘요!')

                if user_input:
                    # 질문하면 아래 내용 보여주기
                    st.write("")
                    st.write(':pencil2::pencil2::pencil2::pencil2::pencil2::pencil2::pencil2::pencil2: 답변 드립니다 :pencil2::pencil2::pencil2::pencil2::pencil2::pencil2::pencil2::pencil2:')

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

                    # 모델 : OpenAI의 gpt-4-turbo-2024-04-10 에서 gpt-4o-2024-05-13 로 수정(24.05.16 10::25)
                    question = user_input
                    chat_box = st.empty()
                    stream_handler = StreamHandler(chat_box)                
                    llm = ChatOpenAI(model_name="gpt-4o-2024-05-13", temperature=0, streaming=True, callbacks=[stream_handler])
                    qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())         
                    response = qa_chain.invoke({"query": question})
                    response_text = response['result']
                    
                    # 답변 글자 수 계산
                    char_count = len(response_text)
                    chat_box.markdown(response_text + f"\n\n총 글자 수: {char_count}")

            else:
                st.error("에러")

if __name__ == "__main__":
    app()