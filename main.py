import streamlit as st
from PIL import Image
import pandas as pd
import tempfile
import os
import io
import time
import socket
import datetime

#from dotenv import load_dotenv

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

# 데이터 저장 함수
def save_data(new_data):
    try:
        existing_data = pd.read_excel("survey.xlsx")
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data

    updated_data.to_excel("survey.xlsx", index=False)
    st.success("설문 응답이 저장되었습니다.")

# 엑셀 파일 다운로드를 위한 함수
def download_excel():
    filename = 'survey.xlsx'
    with open(filename, "rb") as file:
        btn = st.download_button(
                label="설문 결과 다운로드",
                data=file,
                file_name=filename,
                mime="application/vnd.ms-excel"
            )

# 환경변수를 로드한다. (ChatGPT API Key를 .env라는 파일에 넣어야 함. OPENAI_API_KEY=시리얼넘버)
# load_dotenv()
admin_key = os.getenv('ADMIN')

def app():

    col1, col2 = st.columns([6,15])

    with col1:
        st.subheader(':parachute: Survey')
        image = Image.open('survey01.png')
        st.image(image, width=170)

        st.write("")
        st.write(':shamrock: KMLA Chatbot Team')
        st.write(':fire:구환희 전지훈 권휘우')
        st.write("")
        st.write("")
        st.write(':star: 스벅쿠폰 드려요!')     
        st.write('만족도조사 하신분께!')
        st.write('(제출하기 버튼 Click!)')
        st.write('**기간 : 5월13월~18토')  
        st.write(':cupcake:추첨 : 2만원 * 3명')
        st.write(':cupcake:선정 : 2만원 * 2명')
        st.write("")

        # 엑셀 다운로드
        password = st.text_input(":lock: 관리자", type="password")
        if password:
            if password == admin_key:
                # 비밀번호가 맞으면 다운로드 버튼 표시
                st.success("비밀번호 확인 완료")
                download_excel()
            else:
                st.error("에러")

    with col2:
        st.write("")        
        st.write(":heavy_check_mark: 챗봇을 사용하고 느낀점을 솔직히 써주세요! :yum:")
        
        # 사용자 입력 양식
        with st.form(key='survey_form'):
            email_address = st.text_input(":one: e-mail 주소를 적어주세요!")
            st.caption("(e-mail 주소를 추첨/쿠폰지급 이외의 용도로는 절대 사용하지 않습니다.)")
            st.write("")
            satisfaction = st.slider(":two: 답변에 대한 만족도 점수는? (10은 만족, 5는 보통)", 0, 10, 7)
            st.write("")
            positive_feedback = st.text_area(":three: 어떤 점이 마음에 들었죠?")
            st.write("")
            improvement_feedback = st.text_area(":four: 어떤 점을 개선하면 좋을까요?")
            st.write("")
            submit_button = st.form_submit_button(label='제출하기')

            # 제출 버튼이 눌렸을 때의 처리
            if submit_button:
                current_time = datetime.datetime.now() 
                user_ip = get_local_ip()
                new_data = pd.DataFrame({
                    "e-mail주소": [email_address],
                    "만족도": [satisfaction],
                    "좋았던 점": [positive_feedback],
                    "개선하고 싶은 점": [improvement_feedback],
                    "IP 주소": [user_ip],
                    "응답 시간": [current_time] 
                })
                
                # 파일에 데이터 저장
                save_data(new_data)
            st.write("")
            st.write(":smile: 감사합니다!")

if __name__ == "__main__":
    app()