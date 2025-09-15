# Local URL : 내부 PC에서만 접속 가능
# Network URL : 사설 IP (내부망)
# External URL: 공인 IP (외부망)

import os
import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai import GenerationConfig

# LangChain : LLM 모델을 활용한 어플리케이션 개발을 위한 오픈소스 프레임워크
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

st.title("나만의 챗봇")

# 마크다운 형태의 텍스트 표현
st.markdown("streamlit을 사용한 나만의 챗봇!")

load_dotenv()

if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# 대화 기록(session)을 저장하기 위한  (key : value 형태)
# 대화 기록은 []에 저장됨
# st.session_state['messages']=[] >> 이대로 실행하면 새로고침 할 때마다 초기화됨
if 'chat' not in st.session_state:
    st.session_state['chat'] = [] # [('user', '안녕하세요'), ('assistant', '반가워요')]

# 새로운 대화 추가
def add_message(role, message):
    '''
    st.session_state['chat'].append(('user', user_input))
    st.session_state['chat'].append(('assistant', user_input))
    '''
    st.session_state['chat'].append(ChatMessage(role=role, content=message))

# 이전 대화를 출력
def print_messages():
    '''
    for role, message in st.session_state['chat']:
        st.chat_message(role).write(message)
    '''
    for chat_message in st.session_state['chat']:
        st.chat_message(chat_message.role).write(chat_message.content)

def create_chain():
    '''
    # chain = prompt | LLM | parser
    return chain
    '''
    # 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 친절한 AI 챗봇이야"),
            ("human",  "Question:\n{question}"),
        ]
    )
    # LLM 모델 생성
    llm = ChatGoogleGenerativeAI(                 
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")  # (명시해도 됨)
    )

    # 출력 parser
    output_parser = StrOutputParser()

    # Chain 생성
    chain = prompt | llm | output_parser
    return chain

print_messages()

# 입력창 생성
user_input=st.chat_input("궁금한 내용을 입력하세요")

# 입력창에 입력하는 순간 user_input에 입력한 내용이 담긴다
if user_input: # 입력이 들어오면
    # st.write(f"사용자 입력 : {user_input}") # 대화창에 입력값 출력

    # 대화창 기본 틀 : Web에 대화를 출력 >> 이대로 하면 대화 내역이 쌓이지 않음 (session_state 사용)
    # 사용자 입력
    st.chat_message("user").write(user_input)

    # chain 생성
    chain = create_chain()
    llm_answer = chain.invoke({'question': user_input})

    # llm 답변
    st.chat_message("assistant").write(llm_answer)

    # 대화기록을 저장
    add_message('user', user_input)
    add_message('assistant', llm_answer)