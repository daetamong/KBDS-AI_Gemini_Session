# -------------------------------------------------------
# 1. 필요한 라이브러리 불러오기
# -------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain 관련
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Gemini 기반 LLM + Embedding
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 벡터DB (FAISS)
from langchain_community.vectorstores.faiss import FAISS

# 기타 도구
from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter


# -------------------------------------------------------
# 2. 화면 구성
# -------------------------------------------------------
st.title("RAG 기반 나만의 챗봇 (Gemini)")
st.markdown("내 문서를 불러와서 답변하는 챗봇 (Streamlit + LangChain + FAISS + Gemini)")


# -------------------------------------------------------
# 3. API Key 불러오기
# -------------------------------------------------------
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("환경변수에 GOOGLE_API_KEY가 없습니다. .env 파일을 확인하세요.")


# -------------------------------------------------------
# 4. 대화 기록 저장 공간 만들기
# -------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state["chat"] = []


def add_message(role, message):
    """대화 기록에 새로운 메시지 추가"""
    st.session_state["chat"].append(ChatMessage(role=role, content=message))


def print_messages():
    """지금까지의 대화를 출력"""
    for chat_message in st.session_state["chat"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# -------------------------------------------------------
# 5. RAG 세팅 (FAISS 벡터 DB)
# -------------------------------------------------------
# 1) 문서 임베딩 + 저장 (최초 1회)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 예시 텍스트 (여기에 원하는 도메인 문서 넣기 가능)
documents = [
    "Harrison worked at Kensho.",
    "LangChain is a framework for building applications with LLMs.",
    "Streamlit allows you to build interactive web apps in Python."
]

# 저장소 없을 때만 새로 생성
if not os.path.exists("faiss_index"):
    vectorstore = FAISS.from_texts(documents, embedding=embedding_model)
    vectorstore.save_local("faiss_index")

# 2) 기존 저장소 불러오기
vectorstore_new = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore_new.as_retriever()


# -------------------------------------------------------
# 6. RAG 체인 만들기
# -------------------------------------------------------
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Gemini 모델 불러오기
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # 또는 "gemini-1.5-pro"
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 체인 정의 (질문 → retriever → 프롬프트 → LLM → 답변)
chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")}
    | prompt
    | model
    | StrOutputParser()
)


# -------------------------------------------------------
# 7. 대화 출력 & 사용자 입력
# -------------------------------------------------------
print_messages()
user_input = st.chat_input("질문을 입력하세요 (문서 기반 검색 포함)")

if user_input:
    # 사용자 메시지 출력
    st.chat_message("user").write(user_input)

    # RAG 기반 응답 생성
    answer = chain.invoke({"question": user_input})

    # AI 응답 출력
    st.chat_message("assistant").write(answer)

    # 대화 기록 저장
    add_message("user", user_input)
    add_message("assistant", answer)
