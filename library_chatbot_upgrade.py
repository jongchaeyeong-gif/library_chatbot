## 청킹작업 개선버전

import os
import streamlit as st
import nest_asyncio
import re  # 👈 [추가] 정규표현식 모듈

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document  # 👈 [추가] Document 객체
# 👈 [삭제] RecursiveCharacterTextSplitter는 더 이상 사용하지 않음
# from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# Gemini API 키 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# 👈 [변경] cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_pdf(file_path):
    """PDF를 페이지별로 로드합니다. (split X)"""
    loader = PyPDFLoader(file_path)
    return loader.load() # 👈 load_and_split() 대신 load() 사용

# 👈 [변경] 한국어 모델명 변수로 통일
KOREAN_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    """'조' 단위로 텍스트를 분할하고 벡터 스토어를 생성합니다."""

    # --- 👇 [변경] '조' 단위로 텍스트를 파싱하는 로직 추가 ---
    if not _docs:
        st.error("PDF 파일을 로드하지 못했습니다.")
        return None
        
    # 모든 페이지의 텍스트를 하나의 문자열로 합칩니다.
    full_text = "\n\n".join([doc.page_content for doc in _docs])
    source_file = _docs[0].metadata.get("source", "부경대학교 규정집.pdf")

    # '제' (공백*) 숫자 (공백*) '조' (공백*) (예: '제1조', '제 10 조')
    article_pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?\s*\(.+?\))' # (괄호 안의 제목 포함)
    
    # 정규표현식 수정: (제N조) 뿐만 아니라 (제N조의N) 및 (조항 제목)까지 포함
    # 예: 제1조(목적), 제2조(적용범위)
    # 괄호 안의 제목을 포함하는 더 강력한 정규표현식:
    article_pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?\s*\(.+?\))'
    
    # 텍스트를 '제N조(제목)' 기준으로 분할
    articles = re.split(article_pattern, full_text)
    
    split_docs = []
    
    # 첫 번째 조각은 보통 '전문' 또는 '목차'입니다.
    preamble = articles[0].strip()
    if preamble:
        split_docs.append(Document(
            page_content=preamble,
            metadata={"source": source_file, "article": "서문"}
        ))

    # '제1조(목적)' 같은 제목과 그 내용을 다시 합칩니다.
    for i in range(1, len(articles), 2):
        if i + 1 < len(articles):
            article_title = articles[i].strip() # "제1조(목적)"
            article_content = articles[i+1].strip() # "① 이 규정은..."
            
            full_article_text = f"{article_title}\n{article_content}"
            
            # '제1조' 부분만 추출
            article_key_match = re.match(r'(제\s*\d+\s*조(?:의\s*\d+)?)', article_title)
            article_key = article_key_match.group(1) if article_key_match else article_title
            article_key = re.sub(r'\s+', '', article_key) # 공백 제거 '제1조'
            
            split_docs.append(Document(
                page_content=full_article_text,
                metadata={"source": source_file, "article": article_key}
            ))
    # --- 👆 [변경] 파싱 로직 끝 ---

    if not split_docs:
        st.error("PDF에서 규정 '조'항을 파싱하는 데 실패했습니다. 파일 구조를 확인해주세요.")
        st.stop()

    st.info(f"📄 {len(split_docs)}개의 규정 조항(청크)으로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info(f"🤖 임베딩 모델 로드 중... ({KOREAN_EMBEDDING_MODEL})")
    embeddings = HuggingFaceEmbeddings(
        model_name=KOREAN_EMBEDDING_MODEL, # 👈 [변경] 한국어 모델
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    
    # 👈 [변경] 임베딩 모델을 한국어 모델로 통일
    embeddings = HuggingFaceEmbeddings(
        model_name=KOREAN_EMBEDDING_MODEL, 
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(persist_directory):
        st.info("기존 벡터 DB 로드 중...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # _docs (페이지 리스트)를 create_vector_store로 전달
        return create_vector_store(_docs)
        
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    file_path = "[챗봇프로그램및실습] 부경대학교 규정집.pdf"
    
    pages = load_pdf(file_path) # 👈 [변경] 함수명
    
    vectorstore = get_vectorstore(pages) # 👈 pages 리스트 전달
    retriever = vectorstore.as_retriever()

    # (이하 프롬프트 및 LLM 설정은 동일)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=False # 👈 [변경] 경고 제거
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-pro' 모델을 사용해보세요.")
        raise
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("국립부경대 도서관 규정 Q&A 챗봇 💬 📚")

# 👈 [추가] DB 삭제 안내
if os.path.exists("./chroma_db"):
    st.warning("⚠️ **[중요]** 코드가 변경되었습니다. "
               "기존 `chroma_db` 디렉터리를 **수동으로 삭제**한 후 앱을 새로고침(F5)해야 "
               "새로운 청킹 방식과 임베딩 모델이 적용됩니다.")

# 첫 실행 안내 메시지
if not os.path.exists("./chroma_db"):
    st.info("🔄 첫 실행입니다. 임베딩 모델 다운로드 및 PDF 처리 중... (약 3-5분 소요)")
    st.info("💡 이후 실행에서는 10-15초만 걸립니다!")

# Gemini 모델 선택 - 최신 모델명으로 수정
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-pro", "gemini-2.0-flash-exp"), # 👈 [변경] 최신 모델명
    index=0,
    help="최신 Flash 모델이 가장 빠르고 효율적입니다."
)

try:
    with st.spinner("🔧 챗봇 초기화 중... 잠시만 기다려주세요"):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇이 준비되었습니다!")
except Exception as e:
    st.error(f"⚠️ 초기화 중 오류 발생: {str(e)}")
    st.info("PDF 파일 경로와 API 키를 확인해주세요.")
    st.stop()

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                      "content": "국립부경대 도서관 규정에 대해 무엇이든 물어보세요!!!!!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config)
            
            answer = response['answer']
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    # 👈 [변경] 메타데이터 'article' 키 참조
                    article_info = doc.metadata.get('article', 'N/A')
                    st.markdown(f"**출처: {doc.metadata.get('source', 'N/A')} (조항: {article_info})**", 
                                help=doc.page_content)
