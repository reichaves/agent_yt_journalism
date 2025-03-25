import streamlit as st
from process_video import process_video
from rag_question_tab import render_rag_tab

# Aba de configurações no sidebar
with st.sidebar:
    st.header("🔑 Configurações")

    groq_key = st.text_input("Chave da API do Groq", type="password")
    hf_key = st.text_input("Token da API do HuggingFace", type="password")
    whisper_key = st.text_input("Chave da API do Whisper (OpenAI)", type="password")

    if groq_key:
        st.session_state["groq_api_key"] = groq_key
        st.success("Chave da Groq salva!")

    if hf_key:
        st.session_state["huggingface_token"] = hf_key
        st.success("Token do HuggingFace salvo!")

    if whisper_key:
        st.session_state["openai_api_key"] = whisper_key
        st.success("Chave do Whisper salva!")

# Interface principal
st.title("🎥 Assistente de Vídeos para Jornalistas")

tab1, tab2 = st.tabs(["📼 Transcrição e Análise", "📌 Perguntas sobre o Vídeo"])

with tab1:
    if "groq_api_key" in st.session_state and "openai_api_key" in st.session_state and "huggingface_token" in st.session_state:
        process_video()
    else:
        st.warning("Por favor, insira todas as três chaves de API no menu de configurações.")

with tab2:
    render_rag_tab()
