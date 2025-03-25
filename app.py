import streamlit as st
from agent_config import create_agent
from process_video import process_video
from rag_question_tab import render_rag_tab

# Aba de configurações no sidebar
with st.sidebar:
    st.header("🔑 Configurações")
    groq_key = st.text_input("Chave da API do Groq", type="password")
    hf_key = st.text_input("Token da API do HuggingFace", type="password")

    if groq_key:
        st.session_state["groq_api_key"] = groq_key
        st.success("Chave da Groq salva!")

    if hf_key:
        st.session_state["huggingface_token"] = hf_key
        st.success("Token do HuggingFace salvo!")

# Interface principal
st.title("🎥 Assistente de Vídeos para Jornalistas")

tab1, tab2 = st.tabs(["📼 Transcrição e Análise", "📌 Perguntas sobre o Vídeo"])

with tab1:
    process_video()

with tab2:
    render_rag_tab()
