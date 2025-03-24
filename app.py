# app.py atualizado para incluir OpenAI Whisper API
import streamlit as st
from agent_config import create_agent
from rag_question_tab import render_rag_tab
from process_video import process_video

st.set_page_config(page_title="Análise de Vídeos para Jornalismo", layout="wide")
st.title("🧠 Agente de IA para Jornalismo Investigativo com Vídeos do YouTube")

# Entrada das chaves de API
st.subheader("🔐 Chaves de Acesso")
groq_key = st.text_input("🔑 Chave da API do Groq", type="password")
huggingface_token = st.text_input("🔑 Token da API do Hugging Face", type="password")
openai_key = st.text_input("🔑 Chave da API da OpenAI (Whisper API)", type="password")

if groq_key and huggingface_token and openai_key:
    tabs = st.tabs(["Transcrição e análise", "Destaques", "Perguntas"])

    with tabs[0]:
        st.subheader("📺 Analisar vídeo do YouTube")
        youtube_url = st.text_input("Cole aqui a URL do vídeo do YouTube:")

        if st.button("Analisar vídeo"):
            st.session_state.conversation_history = []
            with st.spinner("Processando vídeo com Whisper API..."):
                try:
                    result = process_video(youtube_url, groq_key, huggingface_token, openai_key)

                    if isinstance(result, str):
                        st.session_state.conversation_history.append(("assistant", result))
                    else:
                        st.session_state.conversation_history.append(("assistant", str(result)))

                    st.success("Análise concluída!")
                except Exception as e:
                    st.session_state.conversation_history.append(("assistant", f"Erro ao processar vídeo: {e}"))

        if "conversation_history" in st.session_state:
            for speaker, msg in st.session_state.conversation_history:
                st.markdown(f"**{speaker.title()}:**\n\n{msg}", unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("📰 Destaques jornalísticos")
        if "highlights" in st.session_state:
            st.markdown(st.session_state.highlights)
        else:
            st.info("Nenhum destaque gerado ainda.")

    with tabs[2]:
        render_rag_tab()
else:
    st.info("Por favor, insira todas as chaves de API para iniciar.")
