import streamlit as st
from agent_config import create_agent
from rag_question_tab import render_rag_tab
from process_video import process_video

st.set_page_config(page_title="Análise de Vídeos para Jornalismo", layout="wide")
st.title("🧠 Agente de IA para Jornalismo Investigativo com Vídeos do YouTube")

groq_key = st.text_input("🔑 Chave da API do Groq", type="password")
huggingface_token = st.text_input("🔑 Token da API do Hugging Face", type="password")
openai_key = st.text_input("🔑 Chave da API da OpenAI (para Whisper)", type="password")

if groq_key and huggingface_token and openai_key:
    tabs = st.tabs(["Transcrição e análise", "Destaques", "Perguntas"])

    with tabs[0]:
        st.subheader("📺 Analisar vídeo do YouTube")
        youtube_url = st.text_input("Cole aqui a URL do vídeo do YouTube:")

        if st.button("Analisar vídeo"):
            st.session_state.conversation_history = []
            with st.spinner("Processando vídeo..."):
                try:
                    result = process_video(youtube_url, groq_key, huggingface_token, openai_key)

                    if isinstance(result, str):
                        st.session_state.conversation_history.append(("assistant", result))
                    elif isinstance(result, dict) and "steps" in result:
                        for step in result["steps"]:
                            st.session_state.conversation_history.append(
                                ("assistant", f"**{step['name']}:**\n\n{step['content']}")
                            )
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
    st.info("Por favor, insira suas chaves de API para iniciar.")
