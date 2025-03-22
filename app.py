import streamlit as st
from agent_config import create_agent
from rag_question_tab import render_rag_tab
from process_video import process_video  # função criada anteriormente

# Interface principal
st.set_page_config(page_title="Análise de Vídeos para Jornalismo", layout="wide")
st.title("🧠 Agente de IA para Jornalismo Investigativo com Vídeos do YouTube")

# Entrada das chaves de API
groq_key = st.text_input("🔑 Chave da API do Groq", type="password")
huggingface_token = st.text_input("🔑 Token da API do Hugging Face", type="password")

if groq_key and huggingface_token:
    if "agent" not in st.session_state:
        with st.spinner("Inicializando agente..."):
            st.session_state.agent = create_agent(groq_key, huggingface_token)

    tabs = st.tabs(["Transcrição e análise", "Destaques", "Perguntas"])

    with tabs[0]:
        st.subheader("📺 Analisar vídeo do YouTube")
        youtube_url = st.text_input("Cole aqui a URL do vídeo do YouTube:")

        if st.button("Analisar vídeo"):
            st.session_state.conversation_history = []
            with st.spinner("Processando vídeo..."):
                results = process_video(youtube_url, st.session_state.agent)

                for step in results["steps"]:
                    st.session_state.conversation_history.append(
                        ("assistant", f"**{step['name']}:**\n\n{step['content']}")
                    )

                st.success("Análise concluída!")

        # Mostrar histórico da análise
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
