import streamlit as st

def render_rag_tab():
    st.header("Pergunte sobre o vídeo")

    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        st.warning("Nenhuma transcrição indexada disponível. Transcreva e indexe um vídeo primeiro.")
        return

    user_question = st.text_input("Digite sua pergunta sobre o conteúdo do vídeo:")
    if st.button("Responder") and user_question:
        with st.spinner("Gerando resposta..."):
            try:
                response = st.session_state.agent.run(
                    f"Responda à seguinte pergunta com base na transcrição indexada: {user_question}"
                )
                st.session_state.conversation_history.append(("user", user_question))
                st.session_state.conversation_history.append(("assistant", response))
                st.markdown(f"**Resposta:**\n\n{response}")
            except Exception as e:
                st.error(f"Erro ao gerar resposta: {str(e)}")
