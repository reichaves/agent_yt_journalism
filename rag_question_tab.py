import streamlit as st

def render_rag_tab():
    st.header("📚 Pergunte sobre o conteúdo do vídeo")

    # Verifica se há transcrição indexada
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        st.warning("Nenhuma transcrição indexada disponível. Transcreva e indexe um vídeo primeiro.")
        return

    # Campo de pergunta do usuário
    user_question = st.text_input("Digite sua pergunta sobre o conteúdo do vídeo:")

    if st.button("Responder") and user_question:
        with st.spinner("Gerando resposta..."):
            try:
                # Executa pergunta via agente
                response = st.session_state.agent.run(
                    f"Responda à seguinte pergunta com base na transcrição indexada: {user_question}"
                )

                # Extrai conteúdo textual da resposta
                final_response = response.content if hasattr(response, "content") else str(response)

                # Armazena no histórico
                if "conversation_history" not in st.session_state:
                    st.session_state.conversation_history = []
                st.session_state.conversation_history.append(("user", user_question))
                st.session_state.conversation_history.append(("assistant", final_response))

                st.markdown(f"**Resposta:**\n\n{final_response}")

            except Exception as e:
                st.error(f"Erro ao gerar resposta: {str(e)}")
