import streamlit as st
from process_video import process_video
from tools.rag_query import RAGQueryTool

def rag_question_tab():
    st.header("🔍 Perguntas sobre o vídeo")

    url = st.text_input("URL do vídeo do YouTube", key="rag_url")
    question = st.text_input("Digite sua pergunta sobre o vídeo", key="user_question")

    openai_api_key = st.text_input("Sua OpenAI API Key", type="password", key="openai_rag_key")
    huggingface_api_key = st.text_input("Sua Hugging Face API Key", type="password", key="huggingface_rag_key")

    if url and question and openai_api_key and huggingface_api_key:
        if (
            "vectorstore" not in st.session_state
            or "transcript" not in st.session_state
            or st.session_state.get("processed_url") != url
        ):
            with st.spinner("Processando vídeo e criando index..."):
                transcript, vectorstore = process_video(url, openai_api_key, huggingface_api_key)
                st.session_state.vectorstore = vectorstore
                st.session_state.transcript = transcript
                st.session_state.processed_url = url
        else:
            vectorstore = st.session_state.vectorstore
            transcript = st.session_state.transcript

        if vectorstore:
            rag_tool = RAGQueryTool()
            with st.spinner("Consultando a transcrição..."):
                result = rag_tool.forward(
                    question=question,
                    vectorstore=vectorstore,
                    llm_api_key=openai_api_key
                )
            st.markdown("### Resposta:")
            st.write(result)
