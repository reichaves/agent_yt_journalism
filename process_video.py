from agent_config import create_agent
from groq_model import chunk_text, summarize_chunks
from tools.index_transcript import IndexTranscriptTool
from tools.journalistic_highlight import JournalisticHighlightTool
from tools.web_search import WebSearchTool
from tools.youtube_transcriber import YouTubeTranscriberTool
import streamlit as st

def process_video(url: str, groq_api_key: str, huggingface_api_token: str, openai_api_key: str):
    try:
        st.info("Transcrevendo vídeo...")
        transcriber = YouTubeTranscriberTool()
        transcript = transcriber.forward(url=url, openai_api_key=openai_api_key)
        if not isinstance(transcript, str):
            st.error("Falha na transcrição.")
            return

        st_text = transcript.split("Transcrição do vídeo:")[-1].strip()
        st.session_state.transcript = st_text

        st.success("Transcrição concluída.")

        st.info("Criando agente...")
        agent = create_agent(groq_api_key, huggingface_api_token, max_steps=4)
        st.session_state.agent = agent

        st.info("Resumindo vídeo se necessário...")
        if len(st_text) > 3000:
            chunks = chunk_text(st_text)
            summarized = summarize_chunks(chunks, summarizer_model=agent.model)
            result_text = summarized
        else:
            result_text = st_text

        st.success("Resumo concluído.")

        st.info("Indexando transcrição para RAG...")
        indexer = IndexTranscriptTool()
        vectorstore = indexer.forward(transcript=st_text)
        st.session_state.vectorstore = vectorstore
        st.success("Indexação feita.")

        st.info("Buscando contexto atual na web...")
        search_tool = WebSearchTool()
        context = search_tool.forward(query="Carla Zambelli julgamento STF")

        st.info("Gerando destaques jornalísticos...")
        highlighter = JournalisticHighlightTool()
        highlights = highlighter.forward(
            context=st_text,
            search_results=context,
            llm_api_key=groq_api_key
        )
        st.session_state.highlights = highlights
        st.success("Destaques gerados com sucesso.")

        # Exibição final no app
        st.subheader("📝 Resumo ou Transcrição")
        st.write(result_text)

        st.subheader("🔍 Destaques Jornalísticos")
        st.markdown(highlights)

        with st.expander("🔧 Logs e conteúdo bruto"):
            st.code(st_text[:1000], language="text")
            st.code(highlights[:1000], language="markdown")

        return result_text

    except Exception as e:
        import traceback
        st.error("Erro ao processar o vídeo.")
        return f"Erro ao processar o vídeo: {e}\n\n{traceback.format_exc()}"
