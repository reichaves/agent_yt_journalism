from agent_config import create_agent
from groq_model import chunk_text, summarize_chunks
from tools.index_transcript import IndexTranscriptTool
from tools.journalistic_highlight import JournalisticHighlightTool
from tools.web_search import WebSearchTool
from tools.youtube_transcriber import YouTubeTranscriberTool
import streamlit as st

def process_video(url: str, groq_api_key: str, huggingface_api_token: str, openai_api_key: str):
    try:
        # Transcrever vídeo
        transcriber = YouTubeTranscriberTool()
        transcript = transcriber.forward(url=url, openai_api_key=openai_api_key)
        if not isinstance(transcript, str):
            return "Falha na transcrição."

        st_text = transcript.split("Transcrição do vídeo:")[-1].strip()
        st.session_state.transcript = st_text

        # Criar agente
        agent = create_agent(groq_api_key, huggingface_api_token)
        st.session_state.agent = agent

        # Resumir se for longo
        if len(st_text) > 3000:
            chunks = chunk_text(st_text)
            summarized = summarize_chunks(chunks, summarizer_model=agent.model)
            result_text = summarized
        else:
            result_text = st_text

        # Indexar para RAG
        indexer = IndexTranscriptTool()
        vectorstore = indexer.forward(transcript=st_text)
        st.session_state.vectorstore = vectorstore

        # Buscar contexto na web
        search_tool = WebSearchTool()
        context = search_tool.forward(query="Carla Zambelli julgamento STF")

        # Gerar destaques jornalísticos
        highlighter = JournalisticHighlightTool()
        highlights = highlighter.forward(
            context=st_text,
            search_results=context,
            llm_api_key=groq_api_key
        )
        st.session_state.highlights = highlights

        return result_text

    except Exception as e:
        import traceback
        return f"Erro ao processar o vídeo: {e}\n\n{traceback.format_exc()}"
