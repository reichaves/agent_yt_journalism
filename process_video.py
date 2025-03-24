# Novo process_video.py usando Whisper API externamente
from agent_config import create_agent
from groq_model import chunk_text, summarize_chunks
from tools.youtube_transcriber import YouTubeTranscriberTool

def process_video(url: str, groq_api_key: str, huggingface_api_token: str, openai_api_key: str):
    try:
        # Transcreve o vídeo externamente antes do agente
        transcriber = YouTubeTranscriberTool()
        transcript = transcriber.forward(url, openai_api_key=openai_api_key)

        if not isinstance(transcript, str):
            return "Erro: a transcrição não foi retornada como string."

        # Cria o agente após a transcrição
        agent = create_agent(groq_api_key, huggingface_api_token)

        # Aplica resumo com chunking se necessário
        if len(transcript) > 3000:
            chunks = chunk_text(transcript)
            summarized = summarize_chunks(chunks, summarizer_model=agent.model)
            return summarized

        return transcript
    except Exception as e:
        return f"Erro ao processar o vídeo: {e}"
