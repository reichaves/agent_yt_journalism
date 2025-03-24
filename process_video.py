# process_video.py atualizado para usar GroqModel com chunking e fallback

from agent_config import create_agent
from groq_model import chunk_text, summarize_chunks

# Função principal para processar um vídeo
def process_video(url: str):
    try:
        # Cria o agente com configurações e ferramentas
        agent = create_agent()

        # Define a tarefa principal para o agente
        transcription_task = f"""
Sua tarefa é transcrever e resumir o conteúdo do vídeo do YouTube a seguir:

URL: {url}

O objetivo é fornecer uma transcrição precisa em português do Brasil. Se o vídeo for longo, aplique chunking e resumo.
"""

        transcription_result = agent.run(transcription_task)

        # Se o resultado for muito longo, aplicar chunking + resumo com o próprio modelo
        if hasattr(transcription_result, 'content') and len(transcription_result.content) > 3000:
            chunks = chunk_text(transcription_result.content)
            summarized = summarize_chunks(chunks, summarizer_model=agent.model)
            return summarized

        return transcription_result.content if hasattr(transcription_result, 'content') else transcription_result

    except Exception as e:
        return f"Erro ao processar o vídeo: {e}"
