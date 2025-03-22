import streamlit as st

def process_video(youtube_url, agent):
    """Processa a URL do vídeo do YouTube e executa as etapas de transcrição, resumo, indexação e destaques"""
    results = {
        "transcription": None,
        "summary": None,
        "highlights": None,
        "steps": []
    }

    try:
        # 1. Transcrição
        transcription_task = f"Transcreva o vídeo do YouTube com URL {youtube_url}."
        transcription_result = agent.run(transcription_task)
        results["steps"].append({"name": "Transcrição", "content": transcription_result})
        st.session_state.transcription = transcription_result

        # 2. Resumo
        if transcription_result:
            summary_task = "Resuma a transcrição do vídeo que foi transcrito."
            summary_result = agent.run(summary_task)
            results["steps"].append({"name": "Resumo", "content": summary_result})
            st.session_state.summary = summary_result

        # 3. Indexação
        if transcription_result:
            index_task = "Indexe a transcrição do vídeo para permitir buscas futuras."
            index_result = agent.run(index_task)
            results["steps"].append({"name": "Indexação", "content": "Transcrição indexada com sucesso."})
            st.session_state.vectorstore = index_result

        # 4. Destaques Jornalísticos
        if summary_result:
            highlights_task = "Identifique pontos de interesse jornalístico no vídeo baseado na transcrição ou resumo, considerando o contexto atual."
            highlights_result = agent.run(highlights_task)
            results["steps"].append({"name": "Destaques Jornalísticos", "content": highlights_result})
            st.session_state.highlights = highlights_result

        results["transcription"] = transcription_result
        results["summary"] = summary_result
        results["highlights"] = highlights_result

    except Exception as e:
        import traceback
        error_msg = f"Erro ao processar o vídeo: {str(e)}\n\n{traceback.format_exc()}"
        results["steps"].append({"name": "Erro", "content": error_msg})

    return results
