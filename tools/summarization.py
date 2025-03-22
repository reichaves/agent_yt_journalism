from typing import Any
from smolagents.tools import Tool
from langchain_groq import ChatGroq

class SummarizationTool(Tool):
    name = "video_summarizer"
    description = "Resumes the video transcript in Portuguese."
    inputs = {
        'transcript': {'type': 'string', 'description': 'The video transcript'},
        'llm_api_key': {'type': 'string', 'description': 'API key for the LLM'}
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_initialized = True

    def forward(self, transcript: str, llm_api_key: str) -> str:
        try:
            llm = ChatGroq(
                groq_api_key=llm_api_key,
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.3
            )

            prompt = f"""
            Você é um jornalista experiente. Abaixo está a transcrição de um vídeo em Português.
            Gere um resumo objetivo e claro com os principais pontos abordados:

            {transcript}
            """
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            import traceback
            return f"Erro ao resumir o vídeo: {str(e)}\n\n{traceback.format_exc()}"
