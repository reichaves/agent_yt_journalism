
import os
import yaml
import functools
from typing import Dict, List, Any, Optional
from smolagents import CodeAgent
from tools.youtube_transcriber import YouTubeTranscriberTool
from tools.web_search import WebSearchTool
from tools.rag_query import RAGQueryTool
from tools.journalistic_highlight import JournalisticHighlightTool
from tools.summarization import SummarizationTool
from tools.index_transcript import IndexTranscriptTool
from groq_model import GroqModel

AGENT_DESCRIPTION = (
    "Agente de IA para auxiliar jornalistas a analisar vídeos do YouTube em Português do Brasil."
)

@functools.lru_cache(maxsize=1)
def load_prompt_templates() -> Dict[str, Any]:
    try:
        if os.path.exists("prompts.yaml"):
            with open("prompts.yaml", 'r', encoding='utf-8') as stream:
                return yaml.safe_load(stream)
        else:
            return {
                "system_prompt": """
                Você é um agente de IA especializado em analisar vídeos do YouTube para fins jornalísticos.
                Seu objetivo é ajudar jornalistas a extrair informações valiosas de vídeos em Português do Brasil.
                
                Para resolver as tarefas, você deve seguir um ciclo de 'Thought:', 'Code:', e 'Observation:'.
                
                Em cada etapa, você deve:
                1. Explicar seu raciocínio no bloco 'Thought:'
                2. Escrever código Python simples no bloco 'Code:' (encerrado com '<end_code>')
                3. Observar os resultados no bloco 'Observation:'
                
                Suas respostas devem ser claras, estruturadas e úteis para jornalistas investigativos.
                No final, você deve fornecer uma resposta completa usando a ferramenta `final_answer`.
                
                Você tem acesso às seguintes ferramentas:
                - youtube_transcriber: Transcreve um vídeo do YouTube
                - web_search: Busca informações na web
                - video_summarizer: Resume a transcrição do vídeo
                - index_transcript: Indexa a transcrição para RAG
                - rag_query: Responde perguntas sobre o vídeo
                - journalistic_highlight: Identifica pontos de interesse jornalístico
                - final_answer: Fornece uma resposta final
                
                Lembre-se de que os jornalistas precisam de informações precisas, contextualizadas e com
                valor noticioso. Sempre indique quando uma informação precisa ser verificada ou investigada
                mais a fundo.
                """
            }
    except Exception as e:
        print(f"Error loading prompt templates: {e}")
        return {"system_prompt": "You are a helpful AI assistant."}

def create_final_answer_tool():
    from smolagents.tools import Tool

    class FinalAnswerTool(Tool):
        name = "final_answer"
        description = "Provides a final answer to the given problem."
        inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
        output_type = "any"

        def forward(self, answer: Any) -> Any:
            return answer

        def __init__(self, *args, **kwargs):
            self.is_initialized = True

    return FinalAnswerTool()

def create_agent(groq_api_key: str, huggingface_api_token: str, max_steps: int = 12) -> CodeAgent:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

    youtube_transcriber = YouTubeTranscriberTool()
    web_search = WebSearchTool()
    rag_query = RAGQueryTool()
    journalistic_highlight = JournalisticHighlightTool()
    summarizer = SummarizationTool()
    indexer = IndexTranscriptTool()
    final_answer = create_final_answer_tool()

    prompt_templates = load_prompt_templates()

    model = GroqModel(
        api_key=groq_api_key,
        model="deepseek-r1-distill-llama-70b",
        temperature=0.5,
        max_tokens=4096
    )

    agent = CodeAgent(
        model=model,
        tools=[
            youtube_transcriber,
            web_search,
            summarizer,
            indexer,
            rag_query,
            journalistic_highlight,
            final_answer
        ],
        max_steps=max_steps,
        verbosity_level=2,
        grammar=None,
        planning_interval=None,
        name="JournalistAssistant",
        description=AGENT_DESCRIPTION,
        prompt_templates=prompt_templates
    )

    return agent
