import os
import yaml
from typing import Dict, List, Any, Optional
from smolagents import CodeAgent, HfApiModel
from tools.youtube_transcriber import YouTubeTranscriberTool
from tools.web_search import WebSearchTool
from tools.rag_query import RAGQueryTool
from tools.journalistic_highlight import JournalisticHighlightTool

def load_prompt_templates() -> Dict[str, Any]:
    """Load prompt templates from YAML file or return defaults"""
    try:
        if os.path.exists("prompts.yaml"):
            with open("prompts.yaml", 'r', encoding='utf-8') as stream:
                return yaml.safe_load(stream)
        else:
            # Return default prompt templates
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
                - rag_query: Responde perguntas sobre o conteúdo do vídeo usando RAG
                - journalistic_highlight: Identifica pontos de interesse jornalístico
                - final_answer: Fornece uma resposta final
                
                Lembre-se de que os jornalistas precisam de informações precisas, contextualizadas e com
                valor noticioso. Sempre indique quando uma informação precisa ser verificada ou investigada
                mais a fundo.
                """
            }
    except Exception as e:
        print(f"Error loading prompt templates: {e}")
        # Return a minimal default if there's an error
        return {"system_prompt": "You are a helpful AI assistant."}

def create_final_answer_tool():
    """Create the final answer tool"""
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

def create_agent(groq_api_key: str, huggingface_api_token: str) -> CodeAgent:
    """Create and configure the agent with all necessary tools"""
    
    # Set the Hugging Face API token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
    # Initialize tools
    youtube_transcriber = YouTubeTranscriberTool()
    web_search = WebSearchTool()
    rag_query = RAGQueryTool()
    journalistic_highlight = JournalisticHighlightTool()
    final_answer = create_final_answer_tool()
    
    # Load prompt templates
    prompt_templates = load_prompt_templates()
    
    # Initialize model
    model = HfApiModel(
        max_tokens=4096,
        temperature=0.5,
        model_id="deepseek-r1-distill-llama-70b",
        custom_role_conversions=None
    )
    
    # Create agent with tools
    agent = CodeAgent(
        model=model,
        tools=[
            youtube_transcriber,
            web_search,
            rag_query,
            journalistic_highlight,
            final_answer
        ],
        max_steps=8,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name="JournalistAssistant",
        description="An AI agent that helps journalists analyze YouTube videos in Brazilian Portuguese",
        prompt_templates=prompt_templates
        # Removido: authorized_imports
    )
    
    return agent
