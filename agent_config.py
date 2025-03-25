import os
import yaml
import functools
from typing import Dict, Any
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
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as stream:
                return yaml.safe_load(stream)
        else:
            return {
                "system_prompt": "Você é um assistente de IA útil."
            }
    except Exception as e:
        print(f"Erro ao carregar prompts.yaml: {e}")
        return {"system_prompt": "Você é um assistente de IA útil."}

def create_final_answer_tool():
    from smolagents.tools import Tool

    class FinalAnswerTool(Tool):
        name = "final_answer"
        description = "Fornece uma resposta final para o problema."
        inputs = {'answer': {'type': 'any', 'description': 'A resposta final ao problema'}}
        output_type = "any"

        def forward(self, answer: Any) -> Any:
            return answer

        def __init__(self, *args, **kwargs):
            self.is_initialized = True

    return FinalAnswerTool()

def create_agent(groq_api_key: str, huggingface_api_token: str, max_steps: int = 1) -> CodeAgent:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

    tools = [
        YouTubeTranscriberTool(),
        WebSearchTool(),
        SummarizationTool(),
        IndexTranscriptTool(),
        RAGQueryTool(),
        JournalisticHighlightTool(),
        create_final_answer_tool()
    ]

    prompt_templates = load_prompt_templates()

    model = GroqModel(
        api_key=groq_api_key,
        model="deepseek-r1-distill-llama-70b",
        temperature=0.5,
        max_tokens=4096
    )

    agent = CodeAgent(
        model=model,
        tools=tools,
        max_steps=max_steps,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name="JournalistAssistant",
        description=AGENT_DESCRIPTION,
        prompt_templates=prompt_templates
    )

    return agent
