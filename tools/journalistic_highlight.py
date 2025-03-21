from typing import Any, Optional, Dict, List
from smolagents.tools import Tool

class JournalisticHighlightTool(Tool):
    name = "journalistic_highlight"
    description = "Identifies journalistically relevant highlights from the video based on current events."
    inputs = {
        'context': {'type': 'string', 'description': 'The video transcript or summary'},
        'search_results': {'type': 'string', 'description': 'Additional web search results for context'},
        'llm_api_key': {'type': 'string', 'description': 'API key for the LLM service'}
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_initialized = True

    def forward(self, context: str, search_results: str, llm_api_key: str) -> str:
        """Identifies journalistically relevant highlights from the video content"""
        try:
            from langchain_groq import ChatGroq
            
            # Initialize LLM for highlight analysis
            llm = ChatGroq(
                groq_api_key=llm_api_key,
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.3
            )
            
            prompt = f"""
            Você é um jornalista investigativo experiente. Analise o texto a seguir, que é a transcrição
            ou resumo de um vídeo em Português do Brasil, e destaque trechos que merecem investigação
            jornalística adicional. Considere:
            
            1. Afirmações que podem ser verificadas factualmente
            2. Conexões com notícias ou eventos atuais
            3. Declarações controversas ou potencialmente enganosas
            4. Implicações para políticas públicas ou interesse social
            5. Informações que parecem novas ou pouco divulgadas
            
            Formate sua resposta como:
            
            # Pontos de Interesse Jornalístico
            
            ## Destaque 1: [Título breve]
            **Trecho relevante:** [Trecho exato do texto]
            **Por que investigar:** [Explicação sobre o valor jornalístico]
            **Sugestão de abordagem:** [Como um jornalista poderia verificar ou explorar este ponto]
            
            [Repita o formato para cada destaque, com no mínimo 3 e no máximo 5 destaques]
            
            Texto a analisar:
            {context}
            
            Contexto atual (resultados de busca na web):
            {search_results}
            """
            
            response = llm.invoke(prompt)
            highlights = response.content
            
            return highlights
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return f"Erro ao encontrar destaques jornalísticos: {str(e)}\n\nTraceback:\n{traceback_str}"
