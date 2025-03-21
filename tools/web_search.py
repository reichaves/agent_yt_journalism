from typing import Any, Optional
from smolagents.tools import Tool

class WebSearchTool(Tool):
    name = "web_search"
    description = "Searches the web for information using DuckDuckGo."
    inputs = {'query': {'type': 'string', 'description': 'The search query to find information on the web.'}}
    output_type = "string"

    def __init__(self, max_results=5, **kwargs):
        super().__init__()
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: `pip install duckduckgo-search`."
            ) from e
        self.ddgs = DDGS(**kwargs)
        self.is_initialized = True

    def forward(self, query: str) -> str:
        """Performs a web search and returns formatted results"""
        try:
            results = self.ddgs.text(query, max_results=self.max_results)
            
            if len(results) == 0:
                return f"Nenhum resultado encontrado para a busca: '{query}'"
            
            formatted_results = "Resultados da busca:\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"### Resultado {i}: {result['title']}\n"
                formatted_results += f"{result['body']}\n"
                formatted_results += f"**Fonte:** {result['href']}\n\n"
            
            return formatted_results
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return f"Erro ao realizar a busca: {str(e)}\n\nTraceback:\n{traceback_str}"
