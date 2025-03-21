# Este arquivo está vazio intencionalmente para tornar o diretório 'tools' um pacote Python
# Isso permite que os módulos neste diretório sejam importados em outros lugares do código

from tools.youtube_transcriber import YouTubeTranscriberTool
from tools.web_search import WebSearchTool
from tools.rag_query import RAGQueryTool
from tools.journalistic_highlight import JournalisticHighlightTool

__all__ = [
    'YouTubeTranscriberTool',
    'WebSearchTool',
    'RAGQueryTool',
    'JournalisticHighlightTool'
]
