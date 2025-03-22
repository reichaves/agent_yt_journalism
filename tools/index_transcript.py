from typing import Any
from smolagents.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class IndexTranscriptTool(Tool):
    name = "index_transcript"
    description = "Indexes a transcript using FAISS and HuggingFace embeddings."
    inputs = {
        'transcript': {'type': 'string', 'description': 'Transcript to index'}
    }
    output_type = "object"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_initialized = True

    def forward(self, transcript: str) -> Any:
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = splitter.create_documents([transcript])
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(texts, embeddings)
            return vectorstore
        except Exception as e:
            import traceback
            return f"Erro ao indexar transcrição: {str(e)}\n\n{traceback.format_exc()}"
