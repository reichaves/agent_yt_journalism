from typing import Any, Optional, Dict
from smolagents.tools import Tool
import logging
import groq

class RAGQueryTool(Tool):
    name = "rag_query"
    description = "Answers a question about the video content using Retrieval-Augmented Generation."
    inputs = {
        'question': {'type': 'string', 'description': 'The question to answer about the video content'},
        'vectorstore': {'type': 'object', 'description': 'The vector store containing the video transcript chunks'},
        'llm_api_key': {'type': 'string', 'description': 'API key for the LLM service'}
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_initialized = True

    def forward(self, question: str, vectorstore: Any, llm_api_key: str) -> str:
        """Performs a RAG query against the video transcript"""
        try:
            if vectorstore is None:
                return "Não há transcrição indexada disponível. Por favor, transcreva um vídeo primeiro."
            
            # Initialize Groq client
            client = groq.Client(api_key=llm_api_key)
            
            # Create retriever from vectorstore
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # Retrieve relevant contexts
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response with Groq
            system_content = """Você é um assistente especializado em vídeos que responde perguntas com base 
            em transcrições. Responda apenas com informações encontradas no contexto fornecido. 
            Se a resposta não estiver no contexto, admita que não pode responder com base na 
            transcrição disponível."""
            
            user_content = f"""
            Responda à seguinte pergunta com base na transcrição do vídeo:
            
            Contexto da transcrição:
            {context}
            
            Pergunta: {question}
            """
            
            chat_response = client.chat.completions.create(
                model="deepseek-coder-33b-instruct",  # Modelo DeepSeek disponível no Groq
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            response_content = chat_response.choices[0].message.content
            
            return f"Resposta baseada na transcrição do vídeo:\n\n{response_content}"
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return f"Erro ao responder à pergunta: {str(e)}\n\nTraceback:\n{traceback_str}"
