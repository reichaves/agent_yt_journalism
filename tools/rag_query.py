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

    def forward(self, question: str, vectorstore: Any, llm_api_key: str, use_general_knowledge: bool = True) -> str:
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
            system_content = (
                "Você é um assistente especialista em análise de vídeos e política brasileira.\n\n"
                "Responda à pergunta com base:\n"
                "1. No contexto da transcrição do vídeo\n"
                "2. No seu próprio conhecimento geral\n"
                "3. (Opcional) Em fatos recentes, se aplicável\n\n"
                "Deixe claro quando uma parte da resposta vem da transcrição e quando vem do seu conhecimento geral."
                if use_general_knowledge else
                "Você é um assistente especializado em vídeos que responde perguntas com base apenas na transcrição fornecida.\n"
                "Responda exclusivamente com informações contidas no contexto.\n"
                "Se a informação não estiver no contexto, diga que não pode responder com base no que foi fornecido."
            )

            user_content = f"""
Pergunta: {question}

Transcrição do vídeo (para análise):
{context}
"""

            chat_response = client.chat.completions.create(
                model="deepseek-coder-33b-instruct",
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
