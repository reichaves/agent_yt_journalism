# Novo groq_model.py compatível com SmolAgents (modelo chamável + suporte a kwargs + retorno ChatMessage)
from typing import Any
import groq
from dataclasses import dataclass

@dataclass
class ChatMessage:
    role: str
    content: str

class GroqModel:
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b", temperature: float = 0.3, max_tokens: int = 2048):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = groq.Client(api_key=self.api_key)

    def __call__(self, prompt: str, **kwargs) -> Any:
        """Torna o modelo chamável diretamente e compatível com argumentos adicionais (ex: stop_sequences)"""
        try:
            kwargs.pop("stop_sequences", None)  # remove argumento não suportado pela Groq API

            # Garante que o prompt seja string (Groq não aceita listas ou objetos complexos)
            if isinstance(prompt, list):
                prompt = "\n".join(str(p) for p in prompt)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return ChatMessage(role="assistant", content=response.choices[0].message.content.strip())
        except Exception as e:
            return ChatMessage(role="assistant", content=f"Erro ao executar modelo Groq: {str(e)}")
