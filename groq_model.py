
from typing import Any
import groq

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro ao executar modelo Groq: {str(e)}"
