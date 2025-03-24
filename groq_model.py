# GroqModel atualizado com truncamento, conversão segura e estrutura compatível com SmolAgents
from typing import Any, List
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
        """
        Modelo chamável que adapta o prompt, remove argumentos não aceitos pela Groq
        e retorna um ChatMessage esperado pelo SmolAgents.
        """
        try:
            # Remove parâmetros não suportados pela API da Groq
            # Documentação: https://console.groq.com/docs/api-reference/chat
            # Exemplo de argumento rejeitado: stop_sequences
            kwargs.pop("stop_sequences", None)

            # Garante que prompt seja string
            if isinstance(prompt, list):
                prompt = "\n".join(str(p) for p in prompt)

            # Truncamento para evitar erro 413 (limite de tokens por minuto)
            max_prompt_chars = 15000
            if isinstance(prompt, str) and len(prompt) > max_prompt_chars:
                prompt = f"""
Thought: O prompt é muito longo e pode ultrapassar o limite de tokens da Groq. Vou truncá-lo para evitar erro.

Code:
```py
prompt = prompt[:{max_prompt_chars}] + "\n[Texto truncado para atender limite de tokens da Groq]"
```
<end_code>
Observation: Prompt truncado com sucesso.

{prompt[:max_prompt_chars]}
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            content = str(response.choices[0].message.content).strip()
            return ChatMessage(role="assistant", content=content)

        except Exception as e:
            return ChatMessage(role="assistant", content=f"Erro ao executar modelo Groq: {str(e)}")

# Funções auxiliares para chunking e resumo de transcrições longas

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Divide um texto longo em partes menores com limite de caracteres."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_chunks(chunks: List[str], summarizer_model: Any) -> str:
    """Resume cada chunk usando um modelo de linguagem e concatena os resumos."""
    summaries = []
    for i, chunk in enumerate(chunks):
        structured_prompt = f"""
Thought: Preciso resumir o trecho de vídeo abaixo em português de forma clara.

Code:
```py
texto = """{chunk}"""
summarize(texto)
```
<end_code>
"""
        summary = summarizer_model(structured_prompt)
        summaries.append(summary.content if hasattr(summary, 'content') else str(summary))
    return "\n\n".join(summaries)
