from typing import Any, Dict, List, Optional, Union
from smolagents.models.base import MultiStepModel
import groq
import json

class GroqModel(MultiStepModel):
    """Custom LLM model that interfaces with Groq's API."""
    
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b", temperature: float = 0.5, max_tokens: int = 4096):
        """Initialize the Groq LLM model.
        
        Args:
            api_key: Groq API key
            model: Groq model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.client = groq.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_input_token_count = None
        self.last_output_token_count = None
    
    def __call__(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Dict with 'content' key containing the generated text
        """
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        try:
            # Primeiro, tente com o modelo especificado
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            except groq.errors.NotFoundError:
                # Se o modelo não for encontrado, tente com llama-3-8b-8192, que provavelmente está disponível
                print(f"Model {self.model} not found, falling back to llama-3-8b-8192")
                response = self.client.chat.completions.create(
                    model="llama-3-8b-8192",
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            
            # Store token counts for later reference
            self.last_input_token_count = response.usage.prompt_tokens
            self.last_output_token_count = response.usage.completion_tokens
            
            return {"content": response.choices[0].message.content}
        except Exception as e:
            # Try listing available models to help with debugging
            try:
                available_models = self.client.models.list()
                available_model_ids = [model.id for model in available_models.data]
                error_msg = f"Error generating Groq response: {str(e)}\n\nAvailable models: {', '.join(available_model_ids)}"
            except:
                error_msg = f"Error generating Groq response: {str(e)}"
            
            raise Exception(error_msg)
