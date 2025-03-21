#!/usr/bin/env python
"""
Script para listar os modelos disponíveis na API Groq.
Útil para saber quais modelos estão disponíveis para uso.
"""

import os
import groq
import sys

def list_models(api_key=None):
    """Lista todos os modelos disponíveis na API Groq"""
    if not api_key:
        # Tenta obter a chave da API das variáveis de ambiente
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Erro: Nenhuma chave de API Groq fornecida.")
            print("Use: python list_groq_models.py SUA_API_KEY")
            print("Ou defina a variável de ambiente GROQ_API_KEY")
            sys.exit(1)
    
    try:
        # Inicializa o cliente da API Groq
        client = groq.Client(api_key=api_key)
        
        # Lista todos os modelos disponíveis
        models = client.models.list()
        
        print("\n==== Modelos Disponíveis na API Groq ====\n")
        
        for model in models.data:
            print(f"ID: {model.id}")
            print(f"Criado em: {model.created}")
            print(f"Proprietário: {model.owned_by}")
            print("-" * 50)
        
        print(f"\nTotal de modelos disponíveis: {len(models.data)}")
        
    except Exception as e:
        print(f"Erro ao listar modelos: {str(e)}")

if __name__ == "__main__":
    # Se uma chave de API for fornecida como argumento de linha de comando, use-a
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    list_models(api_key)
