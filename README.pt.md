# 🧠 Agent YouTube Journalism (versão em Português)

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentytjournalism.streamlit.app/) [![Licença: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🔍 Visão geral do projeto

**Agent YouTube Journalism** é um assistente investigativo de IA de código aberto que transcreve, resume, analisa e responde perguntas sobre vídeos do YouTube — especialmente aqueles relacionados à política brasileira e temas de interesse público.

O sistema usa raciocínio multiagente e RAG (geração aumentada por recuperação) para:

- Transcrever vídeos em Português do Brasil (com a API Whisper da OpenAI)
- Gerar um resumo com DeepSeek via Groq Cloud
- Buscar contexto na web (usando DuckDuckGo)
- Destacar trechos relevantes para investigação jornalística
- Indexar e responder perguntas com base na transcrição + conhecimento geral se necessário

🟢 **Acesse online**: https://agentytjournalism.streamlit.app/  
⚠️ É necessário fornecer suas chaves de API do Groq, OpenAI e Hugging Face.

---

## ⚙️ Como funciona

1. **O usuário insere a URL do vídeo e as chaves de API**
2. O app:
   - Baixa o áudio com `yt-dlp`
   - Transcreve usando Whisper (`openai.Audio.transcribe`)
   - Resume com o modelo da Groq (DeepSeek)
   - Busca contexto atual na web
   - Destaca trechos de interesse jornalístico
   - Indexa a transcrição com FAISS
3. O usuário pode:
   - Visualizar a análise
   - Fazer perguntas sobre o vídeo (com RAG + fallback ao conhecimento do modelo)

O sistema usa `smolagents` para estruturar o raciocínio com um ciclo claro:

> **Pensamento → Ação → Observação**

---

## 🗂️ Estrutura do projeto

### 🔹 App principal

- `app.py`: App principal do Streamlit com abas de Análise e Perguntas
- `process_video.py`: Orquestra o pipeline completo (transcrição → resumo → destaques → indexação)
- `rag_question_tab.py`: Aba de perguntas baseada em RAG com controle de estado
- `agent_config.py`: Define as ferramentas usadas pelo agente `smolagents`
- `streamlit_app.yaml`: Arquivo de configuração para deploy no Streamlit Community Cloud
- `prompts.yaml`: Templates de prompts para resumo, destaques e pensamento dos agentes

### 🔹 Integração com Groq

- `groq_model.py`: Executa prompts com o modelo da Groq e controla tamanho dos prompts
- `list_groq_models.py`: Lista modelos disponíveis na conta Groq

### 🔹 Ferramentas (tools)

- `tools/youtube_transcriber.py`: Transcreve o vídeo usando Whisper
- `tools/summarization.py`: Resume a transcrição com DeepSeek
- `tools/web_search.py`: Faz buscas no DuckDuckGo
- `tools/journalistic_highlight.py`: Gera destaques de interesse público
- `tools/index_transcript.py`: Divide e indexa a transcrição com FAISS
- `tools/rag_query.py`: Executa a consulta RAG e permite complementar com conhecimento geral
- `tools/__init__.py`: Torna as ferramentas importáveis

### 🔹 Dependências

- `requirements.txt`: Dependências Python (testado com Python 3.12)
- `packages.txt`: Pacotes de sistema (ex: ffmpeg, ferramentas de build)

---

## 📈 Melhorias e recomendações

### ✅ Já corrigido

- Corrigido reprocessamento do vídeo ao mudar de aba
- Adicionado controle de estado para persistir transcrição e FAISS entre abas
- Permitida resposta mista RAG + conhecimento externo
- Corrigido `requirements.txt` e `packages.txt`

### 🚧 Melhorias futuras

1. **Cache e persistência**
   - Salvar FAISS, resumo e destaques em disco ou com `st.cache_data()`

2. **Melhorar UX no Streamlit**
   - Criar chat de perguntas com memória
   - Adicionar barra de progresso
   - Gerar relatório em PDF

3. **Aprimorar prompts**
   - Marcar partes da resposta como [FATO DO VÍDEO] ou [CONHECIMENTO DO LLM]

4. **Testes**
   - Adicionar testes com `pytest`

5. **Dividir os agentes**
   - Criar `VideoAnalysisAgent` e `QAAgent` separados
   - Opcionalmente usar CrewAI ou LangGraph para fluxos mais complexos

---

## 📄 Licença

Licença MIT. Ver arquivo `LICENSE`.

---

## 🙌 Créditos

Desenvolvido por Reinaldo Chaves (@reichaves) — jornalista, cientista de dados e especialista em jornalismo investigativo.  

---

## 💬 Dúvidas ou sugestões?

Abra uma [issue](https://github.com/reichaves/agent_yt_journalism/issues) ou envie via GitHub.
