# -*- coding: utf-8
#

import streamlit as st
import os
import yaml
import json
import uuid
import datetime
import tempfile
import pytz
import re
from typing import Any, Dict, List, Optional

# Import necessary libraries for LLM integration and tools
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import CodeAgent, tool, load_tool, HfApiModel
from smolagents.tools import Tool

# Import our custom agent configuration
from agent_config import create_agent, load_prompt_templates
from tools.youtube_transcriber import YouTubeTranscriberTool
from tools.web_search import WebSearchTool
from tools.rag_query import RAGQueryTool
from tools.journalistic_highlight import JournalisticHighlightTool

# Configure page settings
st.set_page_config(
    page_title="Análise de Vídeos do YouTube para Jornalistas",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎥",
    menu_items=None
)

# Apply dark theme with CSS
st.markdown("""
<style>
/* Global style */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0e1117 !important;
    color: #fafafa !important;
}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
    background-color: #262730 !important;
    color: #fafafa !important;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebarNav"] .stMarkdown {
    color: #fafafa !important;
}

/* Buttons */
.stButton > button {
    color: #4F8BF9 !important;
    background-color: #262730 !important;
    border-radius: 20px !important;
    height: 3em !important;
    width: 200px !important;
}

/* Text inputs */
.stTextInput > div > div > input {
    color: #fafafa !important;
    background-color: #262730 !important;
}

/* Input labels */
.stTextInput > label, [data-baseweb="label"] {
    color: #fafafa !important;
    font-size: 1rem !important;
}

/* Ensuring text visibility throughout the app */
.stApp > header + div, [data-testid="stAppViewContainer"] > div {
    color: #fafafa !important;
}

/* Forcing text color for specific elements */
div[class*="css"] {
    color: #fafafa !important;
}

/* Adjustments for input elements */
[data-baseweb="base-input"] {
    background-color: #262730 !important;
}
[data-baseweb="base-input"] input {
    color: #fafafa !important;
}

/* Style for the main title */
.yellow-title {
    color: yellow !important;
    font-size: 2.5rem !important;
    font-weight: bold !important;
}

/* Style for the sidebar title */
.orange-title {
    color: orange !important;
    font-size: 1.5rem !important;
    font-weight: bold !important;
}

/* Custom styles for the chat messages */
.user-message {
    background-color: #4F8BF9 !important;
    color: white !important;
    padding: 10px !important;
    border-radius: 10px !important;
    margin: 5px 0 !important;
}

.agent-message {
    background-color: #262730 !important;
    color: white !important;
    padding: 10px !important;
    border-radius: 10px !important;
    margin: 5px 0 !important;
}

/* Styling for the observation sections */
.observation-box {
    background-color: #1e1e1e !important;
    border-left: 3px solid #4F8BF9 !important;
    padding: 10px !important;
    margin: 10px 0 !important;
}

.thought-box {
    background-color: #262730 !important;
    border-left: 3px solid #FFD700 !important;
    padding: 10px !important;
    margin: 10px 0 !important;
}

.action-box {
    background-color: #262730 !important;
    border-left: 3px solid #4CAF50 !important;
    padding: 10px !important;
    margin: 10px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with guidelines
st.sidebar.markdown("<h2 class='orange-title'>Orientações</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
* Este aplicativo transcreve, resume e analisa vídeos do YouTube em Português do Brasil.
* O aplicativo irá destacar trechos de relevância para jornalistas investigativos.
* Para começar, insira as chaves de API necessárias e uma URL do YouTube.

**Obtenção de chaves de API:**
* Você pode fazer uma conta no Groq Cloud e obter uma chave de API [aqui](https://console.groq.com/login)
* Você pode fazer uma conta no Hugging Face e obter o token de API Hugging Face [aqui](https://huggingface.co/docs/hub/security-tokens)

**Atenção:** Este aplicativo utiliza IA generativa para analisar conteúdo. Recomendamos que:
1. Verifique manualmente os resultados antes de utilizá-los em reportagens
2. Considere os resultados como pontos de partida para investigação jornalística
3. Esteja ciente de que modelos de IA podem conter vieses ou imprecisões

**Sobre este app**
Este aplicativo foi desenvolvido para auxiliar jornalistas na análise de conteúdos do YouTube.
""")

st.markdown("<h1 class='yellow-title'>Análise de Vídeos do YouTube para Jornalistas 🎥🔍</h1>", unsafe_allow_html=True)
st.write("Descubra insights e pontos de interesse jornalístico em vídeos do YouTube através de análise com IA")

# Get API keys from secrets if available, otherwise request from user
groq_api_key = ""
huggingface_api_token = ""

# Check for secrets
try:
    if "groq_api_key" in st.secrets:
        groq_api_key = st.secrets["groq_api_key"]
    if "huggingface_api_token" in st.secrets:
        huggingface_api_token = st.secrets["huggingface_api_token"]
except Exception:
    pass

# If not found in secrets, request from user
if not groq_api_key:
    groq_api_key = st.text_input("Insira sua chave de API Groq:", type="password")
if not huggingface_api_token:
    huggingface_api_token = st.text_input("Insira seu token de API Hugging Face:", type="password")

# Define session state variables if they don't exist
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'highlights' not in st.session_state:
    st.session_state.highlights = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

os.remove(audio_file)
            os.rmdir(temp_dir)
            
            return f"Transcrição completa do vídeo: {yt.title}\n\n{result['text']}"
        except Exception as e:
            return f"Erro ao transcrever o vídeo: {str(e)}"
    
    #############################################################
    # TOOL 2: VIDEO SUMMARY TOOL
    #############################################################
    @tool
    def summarize_transcript(transcript: str = None) -> str:
        """Generates a summary of the video transcript.
        Args:
            transcript: The transcript to summarize. If None, uses the stored transcription.
        """
        try:
            # Use the stored transcription if transcript is None
            if transcript is None:
                if st.session_state.transcription is None:
                    return "Não há transcrição disponível. Por favor, transcreva um vídeo primeiro."
                transcript = st.session_state.transcription
            
            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.2
            )
            
            prompt = f"""
            Você é um assistente especializado em criar resumos de vídeos. Por favor, crie um resumo 
            estruturado da seguinte transcrição de um vídeo em Português do Brasil. O resumo deve:
            
            1. Ter um comprimento de aproximadamente 500 palavras
            2. Começar com uma visão geral do assunto principal do vídeo
            3. Incluir os principais pontos e argumentos apresentados
            4. Estar organizado em parágrafos bem estruturados
            5. Manter uma linguagem neutra e objetiva
            
            Transcrição:
            {transcript}
            """
            
            response = llm.invoke(prompt)
            summary = response.content
            
            # Store the summary in session state
            st.session_state.summary = summary
            
            return f"Resumo do vídeo:\n\n{summary}"
        except Exception as e:
            return f"Erro ao gerar o resumo: {str(e)}"
    
    #############################################################
    # TOOL 3: WEB SEARCH TOOL
    #############################################################
    @tool
    def search_web(query: str) -> str:
        """Searches the web for information.
        Args:
            query: The search query to find information on the web.
        """
        try:
            from duckduckgo_search import DDGS
            
            search = DDGS()
            results = search.text(query, max_results=5)
            
            if not results:
                return f"Nenhum resultado encontrado para a busca: '{query}'"
            
            formatted_results = "Resultados da busca:\n\n"
            for result in results:
                formatted_results += f"**{result['title']}**\n{result['body']}\nFonte: {result['href']}\n\n"
            
            return formatted_results
        except Exception as e:
            return f"Erro ao realizar a busca: {str(e)}"
    
    #############################################################
    # TOOL 4: JOURNALISTIC HIGHLIGHT FINDER
    #############################################################
    @tool
    def find_journalistic_highlights(context: str = None, search_results: str = None) -> str:
        """Identifies journalistically relevant highlights from the video based on current events.
        Args:
            context: The video transcript or summary. If None, uses stored data.
            search_results: Additional web search results for context. If None, performs a search.
        """
        try:
            # Use stored data if not provided
            if context is None:
                if st.session_state.summary is not None:
                    context = st.session_state.summary
                elif st.session_state.transcription is not None:
                    context = st.session_state.transcription
                else:
                    return "Não há transcrição ou resumo disponível. Por favor, transcreva e resuma um vídeo primeiro."
            
            # If no search results provided, perform a relevant search
            if search_results is None:
                # Extract keywords from context
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="deepseek-r1-distill-llama-70b",
                    temperature=0.1
                )
                
                keyword_prompt = f"""
                Extraia 3-5 palavras-chave ou frases do seguinte texto que seriam úteis 
                para pesquisar o contexto atual das notícias. Separe-as por vírgulas.
                
                Texto:
                {context[:2000]}  # Limiting to first 2000 chars
                
                Palavras-chave:
                """
                
                keyword_response = llm.invoke(keyword_prompt)
                keywords = keyword_response.content.strip()
                
                # Search for current news context
                search_query = f"{keywords} notícias atuais Brasil"
                search_results = search_web(search_query)
            
            # Initialize LLM for highlight analysis
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.3
            )
            
            prompt = f"""
            Você é um jornalista investigativo experiente. Analise o texto a seguir, que é a transcrição
            ou resumo de um vídeo em Português do Brasil, e destaque trechos que merecem investigação
            jornalística adicional. Considere:
            
            1. Afirmações que podem ser verificadas factualmente
            2. Conexões com notícias ou eventos atuais
            3. Declarações controversas ou potencialmente enganosas
            4. Implicações para políticas públicas ou interesse social
            5. Informações que parecem novas ou pouco divulgadas
            
            Formate sua resposta como:
            
            # Pontos de Interesse Jornalístico
            
            ## Destaque 1: [Título breve]
            **Trecho relevante:** [Trecho exato do texto]
            **Por que investigar:** [Explicação sobre o valor jornalístico]
            **Sugestão de abordagem:** [Como um jornalista poderia verificar ou explorar este ponto]
            
            [Repita o formato para cada destaque, com no mínimo 3 e no máximo 5 destaques]
            
            Texto a analisar:
            {context}
            
            Contexto atual (resultados de busca na web):
            {search_results}
            """
            
            response = llm.invoke(prompt)
            highlights = response.content
            
            # Store highlights in session state
            st.session_state.highlights = highlights
            
            return highlights
        except Exception as e:
            return f"Erro ao encontrar destaques jornalísticos: {str(e)}"
    
    #############################################################
    # TOOL 5: RAG QUERY TOOL
    #############################################################
    @tool
    def query_transcript(question: str) -> str:
        """Answers a question about the video content using RAG.
        Args:
            question: The question to answer about the video content
        """
        try:
            if st.session_state.vectorstore is None:
                return "Não há transcrição indexada disponível. Por favor, transcreva um vídeo primeiro."
            
            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.2
            )
            
            # Create retriever from vectorstore
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # Retrieve relevant contexts
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response with LLM
            prompt = f"""
            Responda à seguinte pergunta com base no contexto fornecido da transcrição de um vídeo.
            Se a informação não estiver disponível no contexto, responda honestamente que não consegue
            encontrar essa informação na transcrição do vídeo.
            
            Contexto da transcrição:
            {context}
            
            Pergunta: {question}
            
            Resposta:
            """
            
            response = llm.invoke(prompt)
            
            return f"Resposta baseada na transcrição do vídeo:\n\n{response.content}"
        except Exception as e:
            return f"Erro ao responder à pergunta: {str(e)}"
    
    #############################################################
    # TOOL 6: FINAL ANSWER TOOL
    #############################################################
    @tool
    def final_answer(answer: Any) -> Any:
        """Provides a final answer to the given problem.
        Args:
            answer: The final answer to the problem
        """
        return answer
    
    # Initialize prompt templates
    prompt_templates = {
        "system_prompt": """
        Você é um agente de IA especializado em analisar vídeos do YouTube para fins jornalísticos.
        Seu objetivo é ajudar jornalistas a extrair informações valiosas de vídeos em Português do Brasil.
        
        Para resolver as tarefas, você deve seguir um ciclo de 'Thought:', 'Code:', e 'Observation:'.
        
        Em cada etapa, você deve:
        1. Explicar seu raciocínio no bloco 'Thought:'
        2. Escrever código Python simples no bloco 'Code:' (encerrado com '<end_code>')
        3. Observar os resultados no bloco 'Observation:'
        
        Suas respostas devem ser claras, estruturadas e úteis para jornalistas investigativos.
        No final, você deve fornecer uma resposta completa usando a ferramenta `final_answer`.
        
        Você tem acesso às seguintes ferramentas:
        - transcribe_youtube_video: Transcreve um vídeo do YouTube
        - summarize_transcript: Resume uma transcrição
        - search_web: Busca informações na web
        - find_journalistic_highlights: Identifica pontos de interesse jornalístico
        - query_transcript: Responde perguntas sobre o conteúdo do vídeo
        - final_answer: Fornece uma resposta final
        
        Lembre-se de que os jornalistas precisam de informações precisas, contextualizadas e com
        valor noticioso. Sempre indique quando uma informação precisa ser verificada ou investigada
        mais a fundo.
        """
    }
    
    # Initialize model using custom configuration
    model = HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id="deepseek-r1-distill-llama-70b",
        custom_role_conversions=None
    )
    
    # Create agent with tools
    agent = CodeAgent(
        model=model,
        tools=[
            transcribe_youtube_video,
            summarize_transcript,
            search_web,
            find_journalistic_highlights,
            query_transcript,
            final_answer
        ],
        max_steps=8,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates
    )
    
    return agent

def display_conversation_history():
    """Display the conversation history with appropriate styling"""
    for i, (role, content) in enumerate(st.session_state.conversation_history):
        if role == "user":
            st.markdown(f"<div class='user-message'><strong>Você:</strong> {content}</div>", unsafe_allow_html=True)
        else:
            # Content might contain Thought/Action/Observation sections
            if "Thought:" in content and "Code:" in content:
                parts = content.split("Thought:")
                intro = parts[0] if parts[0].strip() else ""
                if intro:
                    st.markdown(f"<div class='agent-message'><strong>Assistente:</strong> {intro}</div>", unsafe_allow_html=True)
                
                for part in parts[1:]:
                    if part.strip():
                        thought_parts = part.split("Code:")
                        thought = thought_parts[0] if thought_parts[0].strip() else ""
                        
                        if thought:
                            st.markdown(f"<div class='thought-box'><strong>💭 Pensamento:</strong> {thought}</div>", unsafe_allow_html=True)
                        
                        if len(thought_parts) > 1:
                            code_parts = thought_parts[1].split("Observation:")
                            code = code_parts[0] if code_parts[0].strip() else ""
                            
                            if code:
                                # Format the code nicely
                                if "<end_code>" in code:
                                    code = code.replace("<end_code>", "")
                                st.markdown(f"<div class='action-box'><strong>⚙️ Ação:</strong></div>", unsafe_allow_html=True)
                                st.code(code.strip(), language="python")
                            
                            if len(code_parts) > 1 and code_parts[1].strip():
                                observation = code_parts[1].strip()
                                st.markdown(f"<div class='observation-box'><strong>👁️ Observação:</strong> {observation}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='agent-message'><strong>Assistente:</strong> {content}</div>", unsafe_allow_html=True)

# Main Application Logic
if groq_api_key and huggingface_api_token:
    # Set the Hugging Face API token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token
    
    # YouTube URL input
    youtube_url = st.text_input("Insira a URL do vídeo do YouTube:")
    
    # Create a container for the interface
    main_container = st.container()
    
    with main_container:
        # Set up tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Análise Completa", "Transcrição", "Resumo", "Destaques Jornalísticos"])
        
        with tab1:
            st.subheader("Análise de Vídeo")
            
            # Initialize or retrieve agent
            if 'agent' not in st.session_state:
                with st.spinner("Inicializando o agente de IA..."):
                    try:
                        st.session_state.agent = create_agent(groq_api_key, huggingface_api_token)
                        st.success("Agente inicializado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao inicializar o agente: {str(e)}")
                        st.error(f"Detalhes: {str(e)}")
            
            if youtube_url and 'agent' in st.session_state:
                # Button to analyze video
                if st.button("Analisar Vídeo"):
                    with st.spinner("Analisando o vídeo..."):
                        try:
                            # Process the video with the agent
                            results = process_video(youtube_url, st.session_state.agent)
                            
                            # Add the results to conversation history
                            task = f"Analisar o vídeo do YouTube: {youtube_url}"
                            st.session_state.conversation_history.append(("user", task))
                            
                            # Add each step to the conversation history
                            for step in results["steps"]:
                                st.session_state.conversation_history.append(
                                    ("assistant", f"**{step['name']}:**\n\n{step['content']}")
                                )
                            
                            # Add final completion message
                            st.session_state.conversation_history.append(
                                ("assistant", "✅ Análise completa! Você pode visualizar a transcrição, o resumo e os destaques jornalísticos nas abas correspondentes acima.")
                            )
                            
                            # Ensure we store the results in session state for other tabs
                            if results["transcription"]:
                                st.session_state.transcription = results["transcription"]
                            if results["summary"]:
                                st.session_state.summary = results["summary"]
                            if results["highlights"]:
                                st.session_state.highlights = results["highlights"]
                                
                        except Exception as e:
                            import traceback
                            error_message = f"Erro durante a análise: {str(e)}\n\n{traceback.format_exc()}"
                            st.session_state.conversation_history.append(("assistant", error_message))
                            st.error(error_message)
            
            # Text input for questions about the video
            st.subheader("Faça perguntas sobre o vídeo")
            question = st.text_input("Sua pergunta sobre o conteúdo do vídeo:")
            
            if question and st.button("Enviar Pergunta") and 'agent' in st.session_state:
                with st.spinner("Processando sua pergunta..."):
                    try:
                        # Add user question to history
                        st.session_state.conversation_history.append(("user", question))
                        
                        # Create a RAG-based query to answer the question
                        if st.session_state.vectorstore is not None:
                            # Use our agent to get the answer
                            task = f"Responda a seguinte pergunta sobre o vídeo que foi transcrito: {question}"
                            response = st.session_state.agent.run(task)
                            
                            # Add assistant response to history
                            st.session_state.conversation_history.append(("assistant", response))
                        else:
                            error_message = "Não há transcrição indexada disponível. Por favor, analise um vídeo primeiro."
                            st.session_state.conversation_history.append(("assistant", error_message))
                            st.warning(error_message)
                    except Exception as e:
                        error_message = f"Erro ao processar a pergunta: {str(e)}"
                        st.session_state.conversation_history.append(("assistant", error_message))
                        st.error(error_message)
            
            # Display conversation history
            st.subheader("Histórico da Conversa")
            display_conversation_history()
        
        with tab2:
            st.subheader("Transcrição do Vídeo")
            if st.session_state.transcription:
                st.text_area("Texto completo da transcrição:", st.session_state.transcription, height=400)
                
                # Button to download transcription
                if st.button("Download da Transcrição (TXT)"):
                    transcription_bytes = st.session_state.transcription.encode()
                    st.download_button(
                        label="Baixar Transcrição",
                        data=transcription_bytes,
                        file_name="transcricao_video.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Nenhuma transcrição disponível. Analise um vídeo primeiro.")
        
        with tab3:
            st.subheader("Resumo do Vídeo")
            if st.session_state.summary:
                st.markdown(st.session_state.summary)
                
                # Button to download summary
                if st.button("Download do Resumo (TXT)"):
                    summary_bytes = st.session_state.summary.encode()
                    st.download_button(
                        label="Baixar Resumo",
                        data=summary_bytes,
                        file_name="resumo_video.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Nenhum resumo disponível. Analise um vídeo primeiro.")
        
        with tab4:
            st.subheader("Destaques Jornalísticos")
            if st.session_state.highlights:
                st.markdown(st.session_state.highlights)
                
                # Button to download highlights
                if st.button("Download dos Destaques (MD)"):
                    highlights_bytes = st.session_state.highlights.encode()
                    st.download_button(
                        label="Baixar Destaques",
                        data=highlights_bytes,
                        file_name="destaques_jornalisticos.md",
                        mime="text/markdown"
                    )
            else:
                st.info("Nenhum destaque jornalístico disponível. Analise um vídeo primeiro.")

else:
    st.warning("Por favor, insira as chaves de API do Groq e do Hugging Face para começar.")

if __name__ == "__main__":
    # Set up configuration for Whisper and other dependencies
    try:
        # Ensure required packages are installed
        import pytube
        import whisper
        import duckduckgo_search
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Create temporary directory for file operations if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Create a directory for storing logs
        os.makedirs("logs", exist_ok=True)
        
        # Configure basic logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/app.log"),
                logging.StreamHandler()
            ]
        )
        
        # Log startup information
        logging.info("YouTube Analysis Agent started")
        logging.info("Dependencies loaded successfully")
        
    except ImportError as e:
        st.error(f"Dependências necessárias não encontradas: {str(e)}")
        st.error("Por favor, instale as bibliotecas necessárias usando: pip install -r requirements.txt")
