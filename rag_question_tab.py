import streamlit as st
from groq import Client

def render_rag_tab():
    st.header("📌 Perguntas sobre o vídeo")
    question = st.text_input("Digite sua pergunta sobre o conteúdo transcrito:")
    api_key = st.session_state.get("groq_api_key", "")
    transcript = st.session_state.get("transcript", "")
    vectorstore = st.session_state.get("vectorstore", None)

    if not api_key:
        st.warning("Por favor, insira sua chave de API do Groq na aba 'Configurações'.")
        return

    if not transcript:
        st.warning("Transcrição não encontrada. Transcreva e indexe um vídeo primeiro.")
        return

    if st.button("Responder"):
        with st.spinner("Consultando..."):
            try:
                response = ask_question(question, transcript, api_key)
                st.markdown("### Resposta")
                st.write(response)
            except Exception as e:
                st.error(f"Erro ao processar a pergunta: {str(e)}")

def ask_question(question: str, transcript: str, groq_api_key: str) -> str:
    client = Client(api_key=groq_api_key)

    system_prompt = (
        "Você é um assistente especializado em vídeos que responde perguntas com base "
        "em transcrições. Responda apenas com informações encontradas no contexto fornecido. "
        "Se a resposta não estiver no contexto, diga que não é possível responder com base na transcrição."
    )

    user_prompt = f"""
    Responda à seguinte pergunta com base no conteúdo abaixo:

    Transcrição do vídeo:
    {transcript}

    Pergunta: {question}
    """

    chat_response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=2048
    )

    return chat_response.choices[0].message.content
