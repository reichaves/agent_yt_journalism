# 🧠 Agent YouTube Journalism

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentytjournalism.streamlit.app/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🔍 Project overview

**Agent YouTube Journalism** is an open-source investigative AI assistant that transcribes, summarizes, analyzes, and answers questions about videos from YouTube — especially those related to Brazilian politics and public interest.

The system uses multi-agent reasoning and Retrieval-Augmented Generation (RAG) to:

- Transcribe YouTube videos in Brazilian Portuguese (using OpenAI Whisper API)
- Summarize the transcript with DeepSeek via Groq Cloud
- Search the web for context (via DuckDuckGo)
- Highlight journalistically relevant parts of the video
- Index and answer questions based on the transcript + general knowledge if needed

🟢 **Try it online**: https://agentytjournalism.streamlit.app/  
⚠️ You must provide your own API keys for Groq, OpenAI, and Hugging Face.

---

## ⚙️ How it works

1. **User enters YouTube video URL + API keys**
2. The app:
   - Downloads audio via `yt-dlp`
   - Transcribes using Whisper (`openai.Audio.transcribe`)
   - Summarizes with Groq LLM (DeepSeek)
   - Searches the web for context
   - Highlights journalistic investigation leads
   - Indexes the transcript with FAISS
3. The user can:
   - View the analysis
   - Ask questions based on the video (with RAG + LLM knowledge fallback)

The system uses `smolagents` to structure the reasoning with a clear cycle:

> **Thought → Code → Observation**

---

## 🗂️ Project structure

### 🔹 Main app

- `app.py`: Main Streamlit app with two tabs: Analysis & Questions
- `process_video.py`: Orchestrates full pipeline (transcription → summary → highlight → indexing)
- `rag_question_tab.py`: Handles the RAG-based Q&A flow with session state
- `agent_config.py`: Defines tools and setup for `smolagents` agent
- `streamlit_app.yaml`: Config file for deployment (Streamlit Community Cloud)
- `prompts.yaml`: Prompt templates used for summarization, analysis, and code reasoning

### 🔹 Groq integration

- `groq_model.py`: Executes prompts with Groq LLM and truncates long prompts when needed
- `list_groq_models.py`: Lists all Groq-hosted models available for querying

### 🔹 Tools (used by agents)

- `tools/youtube_transcriber.py`: Downloads and transcribes video audio via Whisper API
- `tools/summarization.py`: Summarizes the transcript using DeepSeek
- `tools/web_search.py`: Searches DuckDuckGo for current context
- `tools/journalistic_highlight.py`: Generates public interest highlights
- `tools/index_transcript.py`: Splits transcript and indexes it with FAISS
- `tools/rag_query.py`: Performs RAG query and allows fallback to general LLM knowledge
- `tools/__init__.py`: Makes the tools importable as a module

### 🔹 Requirements

- `requirements.txt`: All Python dependencies (tested with Python 3.12)
- `packages.txt`: System dependencies (e.g., ffmpeg, build tools)

---

## 📈 Improvements & recommendations

Here are potential enhancements for the project:

### ✅ Already solved

- Fixed repeated video download when switching tabs
- Added session state to persist transcript/vectorstore between tabs
- Enabled mixed-source RAG answers (video + general knowledge)
- Adjusted `requirements.txt` and `packages.txt` for compatibility

### 🚧 Future improvements

1. **Caching**
   - Save FAISS vectorstore, summary, highlights to disk (`.save_local()`) or use `st.cache_data()`

2. **Better Streamlit UX**
   - Enable chat-style Q&A with memory
   - Show progress for each processing step
   - Add "Download PDF" report button

3. **Model prompting**
   - Add clear tags like `[FACT FROM VIDEO]` vs `[LLM KNOWLEDGE]`

4. **Testing**
   - Add unit/integration tests using `pytest`

5. **Agent orchestration**
   - Split into two agents: `VideoAnalysisAgent` and `QAAgent`
   - Optionally adopt CrewAI or LangGraph for more complex flows

---

## 📄 License

MIT License. See `LICENSE` file.

---

## 🙌 Credits

Developed by Reinaldo Chaves (@reichaves) — journalist, data scientist, and investigative technologist.  

Abraji (Brazilian Association of Investigative Journalism) experimental project.

---

## 💬 Questions or feedback?

Open an [issue](https://github.com/reichaves/agent_yt_journalism/issues) or contact via GitHub.
