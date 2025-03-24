# Novo YouTubeTranscriberTool usando Whisper API (OpenAI) e yt-dlp
import requests
import tempfile
import os
import subprocess
from smolagents.tools import Tool

class YouTubeTranscriberTool(Tool):
    name = "youtube_transcriber"
    description = "Transcribes a YouTube video using OpenAI Whisper API."
    inputs = {
        'url': {'type': 'string', 'description': 'The YouTube video URL'},
        'openai_api_key': {'type': 'string', 'description': 'OpenAI API Key'}
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_initialized = True

    def forward(self, url: str, openai_api_key: str) -> str:
        try:
            temp_dir = tempfile.mkdtemp()
            mp3_path = os.path.join(temp_dir, "audio.mp3")

            # Baixar e converter com yt-dlp
            command = [
                "yt-dlp",
                "-x", "--audio-format", "mp3",
                "--output", mp3_path,
                url
            ]
            subprocess.run(command, check=True)

            with open(mp3_path, "rb") as f:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {openai_api_key}"},
                    files={"file": f},
                    data={"model": "whisper-1", "language": "pt"}
                )

            os.remove(mp3_path)
            os.rmdir(temp_dir)

            if response.status_code == 200:
                result = response.json()
                return f"Transcrição do vídeo:

{result['text']}"
            else:
                return f"Erro na transcrição com Whisper API: {response.text}"

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return f"Erro ao transcrever com Whisper API: {str(e)}\n\n{traceback_str}"
