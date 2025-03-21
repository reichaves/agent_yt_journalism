from typing import Optional, Any
import os
import tempfile
from smolagents.tools import Tool

class YouTubeTranscriberTool(Tool):
    name = "youtube_transcriber"
    description = "Transcribes a YouTube video and returns the text."
    inputs = {'url': {'type': 'string', 'description': 'The YouTube video URL'}}
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_initialized = True

    def forward(self, url: str) -> str:
        """Downloads and transcribes a YouTube video using Whisper"""
        try:
            # Import necessary libraries
            from pytube import YouTube
            import whisper
            import logging
            
            logging.info(f"Transcribing YouTube video: {url}")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            logging.info(f"Created temporary directory: {temp_dir}")
            
            # Download audio from YouTube
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_file = audio_stream.download(output_path=temp_dir)
            logging.info(f"Downloaded audio to: {audio_file}")
            
            # Load Whisper model and transcribe
            model = whisper.load_model("base")
            logging.info("Loaded Whisper model, starting transcription...")
            
            result = model.transcribe(audio_file, language="pt")
            logging.info("Transcription complete")
            
            # Clean up
            os.remove(audio_file)
            os.rmdir(temp_dir)
            logging.info("Cleaned up temporary files")
            
            return f"Transcrição completa do vídeo: {yt.title}\n\n{result['text']}"
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return f"Erro ao transcrever o vídeo: {str(e)}\n\nTraceback:\n{traceback_str}"
