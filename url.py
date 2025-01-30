import numpy as np
import requests
from scipy.io.wavfile import write
import tempfile
import os
from faster_whisper import WhisperModel

class WhisperTranscriberFromURL:
    def __init__(self, model_size="large-v2", sample_rate=44100):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.sample_rate = sample_rate

    def download_audio(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with open(temp_file.name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded audio to {temp_file.name}")
            return temp_file.name
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")

    def transcribe_audio(self, file_path):
        segments, info = self.model.transcribe(file_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        full_transcription = ""
        for segment in segments:
            print(segment.text)
            full_transcription += segment.text + " "
        os.remove(file_path)
        return full_transcription

    def run(self, url):
        try:
            print("Downloading audio from URL...")
            file_path = self.download_audio(url)
            print("Transcribing the downloaded audio...")
            transcription = self.transcribe_audio(file_path)
            print("Transcription completed.")
            return transcription
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    url = input("Enter the URL of the audio file: ")
    transcriber = WhisperTranscriberFromURL()
    transcription = transcriber.run(url)
    print(f"Transcription: {transcription}")
