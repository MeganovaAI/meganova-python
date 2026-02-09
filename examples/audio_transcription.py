import os
import sys
from dotenv import load_dotenv
from meganova import MegaNova

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    print("Please set MEGANOVA_API_KEY in your .env file.")
    exit(1)

client = MegaNova(api_key=api_key)

# --- Transcribe an Audio File ---
print("\n--- Audio Transcription ---")

# Accept a file path from the command line, or use a default
audio_path = sys.argv[1] if len(sys.argv) > 1 else "recording.mp3"

if not os.path.exists(audio_path):
    print(f"Audio file not found: {audio_path}")
    print("Usage: python audio_transcription.py <path_to_audio_file>")
    exit(1)

file_size = os.path.getsize(audio_path)
print(f"File: {audio_path} ({file_size:,} bytes)")
print("Transcribing...")

try:
    # Transcribe from a file path
    result = client.audio.transcribe(audio_path)
    print(f"\nTranscription:\n{result.text}")

except Exception as e:
    print(f"Error: {e}")
