import os
import tempfile
import logging
import torch
import torchaudio
import whisper
import asyncio
import time
import threading
from flask import Flask, request, send_file, jsonify
import io
import ChatTTS

# Set the environment variable to restrict to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize Whisper model for transcription (force CPU)
device = "cpu"  # Explicitly use CPU
whisper_model = whisper.load_model("medium").to(device)

# Initialize ChatTTS for text-to-speech (force CPU)
logging.info("Initializing ChatTTS")
chat = ChatTTS.Chat()
chat.load(compile=True)

# Path to the downloaded .pt embedding file
embedding_path = "./seed_1528.pt"
# Load the speaker embedding (use CPU)
spk = torch.load(embedding_path, map_location=torch.device('cpu'))

# Define the inference parameters for ChatTTS
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=spk,
    temperature=0.3,
    top_P=0.7,
    top_K=20
)

# Function to generate audio from text using ChatTTS
async def generate_audio_from_text(text: str):
    logging.info(f"Generating audio from text: {text}")
    # Perform inference with the provided text
    wavs = chat.infer([text], params_infer_code=params_infer_code)

    # Save the generated audio in an in-memory buffer
    buffer = io.BytesIO()
    for i, wav in enumerate(wavs):
        torchaudio.save(buffer, torch.from_numpy(wav).unsqueeze(0), 24000)
    
    buffer.seek(0)  # Reset buffer pointer to the beginning
    return buffer

# Endpoint to receive text and return the generated audio (Text-to-Speech)
@app.route("/speak", methods=["POST"])
async def speak():
    # Receive the text from the request body
    data = await request.get_json()
    text = data.get("text", "")
    
    if not text:
        return "No text provided", 400  # Return error if no text is provided
    
    logging.info("Received text for TTS: %s", text)
    
    # Generate audio from the received text
    audio_buffer = await generate_audio_from_text(text)
    
    # Return the generated audio as a file
    return send_file(audio_buffer, mimetype="audio/wav", as_attachment=True, download_name="output_audio.wav")

# Endpoint to receive audio and return the transcribed text (Speech-to-Text)
@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Check if the audio file is part of the request
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    # Save the uploaded audio file to a temporary location
    audio_file = request.files["audio"]
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    with open(temp_wav, "wb") as f:
        f.write(audio_file.read())
    os.close(fd)
    
    logging.info("Transcribing...")
    
    # Transcribe the audio file using Whisper
    result = whisper_model.transcribe(temp_wav)
    
    # Remove the temporary audio file
    os.unlink(temp_wav)
    
    # Return the transcribed text as a JSON response
    return jsonify({"text": result.get("text", "").strip()})

# Function to simulate the behavior of TTS (called within a separate thread)
def text_to_speech(message):
    logging.info(f"Converting message to speech: {message}")
    print('\nTTS:\n', message.strip())

    # Define the speech function that uses the initialized TTS engine
    def play_speech():
        try:
            logging.info("Converting message to speech")
            # Add a short delay before converting message to speech
            time.sleep(0.5)  # Adjust the delay as needed
            
            # Generate speech using ChatTTS
            wavs = chat.infer([message], params_infer_code=params_infer_code)
            audio_buffer = io.BytesIO()
            for i, wav in enumerate(wavs):
                torchaudio.save(audio_buffer, torch.from_numpy(wav).unsqueeze(0), 24000)
            audio_buffer.seek(0)
            
            # Return the audio as an in-memory file
            logging.info("Speech playback completed")
            return audio_buffer
        except Exception as e:
            logging.error(f"An error occurred during speech playback: {str(e)}")

    # Start the speech in a separate thread
    speech_thread = threading.Thread(target=play_speech)
    speech_thread.start()

if __name__ == "__main__":
    # Run the Flask app on all available IP addresses (0.0.0.0) and port 5000
    app.run(debug=True, host="0.0.0.0", port=os.environ('PORT'), use_reloader=False)
