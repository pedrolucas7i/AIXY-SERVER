import os
import tempfile
import logging
import torch
import torchaudio
import whisper
import time
import threading
from flask import Flask, request, send_file, jsonify
import io
import ChatTTS
import dotenv

dotenv.load_dotenv()

# Set the environment variable to restrict to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize Whisper model for transcription (force CPU)
device = "cpu"
whisper_model = whisper.load_model("medium").to(device)

# Initialize ChatTTS for text-to-speech (force CPU)
logging.info("Initializing ChatTTS")
chat = ChatTTS.Chat()
chat.load(compile=True)

# Load the speaker embedding (use CPU)
embedding_path = "./seed_1528.pt"
spk = torch.load(embedding_path, map_location=torch.device('cpu'))

# Define the inference parameters for ChatTTS
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=spk,
    temperature=0.3,
    top_P=0.7,
    top_K=20
)

# Function to generate audio from text using ChatTTS (sync version)
def generate_audio_from_text(text: str):
    logging.info(f"Generating audio from text: {text}")

    wavs = chat.infer([text], params_infer_code=params_infer_code)

    buffer = io.BytesIO()
    for i, wav in enumerate(wavs):
        torchaudio.save(buffer, torch.from_numpy(wav).unsqueeze(0), 24000, format="wav")

    buffer.seek(0)
    return buffer

# Endpoint to receive text and return the generated audio (Text-to-Speech)
@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return "No text provided", 400

    logging.info("Received text for TTS: %s", text)

    audio_buffer = generate_audio_from_text(text)

    return send_file(audio_buffer, mimetype="audio/wav", as_attachment=True, download_name="output_audio.wav")

# Endpoint to receive audio and return the transcribed text (Speech-to-Text)
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    with open(temp_wav, "wb") as f:
        f.write(audio_file.read())
    os.close(fd)

    logging.info("Transcribing...")

    result = whisper_model.transcribe(temp_wav)
    os.unlink(temp_wav)

    return jsonify({"text": result.get("text", "").strip()})

# Optional thread-based TTS function (for other internal uses)
def text_to_speech(message):
    logging.info(f"Converting message to speech: {message}")
    print('\nTTS:\n', message.strip())

    def play_speech():
        try:
            logging.info("Converting message to speech")
            time.sleep(0.5)
            wavs = chat.infer([message], params_infer_code=params_infer_code)
            audio_buffer = io.BytesIO()
            for i, wav in enumerate(wavs):
                torchaudio.save(audio_buffer, torch.from_numpy(wav).unsqueeze(0), 24000)
            audio_buffer.seek(0)
            logging.info("Speech playback completed")
            return audio_buffer
        except Exception as e:
            logging.error(f"An error occurred during speech playback: {str(e)}")

    speech_thread = threading.Thread(target=play_speech)
    speech_thread.start()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.environ.get('PORT', 9960), use_reloader=False)
