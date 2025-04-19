import os
import io
import tempfile
import logging
import torch
import whisper
import time
from flask import Flask, request, send_file, jsonify
from concurrent.futures import ThreadPoolExecutor
from scipy.io.wavfile import write as wav_write
import ChatTTS
import dotenv

# Load environment variables
dotenv.load_dotenv()
# Set the environment variable to restrict to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Automatically select GPU if available, otherwise fallback to CPU
device = "cpu"
logging.info(f"Using device: {device}")

# Load Whisper model (using medium model for better performance)
logging.info("Loading Whisper model (medium)")
whisper_model = whisper.load_model("small.en").to(device)

# Initialize ChatTTS
logging.info("Initializing ChatTTS")
chat = ChatTTS.Chat()
chat.load(compile=True)

# Load speaker embedding
embedding_path = "./seed_1528.pt"
spk = torch.load(embedding_path, map_location=torch.device(device))

# ChatTTS inference parameters
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=spk,
    temperature=0.3,
    top_P=0.7,
    top_K=20
)

# Thread pool executor for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Generate speech audio from text (without caching)
def generate_audio_from_text(text: str):
    logging.info(f"Generating audio from text: {text}")

    # Generate the audio
    wavs = chat.infer([text], params_infer_code=params_infer_code)
    buffer = io.BytesIO()
    wav_write(buffer, 24000, (wavs[0] * 32767).astype("int16"))
    buffer.seek(0)
    return buffer

# TTS endpoint to handle speech synthesis
@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return "No text provided", 400

    # Execute TTS in background using the executor to avoid blocking the main thread
    future = executor.submit(generate_audio_from_text, text)
    audio_buffer = future.result()

    return send_file(audio_buffer, mimetype="audio/wav", as_attachment=True, download_name="output_audio.wav")

# STT endpoint to transcribe audio to text
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_filename = temp_file.name

    logging.info("Transcribing audio...")
    result = whisper_model.transcribe(temp_filename, fp16=False, language='en')
    os.unlink(temp_filename)

    return jsonify({"text": result.get("text", "").strip()})

# Run the app (preferably use Gunicorn in production)
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9960)), use_reloader=False)
