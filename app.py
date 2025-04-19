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

# Load higher-quality Whisper model
logging.info("Loading Whisper model (medium)")
whisper_model = whisper.load_model("medium.en").to(device)

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

# Generate speech audio from text
def generate_audio_from_text(text: str):
    logging.info(f"Generating audio from text: {text}")
    wavs = chat.infer([text], params_infer_code=params_infer_code)
    buffer = io.BytesIO()
    wav_write(buffer, 24000, (wavs[0] * 32767).astype("int16"))
    buffer.seek(0)
    return buffer

# TTS endpoint
@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return "No text provided", 400

    future = executor.submit(generate_audio_from_text, text)
    audio_buffer = future.result()

    return send_file(audio_buffer, mimetype="audio/wav", as_attachment=True, download_name="output_audio.wav")

# STT endpoint
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

# Optional background TTS function (internal usage)
def text_to_speech(message):
    logging.info(f"Converting message to speech: {message}")
    
    def play_speech():
        try:
            time.sleep(0.5)
            wavs = chat.infer([message], params_infer_code=params_infer_code)
            audio_buffer = io.BytesIO()
            wav_write(audio_buffer, 24000, (wavs[0] * 32767).astype("int16"))
            audio_buffer.seek(0)
            logging.info("Speech playback completed")
            return audio_buffer
        except Exception as e:
            logging.error(f"An error occurred during speech playback: {str(e)}")

    thread = threading.Thread(target=play_speech)
    thread.start()

# Run development server
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9960)), use_reloader=False)
