import os
import io
import tempfile
import logging
import whisper
from flask import Flask, request, jsonify, send_file
from scipy.io.wavfile import write as wav_write
from TTS.api import TTS

# Load environment variables
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Force the use of CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Set device to CPU
device = "cpu"
logging.info(f"Using device: {device}")

# Load Whisper model
logging.info("Loading Whisper model (medium)")
model = whisper.load_model("medium").to(device)

# Initialize Coqui TTS model
logging.info("Loading Coqui TTS model")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# Initialize Coqui TTS model for Portuguese
logging.info("Loading Coqui TTS model for Portuguese")
tts_model_pt = TTS(model_name="tts_models/pt/cv/vits", progress_bar=False)

# Transcription endpoint (STT)
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_filename = temp_file.name

    logging.info("Transcribing audio...")
    result = model.transcribe(temp_filename, fp16=False, language='en')
    os.unlink(temp_filename)

    return jsonify({"text": result.get("text", "").strip()})

# TTS endpoint
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.close()  # Close so TTS can write to it
        tts_model.tts_to_file(text=text, file_path=temp_wav.name)
        wav_path = temp_wav.name

    logging.info("Synthesized speech for text.")
    response = send_file(wav_path, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    os.unlink(wav_path)
    return response

# TTS endpoint for Portuguese
@app.route("/tts-pt", methods=["POST"])
def tts_pt():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.close()  # Close so TTS can write to it
        tts_model_pt.tts_to_file(text=text, file_path=temp_wav.name)
        wav_path = temp_wav.name

    logging.info("Synthesized Portuguese speech for text.")
    response = send_file(wav_path, mimetype="audio/wav", as_attachment=True, download_name="output_pt.wav")
    os.unlink(wav_path)
    return response

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9960)), use_reloader=False)
