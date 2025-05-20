import os
import io
import tempfile
import logging
import whisper
from flask import Flask, request, jsonify
from scipy.io.wavfile import write as wav_write
from flask_cors import CORS

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
CORS(app, origins=["*"])

# Set device to CPU
device = "cpu"
logging.info(f"Using device: {device}")

# Load Whisper model
logging.info("Loading Whisper model (medium)")
model = whisper.load_model("medium").to(device)

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

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9960)), use_reloader=False, ssl_context=("cert.pem", "key.pem"))
