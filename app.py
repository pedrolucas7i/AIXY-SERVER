"""
===============================================================================================================================================================
===============================================================================================================================================================

                                                                   _      ___  __  __ __   __  ____         ___  
                                                                  / \    |_ _| \ \/ / \ \ / / |___ \       / _ \ 
                                                                 / _ \    | |   \  /   \ V /    __) |     | | | |
                                                                / ___ \   | |   /  \    | |    / __/   _  | |_| |
                                                               /_/   \_\ |___| /_/\_\   |_|   |_____| (_)  \___/ 

                                                               
                                                                            SERVER  TTS/STT   CODE
                                                                            by Pedro Ribeiro Lucas
                                                                                                                  
===============================================================================================================================================================
===============================================================================================================================================================
"""

import os
import io
import tempfile
import logging
import whisper
import torch
from flask import Flask, request, jsonify, send_file
from scipy.io.wavfile import write as wav_write
from TTS.api import TTS

# ================== CPU THREAD OPTIMIZATION ==================
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
os.environ["OMP_NUM_THREADS"] = "28"
os.environ["OPENBLAS_NUM_THREADS"] = "28"
os.environ["MKL_NUM_THREADS"] = "28"
os.environ["VECLIB_MAXIMUM_THREADS"] = "28"
os.environ["NUMEXPR_NUM_THREADS"] = "28"
torch.set_num_threads(28)
# =============================================================

# Load .env if available
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# =============================

# ========== FLASK APP ==========
app = Flask(__name__)
device = "cpu"
logging.info(f"Using device: {device}")
# ===============================

# ========== LOAD MODELS ==========
logging.info("Loading Whisper model (medium)")
model = whisper.load_model("medium").to(device)

logging.info("Loading Coqui TTS model (English)")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

logging.info("Loading Coqui TTS model (Portuguese)")
tts_model_pt = TTS(model_name="tts_models/pt/cv/vits", progress_bar=False)
# =================================

# ========== TRANSCRIBE (STT) ==========
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_filename = temp_file.name

    logging.info("Transcribing audio...")
    try:
        result = model.transcribe(temp_filename, fp16=False, language='en')
        text = result.get("text", "").strip()
    finally:
        os.unlink(temp_filename)

    return jsonify({"text": text})
# ======================================

# ========== TTS (ENGLISH) ==========
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.close()
        tts_model.tts_to_file(text=text, file_path=temp_wav.name)
        wav_path = temp_wav.name

    logging.info("Synthesized speech for English text.")
    response = send_file(wav_path, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    os.unlink(wav_path)
    return response
# ===================================

# ========== TTS (PORTUGUESE) ==========
@app.route("/tts-pt", methods=["POST"])
def tts_pt():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav.close()
        tts_model_pt.tts_to_file(text=text, file_path=temp_wav.name)
        wav_path = temp_wav.name

    logging.info("Synthesized speech for Portuguese text.")
    response = send_file(wav_path, mimetype="audio/wav", as_attachment=True, download_name="output_pt.wav")
    os.unlink(wav_path)
    return response
# =======================================

# ========== START SERVER ==========
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9960)), use_reloader=False)
# ==================================
