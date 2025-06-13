from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import soundfile as sf
import tempfile

app = Flask(__name__)

@app.route('/')
def index():
    return '🎵 Downbeat Detection API is running.'

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            file.save(temp_audio.name)
            y, sr = librosa.load(temp_audio.name, sr=None)

        # --- STEP 1: Tempo and Beats ---
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # --- STEP 2: Estimate Downbeats ---
        # Assume 4/4 and take every 4th beat starting from the strongest
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        strongest_period = np.argmax(np.mean(tempogram, axis=1))
        beats_per_bar = 4  # assume 4/4
        downbeats = beat_times[::beats_per_bar] if len(beat_times) >= beats_per_bar else []

        # --- Cleanup and Output ---
        os.remove(temp_audio.name)

        return jsonify({
            'bpm': round(tempo, 2),
            'beats': [round(float(t), 6) for t in beat_times],
            'downbeats': [round(float(t), 6) for t in downbeats]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
