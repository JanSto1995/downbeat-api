from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
import soundfile as sf
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return '🎵 Downbeat Detection API is running.'

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    path = os.path.join('/tmp', filename)
    file.save(path)

    try:
        # Load audio
        y, sr = librosa.load(path, sr=44100)

        # Step 1: Tempo & Beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Step 2: Downbeats estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        strongest_period = np.argmax(np.mean(tempogram, axis=1))
        beats_per_bar = 4
        downbeats = beat_times[::beats_per_bar] if len(beat_times) >= beats_per_bar else []

        return jsonify({
            'filename': filename,
            'bpm': round(tempo, 2),
            'beats': [round(float(t), 6) for t in beat_times],
            'downbeats': [round(float(t), 6) for t in downbeats]
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(path):
            os.remove(path)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
