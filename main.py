from flask import Flask, request, jsonify
import os
import librosa
import soundfile as sf
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
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
        y, sr = librosa.load(path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        downbeats = librosa.frames_to_time(beats, sr=sr).tolist()
        duration = librosa.get_duration(y=y, sr=sr)
        return jsonify({
            'filename': filename,
            'length_seconds': round(duration, 2),
            'downbeats': downbeats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
