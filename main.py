import sys
import json
import numpy as np

# Try to import madmom and librosa; they should be installed in the environment.
try:
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
except ImportError as e:
    sys.stderr.write("Error: required library not found. Make sure madmom is installed.\n")
    sys.exit(1)
try:
    import librosa
except ImportError:
    sys.stderr.write("Error: librosa is required for audio loading. Please install librosa.\n")
    sys.exit(1)

# Usage check
if len(sys.argv) < 2:
    sys.stderr.write(f"Usage: python {sys.argv[0]} <audio_file.wav/mp3>\n")
    sys.exit(1)

audio_path = sys.argv[1]

# 1. Load audio file (wav or mp3) and ensure mono audio at 44100 Hz
try:
    # Use madmom's Signal class to load and resample audio
    signal = Signal(audio_path, sample_rate=44100, num_channels=1)
    sr = signal.sample_rate  # should be 44100 as we set
    audio_data = signal  # madmom Signal object can be passed directly to processors
except Exception as e:
    # Fallback: if madmom Signal fails (e.g., for mp3 if ffmpeg not available), use librosa
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    audio_data = y  # numpy array of audio samples (will be accepted by madmom Processor)

# 2. RNN Downbeat/Beat activation processing
rnn_processor = RNNDownBeatProcessor()  # uses a pre-trained BLSTM model under the hood
activations = rnn_processor(audio_data)  # shape: (N_frames, 2) -> [beat_prob, downbeat_prob] per frame

# 3. Dynamic Bayesian Network (DBN) tracking to get beats and downbeats
# We'll allow 3/4 and 4/4 time signatures (common in music) and use 100 FPS as the RNN output rate.
dbn_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
beat_sequence = dbn_processor(activations)  # numpy array of shape (num_beats, 2)

# The output has two columns: [time, beat_index_in_bar]
# beat_index_in_bar = 1 indicates a downbeat.
if beat_sequence.shape[0] == 0:
    # No beats detected (e.g., silent input)
    result = {"bpm": 0, "beats": [], "downbeats": []}
else:
    # 4. Extract beat times and downbeat times
    beat_times = beat_sequence[:, 0]      # times of all detected beats (in seconds)
    beat_positions = beat_sequence[:, 1]  # position of each beat in the bar (1 = downbeat)

    # Downbeat times are those where beat_positions == 1
    downbeat_times = beat_times[beat_positions == 1]

    # Compute BPM from inter-beat intervals (use median interval for robustness)
    beat_intervals = np.diff(beat_times)
    if len(beat_intervals) > 0:
        median_interval = np.median(beat_intervals)
        bpm = float(60.0 / median_interval)
    else:
        bpm = 0.0  # not enough info (e.g., only one beat detected)

    # Convert numpy arrays to regular Python lists for JSON serialization
    beat_list = [float(t) for t in beat_times]
    downbeat_list = [float(t) for t in downbeat_times]

    result = {
        "bpm": round(bpm, 2),         # round BPM to 2 decimal places (can adjust precision as needed)
        "beats": beat_list,           # list of all beat timestamps in seconds
        "downbeats": downbeat_list    # list of downbeat timestamps in seconds
    }

# 5. Output the result as a JSON string
print(json.dumps(result, indent=2))
