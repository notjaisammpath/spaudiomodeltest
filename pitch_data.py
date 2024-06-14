import librosa
import matplotlib.pyplot as plt
import numpy as np
import json

def pitch_magnitudes(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    # Compute the pitch using librosa's piptrack function
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    # Get the pitch frequencies from the pitch array
    pitch_frequencies = pitch.max(axis=0)
    #   Convert the frame index to time (seconds)
    timestamps = librosa.times_like(pitch_frequencies, sr=sr)

    segment_duration = int((1 * sr)/5) # Number of samples per 5 seconds
    segments = [y[i:i+segment_duration] for i in range(0, len(y), segment_duration)]
    pitch_points_list = []
    pitch_means = []
    for i, segment in enumerate(segments):
        # Compute the pitch using librosa's piptrack function for each segment
        pitch, _ = librosa.piptrack(y=segment, sr=sr)

        pitch_freq_data = pitch.max(axis=0)

        timestamp = librosa.samples_to_time(i * segment_duration, sr=sr)

        # Append the pitch frequency and timestamp to the lists
        if (timestamp != 0.0):
            pitch_points_list.append(pitch_freq_data)
            mean = np.mean(pitch_freq_data)
            pitch_means.append(mean)
            
            
    array = np.array(pitch_means)
    mean_value = float(np.mean(array))
    data = [array.tolist(),mean_value]
    json_data = json.dumps(data)
    return(json_data)