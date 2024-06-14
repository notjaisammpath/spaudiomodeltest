import librosa
import numpy as np
import json

# Load audio file
def calculate_db(audio_path):

    audio, sample_rate = librosa.load("db_samples/" + audio_path)

    # Calculate reference value
    reference_value = 20e-6  # 20 μPa, reference for 0 dB SPL ( Sound pressure Level )

    # Calculate sample frames per second
    frames_per_second = int((1 * sample_rate)/10)

    # Calculate the number of segments (1-second each)
    num_segments = (len(audio) // frames_per_second)

    # Initialize an array to store dB values for each segment
    db_values = []

    # Iterate over each second segment
    for segment_idx in range(num_segments):
        start_frame = segment_idx * (frames_per_second)
        end_frame = (segment_idx + 1) * (frames_per_second)

# Extract audio segment
        audio_segment = audio[start_frame:end_frame]

# Calculate RMS amplitude for the segment
        rms_amplitude = np.sqrt(np.mean(np.square(audio_segment)))

# Calculate dB SPL for the segment
        db_spl = 20 * np.log10(rms_amplitude / reference_value)

        db_values.append(db_spl)

    # Now db_values contains the dB SPL values for each second segment
    array=np.array(db_values)
    array[np.isinf(array)] = 0

    mean_value = float(np.mean(array))
    data = [array.tolist(),mean_value]
    json_data = json.dumps(data)

    return(json_data)