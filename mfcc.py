import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def extract_mfcc(audio_path, n_mfcc=13):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # print(f'MFCCs shape: {mfccs.shape}')
    # # Plot the MFCCs
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()

    mfcc_mean = np.mean(mfccs.T) 
    return mfcc_mean
    
