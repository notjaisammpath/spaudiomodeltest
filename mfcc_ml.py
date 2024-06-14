import librosa
import numpy as np
import os
import mfcc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#training data from TESS - https://tspace.library.utoronto.ca/handle/1807/24487

dataset_dir = "/Users/jaisammpath/Documents/spaudiomodeltest/dataset"

features = []
labels = []

for emotion in os.listdir(dataset_dir):
    full_emotion_dir = os.path.join(dataset_dir, emotion)
    # print (emotion_dir)
    for audio_filename in os.listdir(full_emotion_dir):
        audio_path = os.path.join(full_emotion_dir, audio_filename)
        mfcc_features = mfcc.extract_mfcc(audio_path)
        features.append(mfcc_features)
        labels.append(emotion)

labels = np.array(labels)
features = np.array(features)
features = features.reshape(-1, 1)

print (labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)
print(y_train)
print(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))