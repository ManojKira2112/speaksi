from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
import pandas as pd
from pydub import AudioSegment
import io
import soundfile as sf
import subprocess
import uuid

app = Flask(__name__)


# Load your trained model
model = load_model('audio_classification_model.keras')

# Load your label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def feature_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    chroma_features = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled_features = np.mean(chroma_features.T, axis=0)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_scaled = np.mean(zero_crossing_rate.T, axis=0)
    features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled, zero_crossing_rate_scaled))
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join('./', file.filename)
    file.save(file_path)
    subprocess.run(['ffmpeg', '-i', 'file_path', '-acodec', 'pcm_s16le', '-ar', '44100', f'{uuid.uuid4()}.wav'])
    filename = "output.wav"

    features = feature_extractor(filename)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_label)
    os.remove(file_path)  # Clean up the saved file
    return jsonify({'class': predicted_class[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
