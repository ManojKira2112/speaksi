import pandas as pd
import librosa
import numpy as np
import os
import pickle
from pydub import AudioSegment
import io
import soundfile as sf
import subprocess
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('audio_classification_model.keras')
def feature_extractor(file_name):
    audio,sample_rate=librosa.load(file_name,res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
     # Extract Chroma features
    chroma_features = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled_features = np.mean(chroma_features.T, axis=0)
    
    # Extract Spectral Contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)
    
    # Extract Zero-Crossing Rate features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_scaled = np.mean(zero_crossing_rate.T, axis=0)
    
    # Combine all features into a single array
    features = np.hstack((mfccs_scaled_features, chroma_scaled_features, spectral_contrast_scaled, zero_crossing_rate_scaled))
    

    return features
#predicton
subprocess.run(['ffmpeg', '-i', 'recording.wav', '-acodec', 'pcm_s16le', '-ar', '44100', 'output.wav'])

filename = "output.wav"

prediction_feature=feature_extractor(filename)
prediction_feature=prediction_feature.reshape(1,-1)
prediction=loaded_model.predict(prediction_feature)
predicted_label = np.argmax(prediction, axis=1)
predicted_class = label_encoder.inverse_transform(predicted_label)
print(predicted_class[0])