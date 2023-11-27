#################################################################################################################################
# File:                 Audio_Features
# Authors:              Brooke McWilliams
# Date Created:         11/19/2023
# Date Last Modified:   
# Description:          Strip features out of audio files for our speech emotion recognition 
# Misc:                 . . .
#################################################################################################################################

import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd

path = "/Crema Dataset"

emotionLabels = {"SAD": 0, "ANG": 1, "DIS": 2, "FEA": 3, "HAP": 4, "NEU": 5}

def extract_features(file_path):
    data, sampleRate = librosa.load(file_path)
    data_features = librosa.feature.mfcc(y=data, sr=sampleRate, n_mfcc=20)
    return data_features.flatten()

wav_data = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    features = extract_features(file_path)
    wav_data.append([file] + features.tolist())

columns = ['filename'] + [f'feature_{i}' for i in range(len(wav_data[0]) - 1)]

wave_dataframe = pd.DataFrame(wav_data, columns=columns)