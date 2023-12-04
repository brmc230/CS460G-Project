#################################################################################################################################
# File:                 Audio_Features
# Authors:              Brooke McWilliams
# Date Created:         11/19/2023
# Date Last Modified:   12/03/2023
# Description:          Strip features out of audio files for our speech emotion recognition using the librosa library 
# Misc:                 . . .
#################################################################################################################################

import os
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#################################################################################################################################
def get_data(filepath):
    data, sampleRate = librosa.load(filepath)
    return data, sampleRate
#################################################################################################################################

path = "./Crema Dataset/"
excel_path = "./NewFeatures.xlsx"

wav_data_mfcc_norm = []
wav_data_rms_norm = []
wav_data_zcr_norm = []
labelList = []
wav_data = []
fileName = []
sampleR = []

for file in os.listdir(path):
    if file.endswith(".wav"):               
        file_path = os.path.join(path, file)
        # Get audio data from each data file
        record, sr = get_data(file_path)
        wav_data.append(record)
        sampleR.append(sr)

        # Labels for each data file
        labelList.append(file.split('_')[2])

        # File names list for tacking
        fileName.append(file)

# Find the max audio file length to use for padding
maxL = 0
minL = float('inf')
for arr in wav_data:
    length = len(arr)
    if length > maxL:
        maxL = length
    if length < minL:
        minL = length

# Need to pad the audio file data so that the feature space is evenly distributed 
padded_data = []
for signal in wav_data:
    padW = maxL - len(signal)
    padS = np.pad(signal, (0, padW), mode="constant")
    padded_data.append(padS)

# Extract and normalize data features
for audio, rate in zip(padded_data, sampleR):
    wav_data_mfcc = librosa.feature.mfcc(y=audio, sr=rate, htk=True, n_mfcc=13).flatten()
    wav_data_mfcc_norm.append((wav_data_mfcc - np.mean(wav_data_mfcc)) / np.std(wav_data_mfcc))

    wav_data_rms = librosa.feature.rms(y=audio).flatten()
    wav_data_rms_norm.append((wav_data_rms - np.mean(wav_data_rms)) / np.std(wav_data_rms))

    wav_data_zcr = librosa.feature.zero_crossing_rate(y=audio).flatten()
    wav_data_zcr_norm.append((wav_data_zcr - np.min(wav_data_zcr)) / (np.max(wav_data_zcr) - np.min(wav_data_zcr)))

# Create dataframes to work with for each feature extraction method 
# MFCC
wave_mfcc_dataframe = pd.DataFrame(wav_data_mfcc_norm)
wave_mfcc_dataframe.insert(0, "ID/File", fileName)
wave_mfcc_dataframe.insert(1, "Labels", labelList)

#RMS
wave_rms_dataframe = pd.DataFrame(wav_data_rms_norm)
wave_rms_dataframe.insert(0, "ID/File", fileName)
wave_rms_dataframe.insert(1, "Labels", labelList)

# Zero Crossing Rate
wave_zcr_dataframe = pd.DataFrame(wav_data_zcr_norm)
wave_zcr_dataframe.insert(0, "ID/File", fileName)
wave_zcr_dataframe.insert(1, "Labels", labelList)

# 3 Total sheets in the file for 3 different feature extraction methods
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:

    wave_mfcc_dataframe.to_excel(writer, sheet_name="MFCC Features", index=False)

    wave_rms_dataframe.to_excel(writer,sheet_name="RMS", index=False)

    wave_zcr_dataframe.to_excel(writer, sheet_name="Zero Crossing Rate", index=False)