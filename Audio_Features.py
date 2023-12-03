#################################################################################################################################
# File:                 Audio_Features
# Authors:              Brooke McWilliams
# Date Created:         11/19/2023
# Date Last Modified:   11/27/2023
# Description:          Strip features out of audio files for our speech emotion recognition 
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

wav_data_mfcc = []
wav_data_rms = []
wav_data_zcr = []
labelList = []
wav_data = []
fileName = []

for file in os.listdir(path):
    if file.endswith(".wav"):               
        file_path = os.path.join(path, file)
        # Get audio data from each data file
        record, sr = get_data(file_path)
        wav_data.append(record)

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

for audio in padded_data:
    wav_data_mfcc.append((librosa.feature.mfcc(y=audio, n_mfcc=20)).flatten())
    wav_data_rms.append((librosa.feature.rms(y=audio)).flatten())
    wav_data_zcr.append((librosa.feature.zero_crossing_rate(y=audio)).flatten())

# Normalize data features
wav_data_mfcc = np.array(wav_data_mfcc)
wav_data_rms = np.array(wav_data_rms)
wav_data_zcr = np.array(wav_data_zcr)

scale = StandardScaler()

wav_data_mfcc_norm = scale.fit_transform(wav_data_mfcc)
wav_data_rms_norm = scale.fit_transform(wav_data_rms)
wav_data_zcr_norm = scale.fit_transform(wav_data_zcr)

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
    # Sheet has 7442 rows and 4322 columns
    # There are 4320 features extracted
    wave_mfcc_dataframe.to_excel(writer, sheet_name="MFCC Features", index=False)
    # Sheet has 7442 rows and 218 columns
    # There are 216 features extracted 
    wave_rms_dataframe.to_excel(writer,sheet_name="RMS", index=False)
    # Sheet has 7442 rows and 218 columns
    # There are 216 features extracted 
    wave_zcr_dataframe.to_excel(writer, sheet_name="Zero Crossing Rate", index=False)