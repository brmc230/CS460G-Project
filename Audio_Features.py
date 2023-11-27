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

#################################################################################################################################
def extract_features_mfcc(file_path):
    data, sampleRate = librosa.load(file_path)
    data_features = librosa.feature.mfcc(y=data, sr=sampleRate, n_mfcc=13)

    return data_features.flatten()
#################################################################################################################################
def extract_features_rms(file_path):
    data, sampleRate = librosa.load(file_path)
    data_features = librosa.feature.rms(y=data)

    return data_features.flatten()
#################################################################################################################################
def extract_features_zero_crossing_rate(file_path):
    data, sampleRate = librosa.load(file_path)
    data_features = librosa.feature.zero_crossing_rate(y=data)

    return data_features.flatten()
#################################################################################################################################

path = "./Crema Dataset/"
excel_path = "./Features.xlsx"

wav_data_mfcc = []
wav_data_rms = []
wav_data_zcr = []
labelList = []
for file in os.listdir(path):
    if file.endswith(".wav"):               
        file_path = os.path.join(path, file)

        # MFCC
        features_mfcc = extract_features_mfcc(file_path)
        wav_data_mfcc.append([file] + features_mfcc.tolist())

        #RMS
        features_rms = extract_features_rms(file_path)
        wav_data_rms.append([file] + features_rms.tolist())

        # Zero Crossing Rate
        features_zcr = extract_features_zero_crossing_rate(file_path)
        wav_data_zcr.append([file] + features_zcr.tolist())

        # Labels for files
        labelList.append(file.split('_')[2])

# MFCC
wave_mfcc_dataframe = pd.DataFrame(wav_data_mfcc)
wave_mfcc_dataframe.dropna(axis=1, inplace=True)
wave_mfcc_dataframe.insert(1, "Labels", labelList)
wave_mfcc_dataframe.rename(columns={0: "ID/File"}, inplace=True)

#RMS
wave_rms_dataframe = pd.DataFrame(wav_data_rms)
wave_rms_dataframe.dropna(axis=1, inplace=True)
wave_rms_dataframe.insert(1, "Labels", labelList)
wave_rms_dataframe.rename(columns={0: "ID/File"}, inplace=True)

# Zero Crossing Rate
wave_zcr_dataframe = pd.DataFrame(wav_data_zcr)
wave_zcr_dataframe.dropna(axis=1, inplace=True)
wave_zcr_dataframe.insert(1, "Labels", labelList)
wave_zcr_dataframe.rename(columns={0: "ID/File"}, inplace=True)

# 3 Total sheets in the file for 3 different feature extraction methods
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    # Sheet has 7442 rows and 717 columns
    # There are 715 features extracted
    wave_mfcc_dataframe.to_excel(writer, sheet_name="MFCC Features", index=False)
    # Sheet has 7442 rows and 57 columns
    # There are 55 features extracted 
    wave_rms_dataframe.to_excel(writer,sheet_name="RMS", index=False)
    # Sheet has 7442 rows and 57 columns
    # There are 55 features extracted 
    wave_zcr_dataframe.to_excel(writer, sheet_name="Zero Crossing Rate", index=False)