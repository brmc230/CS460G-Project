#################################################################################################################################
# File:                 CNN_Model
# Authors:              James Birch
# Date Created:         12/01/2023
# Date Last Modified:   12/01/2023
# Description:          Create Basic CNN Model to be used for Speech Emotion Recognition
# Misc:                 . . .
#################################################################################################################################

#from Audio_Features import *
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Activation, BatchNormalization, Dropout, MaxPooling1D, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

#################################################################################################################################
def eval_model(X_train, y_train, X_test, y_test, num_feat):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', input_shape=(X_train.shape[1],1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=128, kernel_size=5, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=32, kernel_size=5, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(60))
    model.add(Dropout(0.5))

    model.add(Dense(num_feat))
    model.add(Activation('softmax'))
    model.add(BatchNormalization())

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    accuracy = model.evaluate(X_test, y_test, verbose=1)

    return accuracy[1]
#################################################################################################################################

# read in data
excel1 = 'Features.xlsx'
excel2 = 'NewFeatures.xlsx'

# wave_df = pd.read_excel(excel1, header = 0, sheet_name='MFCC Features')
# wave_df = pd.read_excel(excel1, header = 0, sheet_name='RMS')
# wave_df = pd.read_excel(excel1, header = 0, sheet_name='Zero Crossing Rate')

wave_df = pd.read_excel(excel2, header = 0, sheet_name='MFCC Features')
# wave_df = pd.read_excel(excel2, header = 0, sheet_name='RMS')
# wave_df = pd.read_excel(excel2, header = 0, sheet_name='Zero Crossing Rate')

print('Download Complete!')

# Converted wave data
X = wave_df.drop(['ID/File','Labels'], axis=1)
y = wave_df['Labels']

encoder = LabelEncoder()
y_1 = encoder.fit_transform(y)

num_feat = len(pd.unique(y_1))

X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size=0.33, random_state=42)

acc = eval_model(X_train, y_train, X_test, y_test, num_feat)
print(f"Model Accuracy: {acc:.3f}")
