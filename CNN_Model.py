#################################################################################################################################
# File:                 CNN_Model
# Authors:              James Birch
# Date Created:         12/01/2023
# Date Last Modified:   12/03/2023
# Description:          Create a CNN Model to be used for Speech Emotion Recognition
# Misc:                 . . .
#################################################################################################################################

#from Audio_Features import *
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Activation, BatchNormalization, Dropout, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#################################################################################################################################
def build_model(X_train, y_train, X_test, y_test, num_feat):

    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', input_shape=(X_train.shape[1],1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, padding='same'))
    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, padding='same'))
    
    model.add(Conv1D(filters=128, kernel_size=5, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3, padding='same'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_feat))
    model.add(Activation('softmax'))

    model.summary()

    # set higher learning rate
    opt = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#################################################################################################################################

# read in data
excel1 = 'Features.xlsx'
excel2 = 'NewFeatures.xlsx'

# wave_df = pd.read_excel(excel1, header = 0, sheet_name='MFCC Features')
# wave_df = pd.read_excel(excel1, header = 0, sheet_name='RMS')
# wave_df = pd.read_excel(excel1, header = 0, sheet_name='Zero Crossing Rate')

# wave_df = pd.read_excel(excel2, header = 0, sheet_name='MFCC Features')
wave_df = pd.read_excel(excel2, header = 0, sheet_name='RMS')
# wave_df = pd.read_excel(excel2, header = 0, sheet_name='Zero Crossing Rate')

# Converted wav data
X = wave_df.iloc[:, 2:]
y = wave_df['Labels']

# encode label column for training
encoder = LabelEncoder()
y =encoder.fit_transform(y)
num_feat = len(pd.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# send training data to model function and return compiled CNN model
model = build_model(X_train, y_train, X_test, y_test, num_feat)

# use early stoping to stop model fitting if accuraccy is continuously not improving
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, epochs=33, batch_size=64, validation_data=(X_test, y_test))#, callbacks=[early_stopping])
acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Accuracy: {acc[1]:.4f}")