import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation, Embedding, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam


def trainWithCNN(xShape, classLen):
    model = Sequential()
    model.add(Conv1D(128, 2, activation="relu", input_shape=xShape))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(classLen, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def trainWithBidirectional(inputMaxLength, classLen):
    # if the x is like RNN don't forget to use this -> X = np.squeeze(X, axis=1)
    # sample input : 
    # [[  0  14   6  65  57 105]
    # [  0   0   0   0   0 142]]
    # X = np.squeeze(X, axis=1)
    model = Sequential()
    model.add(Embedding(classLen, 128, input_length=inputMaxLength))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(classLen, activation='softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def trainWithRNN(xShape, classLen):
    # sample input :
    # [[[  0]
    # [ 14]
    # [  6]
    # [ 65]
    # [ 57]
    # [105]]

    # [[  0]
    # [  0]
    # [  0]
    # [  0]
    # [  0]
    # [142]]]
    # xshape : (6,1) not (1,6)
    # X = np.squeeze(X, axis=1)
    # X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(512, input_shape=xShape, return_sequences=True) )
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(classLen))
    model.add(Activation("softmax"))
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model