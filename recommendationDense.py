import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from dataprocessing.builder import build_training_data
from dataprocessing.find_top_movie_with_sequence import find_top_dataset
from dataprocessing.label_and_feature import getSortedClassContents, saveClassesToFile
from recom_model import trainWithDense

MIN_CONTENTS_ON_USER = 250
MAX_DAYS = 7
MAX_SEQUENCE = 1
DATA_DIR = "data"

recommandation_df = pd.read_csv('{}/normalizedata.csv'.format(DATA_DIR)).sort_values(by=['time'])

sortedClassContents = getSortedClassContents(recommandation_df, MIN_CONTENTS_ON_USER)
saveClassesToFile(sortedClassContents)
lenSortedClassContents = len(sortedClassContents)
print(lenSortedClassContents)

training_data = build_training_data(recommandation_df, sortedClassContents, MAX_SEQUENCE)

validation_data = find_top_dataset(training_data)
print("training size: ", len(training_data))
print("validation size: ", len(validation_data))
random.shuffle(training_data)

features = []
labels = []
for feature, label in training_data:
    features.append(feature)
    labels.append(label)

X = np.array(features)
Y = np.array(labels).astype(np.float32)
print(X.shape)

features_val = []
labels_val = []
for feature, label in validation_data:
    features_val.append(feature)
    labels_val.append(label)

X_val = np.array(features_val)
Y_val = np.array(labels_val).astype(np.float32)

X = np.squeeze(X, axis=1)
model = trainWithDense(MAX_SEQUENCE, lenSortedClassContents)

now = datetime.now()
name = "testing-1-{}".format(now)
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
history = model.fit(X, Y, epochs=45, verbose=1, validation_data=(X_val, Y_val), callbacks=[tensorboard])

predict = model.predict([[0, 0, 0, sortedClassContents.index(1506)]])
print(predict)
print(np.argmax(predict[0]))
print(sortedClassContents[np.argmax(predict[0])])

# saving model
MODEL_DIR = "model"
model.save(MODEL_DIR)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
