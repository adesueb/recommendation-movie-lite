import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from dataprocessing.builder import build_training_data
from dataprocessing.find_top_movie_with_sequence import find_top_dataset
from dataprocessing.label_and_feature import getSortedClassContents, saveClassesToFile
from recom_model import trainWithBidirectional

MIN_CONTENTS_ON_USER = 250
MAX_DAYS = 7
MAX_SEQUENCE = 6
DATA_DIR = "data"

recommandation_df = pd.read_csv('{}/normalizedata.csv'.format(DATA_DIR)).sort_values(by=['time'])

sortedClassContents = getSortedClassContents(recommandation_df, MIN_CONTENTS_ON_USER)
saveClassesToFile(sortedClassContents)
lenSortedClassContents = len(sortedClassContents)
print(lenSortedClassContents)

training_data = build_training_data(recommandation_df, sortedClassContents, MAX_SEQUENCE)

training_data = find_top_dataset(training_data)
print("training size: ", len(training_data))
random.shuffle(training_data)

features = []
labels = []
for feature, label in training_data:
    features.append(feature)
    labels.append(label)

X = np.array(features)
Y = np.array(labels).astype(np.float32)
print(X.shape)

X = np.squeeze(X, axis=1)
model = trainWithBidirectional(MAX_SEQUENCE, lenSortedClassContents)

name = "testing-1"
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
history = model.fit(X, Y, epochs=45, verbose=1, callbacks=[tensorboard])

predict = model.predict([[0, 0, 0, sortedClassContents.index(1506)]])
print(predict)
print(np.argmax(predict[0]))
print(sortedClassContents[np.argmax(predict[0])])

# for embedding LSTM
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6], model.inputs[0].dtype))
MODEL_DIR = "model"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
