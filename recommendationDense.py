import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

from dataprocessing.find_top_movie_with_sequence import find_top_dataset
from recom_model import trainWithDense

# using cpu -> os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def prepare_data():
    DATA_DIR = "backup"
    features = np.loadtxt("{}/backup_features_small.txt".format(DATA_DIR))
    labels = np.loadtxt("{}/backup_labels_small.txt".format(DATA_DIR))
    classes = np.loadtxt("{}/classes_small.txt".format(DATA_DIR))
    print("training data : {}".format(len(features)))

    training_data = []
    for i in range(len(labels)):
        training_data.append([[features[i]], labels[i]])
        training_data.append([[labels[i]], features[i]])
    print(training_data)
    print("training size : {}".format(len(training_data)))
    return training_data, classes


data = prepare_data()
classes = data[1]
lenSortedClassContents = len(classes)
print("classes : {}".format(lenSortedClassContents))

training_data = data[0]
validation_data = find_top_dataset(training_data)
print("validation data : {}".format(len(validation_data)))

random.shuffle(training_data)
features = []
labels = []
for feature, label in training_data:
    features.append(feature)
    labels.append(label)

X = np.array(features)
Y = np.array(labels)
print(X.shape)

features_val = []
labels_val = []
for feature, label in validation_data:
    features_val.append(feature)
    labels_val.append(label)

X_val = np.array(features_val)
Y_val = np.array(labels_val).astype(np.float32)
model = trainWithDense(X.shape[1:], lenSortedClassContents)

now = datetime.now()
name = "testing-1-{}".format(now)
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
model.fit(X, Y, epochs=256, batch_size=64, verbose=1, validation_data=(X_val, Y_val), callbacks=[tensorboard])

classes = list(classes)


def get5TopPredict(predict):
    predicts = predict[0].argsort()[-5:][::-1]
    for i in predicts:
        print(classes[i])


def predictDense(first):
    predict = model.predict([[classes.index(first)]])
    get5TopPredict(predict)
    return classes[np.argmax(predict[0])]


predictResult = predictDense(700)

# saving model
MODEL_DIR = "model"
model.save(MODEL_DIR)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
