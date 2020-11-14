import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional,BatchNormalization, Activation, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from dataprocessing.find_top_movie_no_sequence import find_top_dataset
import datetime
import random


MIN_CONTENTS_ON_USER = 250
MAX_DAYS = 7
MAX_SEQUENCE = 6
DATA_DIR = "data"
recommandation_df = pd.read_csv('{}/normalizedata.csv'.format(DATA_DIR)).sort_values(by=['time'])


def convertToTimemillis(date):
    try:
        d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
        return int(float(d)*1000)
    except:
        pass
    return 0

def getSortedClass():
    levels = []
    levels.append([0, MIN_CONTENTS_ON_USER+1])
    contentLevels = recommandation_df['Content'].values
    for contentLevel in contentLevels:
        add = True
        indexLevel = 0
        for key, level in levels:
            if contentLevel == key:
                add = False
                levels[indexLevel] = ([key, level+1])
                break
            else:
                indexLevel+=1
        if add:
            levels.append([contentLevel, 1])

    classContents = []
    for key, level in levels:
        if level > MIN_CONTENTS_ON_USER:
            classContents.append(key)
            
    sortedClassContents = sorted(classContents)
    print(sortedClassContents)
    return sortedClassContents

def saveClassesToFile(classes):
    label_file = open("label.txt", "w")
    np.savetxt(label_file, classes)
    label_file.close()


def checkingTimeDifferent(nowDate, beforeDate):
    now = convertToTimemillis(nowDate)
    before = convertToTimemillis(beforeDate)
    return (now - before) >  (MAX_DAYS * 86400000)


def buildFeature(paramsContents, sortedClassContents):
    feature = []
    for ignore in range(MAX_SEQUENCE-len(paramsContents)):
        feature.append(0)
    for content in paramsContents:
        feature.append(sortedClassContents.index(content))
        
    return feature

def checkConsists(item, contents):
    next = False
    for content in contents:
        if(content == item):
            next = True
    return next

sortedClassContents = getSortedClass()
saveClassesToFile(sortedClassContents)
lenSortedClassContents = len(sortedClassContents)
print(lenSortedClassContents)

visitors_df = recommandation_df['visitor'].drop_duplicates()
maxItem = recommandation_df['Content'].max()
training_data = []
for index, item in visitors_df.iteritems():
    video = recommandation_df[recommandation_df['visitor'] == item]
    if video.size > 1:
        tempContents = []
        indexContent = 0
        for index, item in video['Content'].iteritems():
            if(item not in sortedClassContents):
                continue
            if len(tempContents) > MAX_SEQUENCE: 
                tempContents = tempContents[1:]
                continue
            if(checkConsists(item, tempContents)):
                continue
            if len(tempContents) > 0:
                nowDate = video['time'].iloc[indexContent]
                beforeDate = video['time'].iloc[indexContent-1]
                
                if checkingTimeDifferent(nowDate, beforeDate):
                    tempContents = []
                else:
                    feature = buildFeature(tempContents, sortedClassContents)
                    label = sortedClassContents.index(item)
                    training_data.append([[feature], label])
            tempContents.append(item)
            indexContent += 1

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

X = np.squeeze(X, axis=1)

model = Sequential()
model.add(Embedding(lenSortedClassContents, 128, input_length=MAX_SEQUENCE))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(lenSortedClassContents, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

name = "testing-1"
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
history = model.fit(X, Y, epochs=45, verbose=1, callbacks=[tensorboard])

predict = model.predict([[0,0, 0,sortedClassContents.index(1506)]])
print(predict)
print(np.argmax(predict[0]))
sortedClassContents[np.argmax(predict[0])]


run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1,4], model.inputs[0].dtype))
MODEL_DIR = "model"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)