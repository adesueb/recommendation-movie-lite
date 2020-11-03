import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional,BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import datetime
import random
import matplotlib.pyplot as plt



def convertToTimemillis(date):
    try:
        d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
        return int(float(d)*1000)
    except:
        pass
    return 0

DATA_DIR = "data"
recommandation_df = pd.read_csv('{}/data.csv'.format(DATA_DIR)).sort_values(by=['time'])

levels = []

MIN_CONTENTS_ON_USER = 600 

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
lenSortedClassContents = len(sortedClassContents)
print(sortedClassContents)
print(lenSortedClassContents)

MAX_DAYS = 7
def checkingTimeDifferent(nowDate, beforeDate):
    now = convertToTimemillis(nowDate)
    before = convertToTimemillis(beforeDate)
    return (now - before) >  (MAX_DAYS * 86400000)

MAX_SEQUENCE = 4
def buildFeature(paramsContents):
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
                    feature = buildFeature(tempContents)
                    label = sortedClassContents.index(item)
                    training_data.append([[feature], label])
            tempContents.append(item)
            indexContent += 1
print("training size: ", len(training_data))

random.shuffle(training_data)

features = []
labels = []
for feature, label in training_data:
    features.append(feature)
    labels.append(label)


X = np.array(features)
Y = np.array(labels).astype(np.float32)
print (X)
print (Y)

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1:]), return_sequences=True) )
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(256, activation='relu'))
model.add(Dense(lenSortedClassContents))
model.add(Activation("softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=120, batch_size=16, validation_split=0.1)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')

predict = model.predict([[0,0, 0,sortedClassContents.index(1506)]])
print(predict)
print(np.argmax(predict[0]))
sortedClassContents[np.argmax(predict[0])]

run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1,4], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "model"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)