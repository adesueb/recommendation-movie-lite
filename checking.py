import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import datetime

def convertToTimemillis(date):
    try:
        d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
        return int(float(d)*1000)
    except:
        pass
    return 0

DATA_DIR = "data"
recommandation_df = pd.read_csv('{}/normalizedata.csv'.format(DATA_DIR)).sort_values(by=['time'])

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

model = tf.keras.models.load_model('model')
predict = model.predict_classes([[0,0, 0,sortedClassContents.index(1506)]])
print(sortedClassContents.index(predict))


