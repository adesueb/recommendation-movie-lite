import random

import numpy as np
import pandas as pd

from dataprocessing.find_top_movie_with_sequence import find_top_dataset
from dataprocessing.label_and_feature import getSortedClassContents, checkConsists, buildFeature, saveClassesToFile
from dataprocessing.time import checkingTimeDifferent
from find_thebest_neural_network import run_hyper_parameter

MIN_CONTENTS_ON_USER = 250
MAX_DAYS = 7
MAX_SEQUENCE = 1
DATA_DIR = "data"
recommandation_df = pd.read_csv('{}/normalizedata.csv'.format(DATA_DIR)).sort_values(by=['time'])

sortedClassContents = getSortedClassContents(recommandation_df, MIN_CONTENTS_ON_USER)
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
            if item not in sortedClassContents:
                continue
            if len(tempContents) > MAX_SEQUENCE:
                tempContents = tempContents[1:]
                continue
            if checkConsists(item, tempContents):
                continue
            if len(tempContents) > 0:
                nowDate = video['time'].iloc[indexContent]
                beforeDate = video['time'].iloc[indexContent - 1]

                if checkingTimeDifferent(nowDate, beforeDate, MAX_DAYS):
                    tempContents = []
                else:
                    feature = buildFeature(tempContents, sortedClassContents, MAX_SEQUENCE)
                    label = sortedClassContents.index(item)
                    training_data.append([[feature], label])
            tempContents.append(item)
            indexContent += 1

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

features_val = []
labels_val = []
for feature, label in validation_data:
    features_val.append(feature)
    labels_val.append(label)

X_val = np.array(features_val)
Y_val = np.array(labels_val).astype(np.float32)

X_val = np.squeeze(X_val, axis=1)
run_hyper_parameter(X, Y, X_val, Y_val, lenSortedClassContents)
