import numpy as np
from dataprocessing.builder import build_training_data
from dataprocessing.label_and_feature import getSortedClassContents, saveClassesToFile
from dataprocessing.save_features_labels import save_data
from datetime import datetime
import pandas as pd

MIN_CONTENTS_ON_USER = 50
MAX_DAYS = 7
MAX_SEQUENCE = 1
DATA_DIR = "../data"

recommandation_df = pd.read_csv('{}/data.csv'.format(DATA_DIR)).sort_values(by=['time'])

now = datetime.now()
filePathLabels = '{}/backup_labels-{}.txt'.format(DATA_DIR, now)
filePathFeatures = '{}/backup_features-{}.txt'.format(DATA_DIR, now)
sortedClassContents = getSortedClassContents(recommandation_df, MIN_CONTENTS_ON_USER)
saveClassesToFile(sortedClassContents)
lenSortedClassContents = len(sortedClassContents)
print(lenSortedClassContents)

data1 = pd.read_csv('{}/data1.csv'.format(DATA_DIR)).sort_values(by=['time'])
training_data = build_training_data(data1, sortedClassContents, MAX_SEQUENCE)
print(len(training_data))
save_data(training_data, filePathFeatures, filePathLabels)

data2 = pd.read_csv('{}/data2.csv'.format(DATA_DIR)).sort_values(by=['time'])
training_data = np.concatenate(training_data, build_training_data(data2, sortedClassContents, MAX_SEQUENCE))
print(len(training_data))
save_data(training_data, filePathFeatures, filePathLabels)

data3 = pd.read_csv('{}/data3.csv'.format(DATA_DIR)).sort_values(by=['time'])
training_data = np.concatenate((training_data, build_training_data(data3, sortedClassContents, MAX_SEQUENCE)))
print(len(training_data))
save_data(training_data, filePathFeatures, filePathLabels)

data4 = pd.read_csv('{}/data4.csv'.format(DATA_DIR)).sort_values(by=['time'])
training_data = np.concatenate((training_data, build_training_data(data4, sortedClassContents, MAX_SEQUENCE)))
print(len(training_data))
save_data(training_data, filePathFeatures, filePathLabels)

data5 = pd.read_csv('{}/data5.csv'.format(DATA_DIR)).sort_values(by=['time'])
training_data = np.concatenate((training_data, build_training_data(data5, sortedClassContents, MAX_SEQUENCE)))
print(len(training_data))
save_data(training_data, filePathFeatures, filePathLabels)

data6 = pd.read_csv('{}/data6.csv'.format(DATA_DIR)).sort_values(by=['time'])
training_data = np.concatenate((training_data, build_training_data(data6, sortedClassContents, MAX_SEQUENCE)))
print(len(training_data))
save_data(training_data, filePathFeatures, filePathLabels)
