from datetime import datetime

import numpy as np


def save_data(data, dataPath):
    features = []
    labels = []
    for feature, label in data:
        features.append(feature)
        labels.append(label)

    a_file = open(dataPath, "w")
    for row in features:
        np.savetxt(a_file, row[0])
    a_file.close()

    a_file = open(dataPath, "w")
    np.savetxt(a_file, label)
    a_file.close()
