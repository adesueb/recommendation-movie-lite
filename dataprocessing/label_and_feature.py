import numpy as np


def getSortedClassContents(dataFrame, minimum_contents_clicked):
    levels = []
    levels.append([0, minimum_contents_clicked + 1])
    contentLevels = dataFrame['Content'].values
    for contentLevel in contentLevels:
        add = True
        indexLevel = 0
        for key, level in levels:
            if contentLevel == key:
                add = False
                levels[indexLevel] = ([key, level + 1])
                break
            else:
                indexLevel += 1
        if add:
            levels.append([contentLevel, 1])

    classContents = []
    for key, level in levels:
        if level > minimum_contents_clicked:
            classContents.append(key)

    sortedClassContents = sorted(classContents)
    print(sortedClassContents)
    return sortedClassContents


def checkConsists(item, contents):
    next = False
    for content in contents:
        if (content == item):
            next = True
    return next


def buildFeature(paramsContents, sortedClassContents, max_video_per_data):
    feature = []
    for ignore in range(max_video_per_data - len(paramsContents)):
        feature.append(0)
    for content in paramsContents:
        feature.append(sortedClassContents.index(content))

    return feature


def saveClassesToFile(classes):
    label_file = open("data/classes.txt", "w")
    np.savetxt(label_file, classes)
    label_file.close()
