from datetime import datetime

from dataprocessing.label_and_feature import checkConsists, buildFeature


def build_training_data(data_frame, sortedClassContents, maxSequence):
    # datetime object containing current date and time
    now = datetime.now()

    print("start =", now)

    visitors_df = data_frame['visitor'].drop_duplicates()
    training_data = []
    for index, item in visitors_df.iteritems():
        video = data_frame[data_frame['visitor'] == item]
        if video.size > 1:
            tempContents = []
            indexContent = 0
            for index, item in video['Content'].iteritems():
                if item not in sortedClassContents:
                    continue
                if len(tempContents) > maxSequence:
                    tempContents = tempContents[1:]
                    continue
                if checkConsists(item, tempContents):
                    continue
                if len(tempContents) > 0:
                    feature = buildFeature(tempContents, sortedClassContents, maxSequence)
                    label = sortedClassContents.index(item)
                    training_data.append([[feature], label])
                tempContents.append(item)
                indexContent += 1

    now = datetime.now()
    print("end =", now)
    return training_data
