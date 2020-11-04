a = [[[0,0,0,1], 5], 
[[0,0,0,2], 1],
[[0,0,0,3], 2],
[[0,0,0,4], 3],
[[0,0,0,5], 8],
[[0,0,0,3], 2],
[[0,0,0,3], 1],
[[0,0,0,4], 1],
[[0,0,0,4], 1],
[[0,0,0,2], 1],
[[0,0,0,2], 2],
[[0,0,0,5], 1],
[[0,0,0,3], 4],
[[0,0,0,2], 1],
[[0,0,1,2], 7],
[[0,0,1,2], 1],
[[0,0,1,2], 1]]

def findTopDataSet(dataSet):
    unique = []
    for x, y in dataSet:
        new = True
        i = 0
        for u, l in unique:
            if u == x:
                new = False
                l.append(y)
                unique[i] = [u, l]
            i+=1
        if new:
            unique.append([x,[y]])
    result = []
    for x, y in unique:
        scores = []
        for i in y:
            new = True
            numbersIndex = 0
            for z, score in scores:
                if i == z:
                    new = False
                    scores[numbersIndex] = [i, score+1]
                numbersIndex += 1
            scores.append([i, 0])
        maxY = 0
        topY =  scores[0][0]
        for n, score in scores:
            if(score > maxY):
                maxY = score
                topY = n
        result.append([x,topY])
    return result
print(findTopDataSet(a))
    