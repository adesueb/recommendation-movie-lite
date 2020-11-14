def is_array_equals_without_sequence(a, b):
    checker = b.copy()
    for z in a:
        different = True
        for index, zz in enumerate(checker):
            if z == zz:
                different = False
                del checker[index]
                break
        if different:
            return False
    return True


def find_top_dataset(self):
    unique = []
    for x, y in self:
        new = True
        i = 0
        for u, l in unique:
            if is_array_equals_without_sequence(x[0], u[0]):
                new = False
                l.append(y)
                unique[i] = [u, l]
            i += 1
        if new:
            unique.append([x, [y]])
    result = []
    for x, y in unique:
        scores = []
        for i in y:
            numbersIndex = 0
            for z, score in scores:
                if i == z:
                    scores[numbersIndex] = [i, score + 1]
                numbersIndex += 1
            scores.append([i, 0])
        maxY = 0
        topY = scores[0][0]
        for n, score in scores:
            if (score > maxY):
                maxY = score
                topY = n
        result.append([x, topY])
    return result


a = [0, 2, 1]
b = [0, 1, 2]
print(is_array_equals_without_sequence(a, b))

dataset = [
    [[[0, 0, 0, 74]], 3],
    [[[0, 0, 74, 0]], 1],
    [[[0, 0, 0, 2]], 3],
    [[[0, 0, 2, 1]], 2],
    [[[0, 0, 0, 2]], 4],
    [[[0, 0, 0, 2]], 4],
    [[[0, 0, 0, 3]], 5],
    [[[0, 0, 0, 4]], 7],
    [[[0, 0, 2, 1]], 1],
    [[[0, 0, 2, 1]], 1],
    [[[0, 0, 0, 1]], 1],
    [[[0, 0, 1, 2]], 3],
    [[[0, 0, 0, 3]], 5],
    [[[0, 0, 0, 4]], 7],
    [[[0, 0, 0, 1]], 1],
    [[[0, 0, 0, 1]], 1]
]

print(find_top_dataset(dataset))
