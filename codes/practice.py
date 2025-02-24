def getLargestIndexLen(feature1, feature2):
    count = 0

    for i in range(len(feature1)-1):
        if feature1[i] > feature1[i+1] and feature2[i] > feature2[i+1]:
            count += 1
        elif feature1[i] < feature1[i+1] and feature2[i] < feature2[i+1]:
            count += 1

    count += 1
    return count

print(getLargestIndexLen([3,2,1], [6,5,4]))
print(getLargestIndexLen([1,2,3,4,5], [5,4,3,2,1]))
print(getLargestIndexLen([4,5,3,1,2], [2,1,3,4,5]))