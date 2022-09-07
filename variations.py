from itertools import count
import math
from os import path
from time import process_time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

MAX_RANDOM = 100
USE_LOGGING = False
FLOAT_ROUND = 2

def euclid(x, y):
    return np.sqrt(sum((a - b) * (a - b) for a,b in zip(x, y)))

def manhatn(x, y):
    return np.abs(x - y)

def getDistance(sums, i, j):
    if (i <= j):
        return sums[i][j - (i + 1)]
    return sums[j][i - (j + 1)]

def calcSum(clusters, k, sums):
    sum = 0
    for clust in range(k):
        cluster = [ind for ind,val in enumerate(clusters) if val == clust]
        s = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                s = s + getDistance(sums, i, j)
        sum += s
    return sum

def validVariation(usedClusters):
    return usedClusters.count(0) == 0

def nextVariation(clusters, k, usedClusters):
    ind = len(clusters) - 1
    for ind in range(len(clusters) - 1, -1, -1):
        if clusters[ind] == k - 1:
            clusters[ind] = 0
            usedClusters[0] = usedClusters[0] + 1
            if ind == 0:
                return False
        else:
            usedClusters[clusters[ind]] = usedClusters[clusters[ind]] - 1
            clusters[ind] = clusters[ind] + 1
            usedClusters[clusters[ind]] = usedClusters[clusters[ind]] + 1
            break
    return True

def bruteForce(data: pd.DataFrame, k, metric):
    result = {}
    result["error"] = ""
    result["clusters"] = []
    result["history"] = []
    result["min"] = math.inf

    if len(data) < k :
        result["error"] = "Number of elements is smaller then number of expected clusters"
        return result
    
    if len(data) == k :
        result["clusters"] = [i for i in range(k)]
        return result

    clusters = [0 for _ in range(len(data))]
    result["clusters"] = clusters
    sums = [[round(metric(data.iloc[i], data.iloc[j]), FLOAT_ROUND) for j in range(i+1, len(data))] for i in range(len(data))]
     
    usedClusters = [0 for _ in range(k)]
    usedClusters[0] = clusters.count(0)

    while True:
        sum = calcSum(clusters, k, sums)

        if validVariation(usedClusters):

            if USE_LOGGING:
                print(clusters)

            if sum < result["min"]:
                result["history"].append(sum)
                result["min"] = sum
                result["clusters"] = clusters.copy()

        if not nextVariation(clusters, k, usedClusters):
            break

    return result

if __name__ == "__main__":

    data = pd.DataFrame()
    k = 3

    if len(sys.argv) > 1 and path.exists(sys.argv[1]):
        data = pd.read_csv(sys.argv[1])
        print('Read data with {} instances'.format(len(data)))
    else:
        print('The given file doesnt exist generating {} random points'.format(MAX_RANDOM))
        np.random.seed()
        data = pd.DataFrame({0: np.random.randint(0, 1000, MAX_RANDOM), 1: np.random.randint(0, 1000, MAX_RANDOM)})

    cat_cols = [col for col in data.columns if data[col].dtype == pd.StringDtype]
    if len(cat_cols) > 0 :
        print("There are string type collumns {} who will be removed ".format(cat_cols))
        data.drop(cat_cols, axis = 1, inplace = True)

    t_time = process_time()
    res = bruteForce(data, k, euclid)
    t_end = process_time()

    print('Time took to complete algorithm : {}'.format(t_end - t_time))
    print('Minimal sum : {}'.format(res["min"]))

    if len(data.columns) >= 2:
        hist = plt.subplot(1, 2, 1)
        plt.title("BRUTE FORCE ALGORITHM")
        hist.set_xlabel("iteration")
        hist.set_ylabel("min")

        plt.plot([i for i in range(len(res["history"]))], res["history"])

        plt.subplot(1, 2, 2)
        for i in range(k):
            cluster = [row for row,val in enumerate(res["clusters"]) if val == i]
            plt.scatter(data.iloc[cluster,0], data.iloc[cluster,1], label = i)
        plt.legend(title="clusters")
        plt.show()