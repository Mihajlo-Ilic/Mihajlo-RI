import math
from os import path
from time import process_time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

MAX_RANDOM = 100
MAX_ITER = 1000
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
        result["clusters"] = [np.array(x) for x in data]
        return result

    #Initializing clusters and sums
    #For each cluster we store index of the point that we assigned to it
    clusters = [0] * len(data)
    clust = 0
    clusterSize = int(len(data) / k)
    for i in range(len(data)):
        if i >= clusterSize and clust < k - 1:
            clust = clust + 1
            clusterSize = clusterSize + clusterSize
        clusters[i] = clust
    if USE_LOGGING:
        print("---------CLUSTER INDICES--------")
        print(clusters)

    #We store distances between each point in a matrix like structure
    sums = [[round(metric(data.iloc[i], data.iloc[j]), FLOAT_ROUND) for j in range(i+1, len(data))] for i in range(len(data))]
    if USE_LOGGING:
        print("---------DISTANCES--------")
        print(" ", end="")
        for i in range(len(data)):
            print("  " + str(i) + " ", end="")
        print()
        for i in range(len(sums)):
            print("{} ".format(i), end="")
            print(" " * 4 * (i + 1), end="")
            for j in sums[i]:
                print("{:.1f} ".format(j), end="")
            print()

    clusterSums = [0] * k
    for i in range(k):
        cluster = [ ind for ind,j in enumerate(clusters) if j == i]
        for ind in range(len(cluster)):
            for j in range(ind + 1, len(cluster)):
                clusterSums[i] = clusterSums[i] + getDistance(sums, ind, j)
    if USE_LOGGING:
        print("---------CLUSTER LOCAL DISTANCES--------")
        for i in range(len(clusterSums)):
            print("{} => {:.2f}".format(i,clusterSums[i]))

    result['min'] = sum(clusterSums)
    result["history"].append(result["min"])

    #Move elements to another cluster until we can't get a better solution
    betterExists = True

    if USE_LOGGING:
        print("---------BRUTE FORCE MAIN--------")

    currentIteration = 0
    result["clusters"] = clusters

    while betterExists:
        betterExists = False

        if currentIteration > MAX_ITER:
            result["error"] = "Stopped due to large number of iterations"
            break

        currentIteration = currentIteration + 1

        for i in range(len(data)):
            #Make sure its not the only element in the cluster
            if not clusterSums[clusters[i]] > 0:
                continue

            for j in range(k):
                if j == clusters[i]:
                    continue
                prevCluster = clusters[i]
                clusters[i] = j

                prevC = [ind for ind,val in enumerate(clusters) if val == prevCluster]
                newC = [ind for ind,val in enumerate(clusters) if val == clusters[i]]

                prevCSum = 0
                for ind in range(len(prevC)):
                    for jind in range(ind + 1, len(prevC)):
                        prevCSum = prevCSum + getDistance(sums, prevC[ind], prevC[jind])

                newCSum = 0
                for ind in range(len(newC)):
                    for jind in range(ind + 1, len(newC)):
                        newCSum = newCSum + getDistance(sums, newC[ind], newC[jind])

                newMin = result["min"] - clusterSums[prevCluster] - clusterSums[clusters[i]] + prevCSum + newCSum
                if newMin < result["min"] :             
                    if USE_LOGGING:
                        print("FOUND IMPROVEMENT PLACING {} FROM {} => {} [{}]".format(i, prevCluster, clusters[i], currentIteration))

                    result["min"] = newMin
                    result["history"].append(newMin)
                    result["clusters"] = clusters.copy()
                    clusterSums[prevCluster] = prevCSum
                    clusterSums[clusters[i]] = newCSum
                    betterExists = True
                    break
                else:
                    clusters[i] = prevCluster

            if betterExists == True:
                break 
    result["clusters"] = clusters
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
        hist.set_xlabel("iteration")
        hist.set_ylabel("min")
        plt.plot([i for i in range(len(res["history"]))], res["history"])

        plt.subplot(1, 2, 2)
        for i in range(k):
            cluster = [row for row,val in enumerate(res["clusters"]) if val == i]
            plt.scatter(data.iloc[cluster,0], data.iloc[cluster,1], label = i)
        plt.legend(title="clusters")
        plt.show()