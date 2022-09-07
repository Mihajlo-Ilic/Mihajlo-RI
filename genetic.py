import math
from operator import le
from os import path
import random
import time
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


class Unit:
    def __init__(self, n, distances, k, metric):
        self.n = n
        self.distances = distances
        self.k = k
        self.metric = metric

        self.clusters = [random.randint(0, k - 1) for _ in range(n)]
        self.sum = self.calcSum()
        self.fitness = 1.0 / self.sum 
    
    def calcSum(self):
        sum = 0
        for cluster in range(self.k):
            points = [i for i,clust in enumerate(self.clusters) if clust == cluster]
            localSum = 0
            if len(points) > 1:
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        localSum = localSum + getDistance(self.distances, points[i], points[j])
            sum = sum + localSum
        return sum

    def __lt__(self,other): 
        return self.fitness < other.fitness

POPULATION_SIZE = 100
ELITISM_SIZE = POPULATION_SIZE // 5
SELECTION_SIZE = 15
MUTATION_CHANCE = 0.1

def selection(population):
    return max(random.sample(population, SELECTION_SIZE))

def breed(parent1, parent2, population, i, j):
    crossPoint = random.randint(1, len(parent1.clusters) - 1)

    population[i].clusters[:crossPoint] = parent1.clusters[:crossPoint]
    population[j].clusters[:crossPoint] = parent2.clusters[:crossPoint]

    population[i].clusters[crossPoint:] = parent2.clusters[crossPoint:]
    population[j].clusters[crossPoint:] = parent1.clusters[crossPoint:]

def mutate(population, i):
    if random.random() < MUTATION_CHANCE:
        pos = random.randint(0, len(population[i].clusters) - 1)
        population[i].clusters[pos] = random.choice([clust for clust in range(population[i].k) if clust != population[i].clusters[pos]])

def geneticSolution(data: pd.DataFrame , k, metric):
    sums = [[round(metric(data.iloc[i], data.iloc[j]), FLOAT_ROUND) for j in range(i+1, len(data))] for i in range(len(data))]
    population = [Unit(len(data), sums, k, metric) for _ in range(POPULATION_SIZE)]

    result = {}
    result["error"] = ""

    best = max(population)
    result["fitness"] = best.fitness
    result["clusters"] = best.clusters
    result["history"] = [best.fitness]
    result["min"] = best.sum

    for iter in range(MAX_ITER):
        population.sort(reverse=True)

        if result["fitness"] < population[0].fitness:
            result["fitness"] = population[0].fitness
            result["clusters"] = population[0].clusters.copy()
            result["min"] = population[0].sum
        result["history"].append(population[0].fitness)

        for i in range(ELITISM_SIZE, POPULATION_SIZE, 2):
            parent1 = selection(population)
            parent2 = selection(population)

            breed(parent1, parent2, population, i, i + 1)
            mutate(population, i)
            mutate(population, i + 1)

            population[i].sum = population[i].calcSum()
            population[i + 1].sum = population[i + 1].calcSum()
            population[i].fitness = 1.0 / (population[i].sum)
            population[i + 1].fitness =  1.0 / (population[i + 1].sum)
    b = Unit(len(data),sums, k, metric)
    b.clusters = result["clusters"]
    print("NAJBOLJI")
    print(b.clusters)
    print(b.calcSum())
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

    t_start = time.process_time()
    res = geneticSolution(data, k, euclid)
    t_end = time.process_time()

    print('Time took to complete algorithm : {}'.format(t_end - t_start))
    print('Minimal sum : {}'.format(res["min"]))

    if len(data.columns) >= 2:
        hist = plt.subplot(1, 2, 1)
        plt.title("GENETIC ALGORITHM")
        hist.set_xlabel("iteration")
        hist.set_ylabel("fitness")
        plt.plot([i for i in range(len(res["history"]))], res["history"])

        plt.subplot(1, 2, 2)
        for i in range(k):
            cluster = [row for row,val in enumerate(res["clusters"]) if val == i]
            plt.scatter(data.iloc[cluster,0], data.iloc[cluster,1], label = i)
        plt.legend(title="clusters")
        plt.show()