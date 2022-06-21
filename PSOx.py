import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sko.PSO import PSO_TSP
import time

my_data = genfromtxt('cities.csv', delimiter=',', skip_header=1, usecols=[1, 2])
cities = pd.read_csv('cities.csv', index_col=0, header=0)

# liczba "miast"
N_CITIES = len(cities.index)


# oblicz odległość pomiędzy dwoma miastami
def calculate_distance(X, Y):
    return np.sqrt((X[1] - Y[1]) ** 2 + (X[0] - Y[0]) ** 2)


# Oblicz odleglos pomiedzy wszystkimi miastami
# Stworz tablice 2-wymiarowa z n-miastami gdzie kazda tablica ma n-elementow ( odleglosci od poszczegolnych miast)
# odejmujemy "-1" aby indeksować tablice miast od 0

def calculate_all_distances():
    array2d = [[0 for _ in range(N_CITIES)] for _ in range(N_CITIES)]
    for i in range(1, N_CITIES + 1):

        for j in range(1, N_CITIES + 1):
            array2d[i - 1][j - 1] = calculate_distance(cities.loc[i], cities.loc[j])
    return array2d


allDistances = np.array(calculate_all_distances())


# # 1 funkcja fitness
def fitness_func(solution):
    return sum([allDistances[solution[i % N_CITIES], solution[(i + 1) % N_CITIES]] for i in range(N_CITIES)])


# druga funkcja fitness
# def fitness_func(solution):
#     dist = 0
#
#     for i in range(N_CITIES - 1):
#         x = int(solution[i] - 1)
#         y = int(solution[i + 1] - 1)
#         dist += allDistances[x][y]
#     first = int(solution[0] - 1)
#     last = int(solution[N_CITIES - 1] - 1)
#     dist += allDistances[first][last]
#     return dist


# PSO

pso_tsp = PSO_TSP(func=fitness_func, n_dim=N_CITIES, size_pop=250, max_iter=100, w=0.8, c1=0.1, c2=0.1)
start = time.time()
best_points, best_distance = pso_tsp.run()
end = time.time()
czas = end - start
print("Czas działania programu", czas)

print('Najlepszy dystans', best_distance)

#  plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = my_data[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(pso_tsp.gbest_y_hist)
plt.show()
