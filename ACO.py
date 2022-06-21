import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sko.ACA import ACA_TSP

cities = pd.read_csv('cities.csv', index_col=0, header=0)
# liczba "miast"
N_CITIES = len(cities.index)
num_points = N_CITIES
my_data = genfromtxt('cities.csv', delimiter=',', skip_header=1, usecols=[1, 2])


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


#
allDistances = np.array(calculate_all_distances())


def fitness_func(solution):
    return sum([allDistances[solution[i % num_points], solution[(i + 1) % num_points]] for i in range(num_points)])


#  ACO

import time

aca = ACA_TSP(func=fitness_func, n_dim=num_points,
              size_pop=2, max_iter=200,
              distance_matrix=allDistances)
start = time.time()
best_x, best_y = aca.run()
end = time.time()
czas = end - start

print("Distance", best_y)
print("Czas działania programu", czas)
# %% Plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = my_data[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()
