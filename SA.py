import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from sko.SA import SA_TSP

cities = pd.read_csv('cities.csv', index_col=0, header=0)
# liczba "miast"
N_CITIES = len(cities.index)
points_coordinate = np.genfromtxt('cities.csv', delimiter=',', skip_header=1, usecols=[1, 2])


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

distance_matrix = allDistances  # 1 degree of lat/lon ~ = 111000m


def fitness_func(solution):
    return sum([distance_matrix[solution[i % N_CITIES], solution[(i + 1) % N_CITIES]] for i in range(N_CITIES)])


sa_tsp = SA_TSP(func=fitness_func, x0=range(N_CITIES), T_max=5, T_min=1, L=200)
import time
start = time.time()
best_points, best_distance = sa_tsp.run()
end = time.time()
print("Czas", end - start)
print(best_distance)
#  Plot


fig, ax = plt.subplots(1, 2)

best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(sa_tsp.best_y_history)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Distance")
ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
           marker='o', markerfacecolor='b', color='c', linestyle='-')
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")
plt.show()

#  animacja

best_x_history = sa_tsp.best_x_history

fig2, ax2 = plt.subplots(1, 1)
ax2.set_title('title', loc='center')
line = ax2.plot(points_coordinate[:, 0], points_coordinate[:, 1],
                marker='o', markerfacecolor='b', color='c', linestyle='-')
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.ion()
p = plt.show()


def update_scatter(frame):
    ax2.set_title('iter = ' + str(frame))
    points = best_x_history[frame]
    points = np.concatenate([points, [points[0]]])
    point_coordinate = points_coordinate[points, :]
    plt.setp(line, 'xdata', point_coordinate[:, 0], 'ydata', point_coordinate[:, 1])
    return line


ani = FuncAnimation(fig2, update_scatter, blit=True, interval=25, frames=len(best_x_history))
plt.show()

# ani.save('sa_tsp.gif', writer='pillow')
