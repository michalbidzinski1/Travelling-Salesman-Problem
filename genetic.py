import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygad

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
    array2d = [[0 for i in range(N_CITIES)] for y in range(N_CITIES)]
    for i in range(1, N_CITIES + 1):

        for j in range(1, N_CITIES + 1):
            array2d[i - 1][j - 1] = calculate_distance(cities.loc[i], cities.loc[j])
    return array2d


# allDistances = calculate_all_distances()
# do drugiej funckji fitness
allDistances = np.array(calculate_all_distances())


# tablica 2 wymiarowa z n-miastami o n-elementach ( elementy to odleglosci od miast)


# def fitness_func(solution, solution_idx):
#     return 1 / sum([allDistances[int(solution[i % N_CITIES] - 1), int(solution[(i + 1) % N_CITIES] - 1)] for i in
#                     range(N_CITIES)])


#
# def fitness_func(solution, solution_idx):
#     distance = 0
#     for i in range(N_CITIES - 1):
#         distance += allDistances[int(solution[i] - 1)][int(solution[i + 1] - 1)]
#     distance += allDistances[int(solution[0]) - 1][int(solution[N_CITIES - 1] - 1)]  # ostatnie i pierwsze miasto
#     return 1 / distance

# zdefiniowanie funkcji fitness
# odejmujemy "-1" aby indeksować tablice  od 0
# im wyższy fitness tym lepiej
def fitness_func(solution, solution_idx):
    dist = 0
    for i in range(N_CITIES - 1):
        x = int(solution[i] - 1)
        y = int(solution[i + 1] - 1)
        dist += allDistances[x][y]
    first = int(solution[0] - 1)
    last = int(solution[N_CITIES - 1] - 1)
    dist += allDistances[first][last]
    return 1 / dist


def on_generation(g):
    print("Generation:", g.generations_completed, "\tDistance:",
          round(1 / g.last_generation_fitness[0], 4))

    # po "x" iteracjach bez zmian w wyliczanej najkrótszej odległośći zatrzymujemy alogrytm.  "reach_criteria" w pygad
    x = 100
    if stop_criteria(g, x):
        return "stop"


# po "x" iteracjach bez zmian w wyliczanej najkrótszej odległośći zatrzymujemy alogrytm.  "reach_criteria" w pygad
# generations_completed: Holds the number of the last completed generation.
# best_solutions_fitness: A list holding the fitness values of the best solutions for all generations.
def stop_criteria(g, x):
    if g.generations_completed > x:
        last_index = g.generations_completed - 1
        # bieżąca wartość fitness, którą porównujemy z "x" wcześniejsza wartościa fitness i sprawdzamy, czy doszło
        # do jakiś zmian
        last_fitness = g.best_solutions_fitness[last_index]

        # wartosć fitness od której liczymy ilość iteracji bez żadnych zmian
        x_back_fitness = g.best_solutions_fitness[last_index - x]
        if math.isclose(last_fitness, x_back_fitness, abs_tol=0.00003):
            return True
    return False


# wypisanie ostatecznego dystansu jaki musiał przebyc np. kurier
def print_distance(g):
    dist = fitness_func(g, 1)
    print("Final distance: ", 1 / dist)


# Droga jako graf

def print_route(s):
    x_coordinates = []
    y_coordinates = []
    for i in range(N_CITIES + 1):  # dodajemy pierwsze miasto na koniec
        city = cities.loc[s[i]]
        x_coordinates.append(city[0])
        y_coordinates.append(city[1])
    plt.scatter(x_coordinates, y_coordinates)
    plt.plot(x_coordinates, y_coordinates)
    plt.xlabel(" X ")
    plt.ylabel(" Y")
    plt.title("Best  Route")


def ga():
    fitness_function = fitness_func

    num_generations = 100

    sol_per_pop = 25

    num_genes = N_CITIES

    gene_space = range(1, N_CITIES + 1)

    mating_percent = 25
    num_parents_mating = math.ceil(sol_per_pop * mating_percent / 100)

    keep_percent = 5
    keep_parents = math.ceil(sol_per_pop * keep_percent / 100)

    parent_selection_type = "sss"

    crossover_type = "single_point"

    mutation_type = "adaptive"

    mutation_percent_genes = [20, 5]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_space=gene_space,
                           on_generation=on_generation,
                           allow_duplicate_genes=False)
    start = time.time()
    ga_instance.run()
    end = time.time()
    czas = end - start
    print("Czas działania programu", czas)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    s_print = solution
    s_print = np.append(s_print, s_print[0])

    print("Best Route:\n", s_print, "\n")
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    ga_instance.plot_fitness()
    print_route(s_print)
    plt.show()
    return solution


print("\n")
print_distance(ga())
print("\n")
