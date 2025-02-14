# import populationGen as pg
# import CrossOver as co
# import Mutation as mu
# import populationGen as pg
# # Define individuals
# individuals = [[1, 2, 3, 4, 6, 5], [2, 4, 4, 4, 6]]

# # Define a fitness function (e.g., sum of elements)
# # def calculate_fitness(individual):
# #     return sum(individual)  # Fitness is the sum of the list elements

# # Add fitness to each individual
# # fitness_values = [calculate_fitness(ind) for ind in individuals]

# # Combine individuals with their fitness
# # individuals_with_fitness = list(zip(individuals, fitness_values))

# # # Display results
# # for individual, fitness in individuals_with_fitness:
# #     print(f"Individual: {individual}, Fitness: {fitness}")

# population = pg.populationGeneration(100,20)
# print(len(population))
# newPopulation = []

# # for popSize/2 times do
# pivot1 = 0
# pivot2 = 20-1
# #print("beforeprodCount ", pop_size)

# productionCount = int(20/2)
# # print("prodCount ", productionCount)
# while productionCount > 0:
    
#     parent1 = population[pivot1]
#     parent2 = population[pivot2]

#     pivot1+=1
#     pivot2-=1

#     children1, children2 = co.cx_partially(parent1,parent2)
#     # print("Children 1",children1)
#     # print("Children 2",children2)
#     mutatedc1 = mu.inverse_mutation(children1)
#     mutatedc2 = mu.inverse_mutation(children2)

#     newPopulation.append(mutatedc1)
#     newPopulation.append(mutatedc2)

#     productionCount-=1

# print("NEW POPULATION ",len(newPopulation))


# lst=[50, 51, 97, 74, 25, 1, 53, 72, 91, 65, 49, 82, 89, 80, 16, 62, 96, 19, 77, 39, 6, 28, 78, 32, 42, 56, 66, 30, 36, 99, 27, 33, 94, 63, 98, 73, 20, 18, 24, 85, 5, 87, 71, 67, 22, 55, 23, 21, 58, 83, 12, 45, 95, 4, 92, 15, 44, 100, 61, 14, 70, 26, 69, 9, 93, 54, 17, 38, 57, 11, 64, 13, 90, 31, 3, 34, 2, 52, 37, 46, 88, 29, 43, 75, 48, 41, 10, 86, 84, 8, 40, 76, 79, 81, 60, 35, 59, 68, 7, 47]
# cars = [4,2,1,5]

# lst.sort()

# print(lst)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.
plt.show()                           # Show the figure.



fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.
plt.show()   