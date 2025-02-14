import matplotlib.pyplot as plt
import numpy as np

# Example fitness values for each generation
fitness_values = {
    1: [1, 2, 4, 5, 6, 8, 10],
    2: [4, 5, 6, 9, 10, 12],
    # Add more generations as needed
}
def fitnessMetricsAccrossGeneration(fitness_values):
    print("HHHHHHHHHH")
    generations = list(fitness_values.keys())
    best_fitness = []
    avg_fitness = []
    worst_fitness = []
    print(generations)

    # Calculate best, average, and worst fitness for each generation
    for gen in generations:
        values = fitness_values[gen]
        best_fitness.append(max(values))
        avg_fitness.append(np.mean(values))
        worst_fitness.append(min(values))

    # # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label="Best Fitness", marker='o', color='green')
    plt.plot(generations, avg_fitness, label="Average Fitness", marker='o', color='blue')
    plt.plot(generations, worst_fitness, label="Worst Fitness", marker='o', color='red')

    plt.title("Fitness Metrics Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


def newGraph(check):
    
    print("called it" ,check)