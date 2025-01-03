import random

def inverse_mutation(individual):
    
    start, stop = sorted(random.sample(range(len(individual)), 2))
    temp = individual[start:stop+1]
    temp.reverse()
    individual[start:stop+1] = temp
    return individual


def swap_mutation(individual):
    
    size = len(individual)
    
    # Randomly select two positions to swap
    pos1, pos2 = random.sample(range(size), 2)
    
    # Swap the values at the selected positions
    mutated_chromosome = individual[:]
    mutated_chromosome[pos1], mutated_chromosome[pos2] = mutated_chromosome[pos2], mutated_chromosome[pos1]
    
    return mutated_chromosome



print(swap_mutation([1,2,3,4,5,6,7,8,9]))