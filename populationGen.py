import numpy as np


def populationGeneration(individual_size,pop_size,seed=50):
    
    if seed is not None:
        np.random.seed(seed)
    
    newPopulation = []

    for i in range (pop_size):
        random_list = np.random.permutation(individual_size) + 1 
        newPopulation.append(random_list.tolist())

    return newPopulation


lst = [1,2,3,4,5,6]

# print(lst[1::2])
# print(lst[::2])
# l = zip(lst[::2],lst[1::2])
# print(l)

# for child1, child2 in zip(lst[::2], lst[1::2]):
#     print(child1)
#     print(child2)


# print(populationGeneration(100,2))