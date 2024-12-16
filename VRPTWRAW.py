import random
import os
import FileSystem as fs
import CrossOver as co
import Mutation as Mu
import RouteEncoding as Re
import EuclidianDistance as Ed
import RoutePrint as Rp
import FitnessFunction as ff
from deap import base, creator, tools



#driver code
def run_vehicleRoutingMainFunction(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, \
    cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
 
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
   
    instance = fs.load_instance(json_file=json_file)
   
    if instance is None:
        print("Please Check Your file path")
        return
    

    creator.create('FitnessMax', base.Fitness, weights=(1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # # Operator registering
    toolbox.register('evaluate', ff.evaluate_individual, instance=instance, unit_cost=unit_cost, \
        init_cost=init_cost, wait_cost=wait_cost, delay_cost=delay_cost)
    # # toolbox.register('select', tools.selRoulette) #FPS
    # toolbox.register('select', tools.selRoulette) #Fitness Proportionate
    # toolbox.register('mate', cx_partially_mapped)
    # toolbox.register('mutate', inverse_mutation)
    pop = toolbox.population(n=pop_size)
    
    print('Start of evolution')
    # Evaluate the entire population
    #print(len(pop))
    fitnesses = list(map(toolbox.evaluate, pop))
    #print("Data ",fitnesses)
   
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    #print(f'  Evaluated {len(pop)} individuals')
    # Begin the evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        print("OFFFFFFSPRING ",offspring)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'  Evaluated {len(invalid_ind)} individuals')
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)


        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
       
    #print(JSONData)
    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print("Total Selected ",len(best_ind))
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    Rp.print_route(Re.individual_to_route_decoding(best_ind, instance),instance)
    print(f'Total cost: {1 / best_ind.fitness.values[0]}')


#main Function
def main():
    '''main()'''
    random.seed(64)

    instance_name = 'C105'

    unit_cost = 8.0
    init_cost = 100.0
    wait_cost = 1.0
    delay_cost = 1.5

    ind_size = 100
    pop_size = 1
    cx_pb = 0.85
    mut_pb = 0.02
    n_gen = 1

    export_csv = True

    run_vehicleRoutingMainFunction(instance_name=instance_name, unit_cost=unit_cost, init_cost=init_cost, \
        wait_cost=wait_cost, delay_cost=delay_cost, ind_size=ind_size, pop_size=pop_size, \
        cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen, export_csv=export_csv)

#this BASE_DIR is dedicated for base path
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('__file__')))
file= open(fs.filePath(),'r')


if __name__ == '__main__':
    main()