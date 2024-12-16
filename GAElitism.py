import populationGen as pg
import FitnessFunction as ff
import CrossOver as co
import Mutation as mu
import RouteEncoding as re
import RoutePrint as rp
import FileSystem as fs
import os

#popSize <-- desired population size
#individualSize <-- individual size
#n <-- desired number of elite individual






def geneticAlgorithmElitism(pop_size,n,individualSize,nGen, instance_name, unit_cost=1.0,
        init_cost=0, wait_cost=0, delay_cost=0):
    
    
    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = fs.load_instance(json_file=json_file)
    #print(instance)
    if instance is None:
        print("Please Check Your file path")
        return


    #population <-- {}
    population = pg.populationGeneration(individualSize,pop_size)
    #print("L",population)

    #best <-- []
    bestIndividual = None
    bestIndividualFitness = None
    i=0
    currentbestIndex = 0
    
    # #Repeat Until Loop
    while i < nGen:
        print("run: ",i)
        fitnessList = []
        #for each individual Pi belongs to P do
        index = 0
        for singlePeople in population:
             
            #AssessFitness(Pi)
            fitnesses = ff.evaluate_individual(singlePeople, instance, unit_cost,init_cost, wait_cost, delay_cost)
            
            fitnessList.append(fitnesses)
            
            # print("fitness ",fitnesses)
            # print("bestInd ",bestIndividual)
            #if best is empty or Fitness(Pi)>Fitness(best) then
            # print("Fitnesss ",fitnesses)
            if bestIndividual is None or fitnessList[index] > bestIndividualFitness:
                #best <-- Pi
                bestIndividual = singlePeople
                bestIndividualFitness = fitnessList[index]
                

            index+=1   

        # Q <-- {}
        newPopulation = []

        # for popSize/2 times do
        pivot1 = 0
        pivot2 = pop_size-1
        

        productionCount = int(pop_size/2)
        # print("prodCount ", productionCount)
        while productionCount > 0:
            
            parent1 = population[pivot1]
            parent2 = population[pivot2]

            pivot1+=1
            pivot2-=1

            children1, children2 = co.cx_partially(parent1,parent2)
            
            mutatedc1 = mu.inverse_mutation(children1)
            mutatedc2 = mu.inverse_mutation(children2)

            newPopulation.append(mutatedc1)
            newPopulation.append(mutatedc2)

            productionCount-=1
        # np = newPopulation.sort() 
        # print("NEW POPULATION ",np)
        
        
        # fits = [ind.fitness.values[0] for ind in population]
        fits = fitnessList
        length = pop_size


        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')

        population = newPopulation
        
        i+=1
        
    
    
    # print("LEN POP ",len(population))
    print('-- End of (successful) evolution --')
    best_ind = bestIndividual
    print("Total Selected ",len(best_ind))
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {fitnessList[currentbestIndex]}')
    rp.print_route(re.individual_to_route_decoding(best_ind, instance),instance)
    print(f'Total cost: {1 / fitnessList[currentbestIndex]}')


def selectBestFit(population):
    bestFit = None
    for singlePeople in population:
        if(bestFit is None or singlePeople.fitness.values):
            bestFit = singlePeople

    return bestFit


def main():
            
    unit_cost = 8.0
    init_cost = 100.0
    wait_cost = 1.0
    delay_cost = 1.5

    ind_size = 100
    pop_size = 100
    cx_pb = 0.85
    mut_pb = 0.02
    n_gen = 10000
    instance_name = 'C107'

    geneticAlgorithmElitism(pop_size, 1, ind_size, n_gen, instance_name, 
                            unit_cost, init_cost, wait_cost, delay_cost)




#this BASE_DIR is dedicated for base path
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('__file__')))
file= open(fs.filePath(),'r')


if __name__ == '__main__':
    main()