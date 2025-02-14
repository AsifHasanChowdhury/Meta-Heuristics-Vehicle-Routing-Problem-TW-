import populationGen as pg
import FitnessFunction as ff
import CrossOver as co
import Mutation as mu
import RouteEncoding as re
import RoutePrint as rp
import FileSystem as fs
import elitistGen as eg
import math
import os
import finalDQN as fdqn
import charts as ch
import io as io
from csv import DictWriter
#popSize <-- desired population size
#individualSize <-- individual size
#n <-- desired number of elite individual






def geneticAlgorithmElitism(pop_size,n,individualSize,nGen, instance_name, unit_cost=1.0,
        init_cost=0, wait_cost=0, delay_cost=0,export_csv=True):
    # print("Hello World")
    # print(f'{BASE_DIR} ssssss')    
    csv_data = []
    list_size = 100
    action_size = 4
    agent = fdqn.DQNAgent(list_size, action_size)


    # # Evaluate the trained model
    # list1 = [1, 1, 1, -4, 5]
    # list2 = [0, 5, -4, -5, -6]
    # best_action, q_values = agent.evaluate(list1, list2)
    # actions = ["SumOfList", "MultiplicationOfList"]
    # print(f"Best Action: {actions[best_action]}")
    # print(f"Q-values: {q_values}")


    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = fs.load_instance(json_file=json_file)
   
    #agent.train_model(500,[1,2,8,4,5],[1,2,6,4,5],instance,unit_cost,init_cost, wait_cost, delay_cost)
    #best_action, q_values = agent.evaluate(parent1, parent1)

    if instance is None:
        print("Please Check Your file path")
        return

    fitness_dict = {}
    population_dict = {}
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
        #print("run: ",i)
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
        sortedPopulation = eg.elitistSelection(population,fitnessList)
        
        selection_count = math.ceil(0.2 * pop_size)

        newPopulation = sortedPopulation[:selection_count]

        # for popSize/2 times do
        pivot1 = 0
        pivot2 = pop_size-1
        

        productionCount = int((pop_size-selection_count)/2)
        # print("prodCount ", productionCount)
        while productionCount > 0:
            
            parent1 = population[pivot1]
            parent2 = population[pivot2]
            

            #Train the agent for crossover
            agent.train_model(2,parent1,parent2,instance,unit_cost,init_cost, wait_cost, delay_cost,1)
            best_action, q_values = agent.evaluate(parent1, parent1,1)
            
            
            #region It might be important block 
            children1 = None
            children2 = None

            if(best_action == 0):
              #print(f'Action 0 {best_action}')
              children1,children2 = co.cx_partially(parent1,parent2)

            elif(best_action == 1):
               #print(f'Action 1 {best_action}')
               children1,children2 = co.order_crossover(parent1,parent2)

        
            
            # mutatedc1 = mu.inverse_mutation(children1)
            # mutatedc2 = mu.inverse_mutation(children2)


            # agent.train_model(50,mutatedc1,mutatedc2,instance,unit_cost,init_cost, wait_cost, delay_cost,2)
            # agent.evaluate(mutatedc1, mutatedc2,2)
            
            # swmutatedc1 = mu.swap_mutation(children1)
            # swmutatedc2 = mu.swap_mutation(children2)


            agent.train_model(2,children1,children2,instance,unit_cost,init_cost, wait_cost, delay_cost,2)
            best_action, q_values = agent.evaluate(children1, children2,2)
            


            if(best_action == 2):
              mutatedc1 = mu.inverse_mutation(children1)
              mutatedc2 = mu.inverse_mutation(children2)
              newPopulation.append(mutatedc1)
              newPopulation.append(mutatedc2)

            elif(best_action == 3):
              swmutatedc1 = mu.swap_mutation(children1)
              swmutatedc1 = mu.swap_mutation(children2)
              newPopulation.append(swmutatedc1)
              newPopulation.append(swmutatedc1)

            # newPopulation.append(children1)
            # newPopulation.append(children2)

            pivot1+=1
            pivot2-=1
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
        
        
        
        
        
        if export_csv:
            csv_row = {
                'generation': i,
                'evaluated_individuals': len("TEST"),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)
        
        
        fitness_dict[i] = fitnessList
        
        
        for gen, genPopulation in enumerate(population, start=0):
            population_dict[gen] = genPopulation
        
        
        population = newPopulation
        
        i+=1
        
    
    
    # print("LEN POP ",len(population))
    print('-- End of (successful) evolution --')
    best_ind = bestIndividual
    print("Total Selected ",len(best_ind))
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {fitnessList[currentbestIndex]}')
    
    
    
    if export_csv:
        print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
        print(f'base dir {BASE_DIR}')
        csv_file_name = f'{instance_name}_uC{unit_cost}_iC{init_cost}_wC{wait_cost}' \
            f'_dC{delay_cost}_pS{pop_size}.csv'
        csv_file = os.path.join(BASE_DIR, 'results', csv_file_name)
        print(f'Write to file: {csv_file}')
        fs.make_dirs_for_file(path=csv_file)
        if not fs.exist(path=csv_file, overwrite=True):
            with io.open(csv_file, 'wt', encoding='utf-8', newline='') as file_object:
                fieldnames = [
                    'generation',
                    'evaluated_individuals',
                    'min_fitness',
                    'max_fitness',
                    'avg_fitness',
                    'std_fitness',
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)
    
    
    
    rp.print_route(re.individual_to_route_decoding(best_ind, instance),instance)
    print(f'Total cost: {1 / fitnessList[currentbestIndex]}')
    ch.fitnessMetricsAccrossGeneration(fitness_dict)
    ch.newGraph(population_dict)
    
    
    
    
    
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
    pop_size = 20
    cx_pb = 0.85
    mut_pb = 0.02
    n_gen = 60
    instance_name = 'RC204'

    # Q Vehicle fuel tank capacity /79.69/
    # C Vehicle load capacity /200.0/
    # r fuel consumption rate /1.0/
    # g inverse refueling rate /3.39/
    # v average Velocity /1.0/

    batteryCapacity = 40


    geneticAlgorithmElitism(pop_size, 1, ind_size, n_gen, instance_name, 
                            unit_cost, init_cost, wait_cost, delay_cost)




#this BASE_DIR is dedicated for base path
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('__file__')))
file= open(fs.filePath(),'r')


if __name__ == '__main__':
    main()