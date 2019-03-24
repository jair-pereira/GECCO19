from multiprocessing import Pool
from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from stats.stats import stats, get_stats
from utilities.stats import trackers
from operators.initialisation import initialisation
from utilities.algorithm.initialise_run import pool_init
import datetime # 190313 timestamp

import numpy as np

def set_M():
    learning = params['LEARNING_METHOD']
    gen      = params['GENERATIONS']
    mult     = params['MULTIPLIER']

    if learning == 'linear':
        a = np.array([mult[0]/(10**i) for i in range(0, mult[1]+1, 1)])
        b = np.arange(0, gen, gen/len(a)) + gen/len(a)
        
        mult = list(zip(a,b))
        m = a[0]
    elif learning == 'adaptative':
        m = mult
    elif learning == 'static':
        m = mult
    else:
        m = 1
            
    params['MULTIPLIER'] = mult
    params['M_aux']      = 0
    params['M']          = m
                
def update_M(gen, individuals):
    import numpy as np
    
    learning = params['LEARNING_METHOD']
    ind      = params['M_aux']
    mult     = params['MULTIPLIER']
    threshold = params['MULT_T']

    if learning == 'linear' and gen >= mult[ind][1]:
        params['M_aux'] = ind + 1
        params['M']     = mult[ind][0]
        
    elif learning == 'adaptative' and \
        np.nanmedian([indv.fitness for indv in individuals]) >= threshold:
        params['M'] /= 2
        
def write_log(file, generation, m, individuals):
    output_list = []
    output_list.append(generation)
    output_list.append(m)
    output_list.append(np.nanmedian([indv.fitness for indv in individuals]))
    for indv in individuals:
        output_list.append(indv.fitness)
    file.write(",".join(map(str,output_list))+"\n")
    file.flush()
    
    return
    
def write_best(generation, m, individuals):
    file = open(params['FILE_PATH']+"/"+str(generation)+".txt", 'w')
    output_list = []
    output_list.append(generation)
    output_list.append(m)
    
    best = max(individuals[1:])
    output_list.append(best.fitness)
    output_list.append(best.phenotype)    
    
    file.write(",".join(map(str,output_list))+"\n")
    file.flush()
    file.close()

def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """
    logf = open(params['FILE_PATH']+"/log.csv", 'w') #190312: log
    set_M()#190307: our mod for learning multiplier
    
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # Generate statistics for run so far
    get_stats(individuals)
    write_log(logf, 0, params['M'], individuals )#190312: log
    #write_best(0, params['M'], individuals)
    
    # Traditional GE
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation

        update_M(generation, individuals) # 190307: our mod for learning multiplier
        
        # New generation
        individuals = params['STEP'](individuals)
        
        write_log(logf, generation, params['M'], individuals)#190312: log
        #write_best(generation, params['M'], individuals)
        print("generation ", generation, "/",params['GENERATIONS'], " finished at ",datetime.datetime.now())# 190313: timestamp

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    logf.close()    
     
    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """
    
    individuals = trackers.state_individuals
    
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)
    
    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        
        # New generation
        individuals = params['STEP'](individuals)
    
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()
    
    return individuals