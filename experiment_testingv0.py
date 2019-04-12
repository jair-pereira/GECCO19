import numpy as np
import pandas as pd
import sys, pickle
import cocoex, cocopp
#from src.algorithm import parameters as ponyge

def process_training_log():
    #load data
    data_log = pd.read_csv("./results/"+experiment_name+"/log.csv", header=None)

    #header
    header = ["GENERATIONS", "PRECISION", "FIT_MEDIAN"]
    for i in range(1, data_log.shape[1] - 2):
        header.append("FIT_INDV_"+str(i))
    data_log.columns = header
    
    return data_log

def get_best_indv():
    pdlog = process_training_log()
    filter_precision = pdlog[pdlog['PRECISION'] == pdlog['PRECISION'].min()]
    
    generation = -1
    fitness = 0
    for i, row in filter_precision.filter(regex="FIT_INDV").iterrows():
        bestfitness = row.max()
    
        if bestfitness >= fitness:
            generation = i
            fitness    = bestfitness
            
    print("Best individual at generation :", generation, "(",experiment_name,")")
    return generation

def write_log(d_fitness, result):
    file = open("./results/"+experiment_name+"/_testing.txt", 'w')
    
    header = []
    header.append("hh_fit")
    for key in d_fitness.keys():
        header.append(key)
    file.write(",".join(map(str,header))+"\n")
    
    data = []
    data.append(result)
    for val in d_fitness.values():
        data.append(val)
    file.write(",".join(map(str,data))+"\n")
                
    file.flush()
    file.close()
    
    return
                
def experiment_test(experiment_name, max_nfe, precision, suite):
    #bbob optimums
    file = open("src/bbob_final_target_fvalue1.pkl",'rb')
    ftarget_values = pickle.load(file)
    file.close()

    #extract code
    #ponyge.load_params("./results/"+experiment_name+"/parameters.txt")
    file = open("./results/"+experiment_name+"/"+str(get_best_indv())+".txt", 'r')
    best = file.readlines()
    file.close()

    code = "import numpy as np\nfrom src.src.solution import Solution\nimport src.src.operators as op\n"
    for line in best[6:-8]:    code += line
                
    #run code on each problem in suite
    d_target_hit = {}
    d_fitness    = {}
    for problem in suite:
        d = {
            "max_nfe"  : max_nfe, 
            "dimension": problem.dimension,
            "my_func"  : problem,
            "bounds"   : (problem.lower_bounds[0], problem.upper_bounds[0])
            }

        exec(code, d)

        d_fitness[problem.id] = d['XXX_output_XXX']
        d_target_hit[problem.id] = 0
        if np.abs(d_fitness[problem.id] - ftarget_values[problem.id]) <= precision:
            d_target_hit[problem.id] = 1

    result = sum(d_target_hit.values()) / len(suite)

    #write results to a file
    write_log(d_fitness, result)
    
    
if __name__ == "__main__":
    precision   = float(sys.argv[1])
    max_nfe     = int(sys.argv[2])
    experiments = sys.argv[3:]
    suite = cocoex.Suite("bbob", "", "function_indices:1,15 dimensions:20,40 instance_indices:1-10")
    
    for experiment_name in experiments:
        experiment_test(experiment_name, max_nfe, precision, suite)