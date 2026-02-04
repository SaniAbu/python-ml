from matplotlib import pyplot as plt
import random
from prettytable import PrettyTable
import subprocess
import os
import re
import psutil
import time

MAX_PIPELINE_WIDTH = 23
MIN_PIPELINE_WIDTH = 4
MIN_ISSUE_WIDTH = 9
MIN_EXP = 4
MAX_EXP = 10
MIN_BTB = 2**7
MIN_RAS = 2**4

memo = {}

def fitness_function(name, pipeline_width, issue_wb_pipeline_width, rob_size, iq_size, lq_size, sq_size, lsq_size, local_pred_size, global_pred_size, btb_size, ras_size, l1_data_size, l1_inst_size, l2_size):
    mininum_phys_reg_size = 49
    if rob_size < mininum_phys_reg_size:
      phys_regs = mininum_phys_reg_size
    else:
      phys_regs = rob_size

    subprocess.run("python /homes/lp721/aca-gem5/simulate.py --window-size "+str(rob_size)+","+str(iq_size)+","+str(lsq_size)+" --branch-pred-size "+str(local_pred_size)+","+str(global_pred_size)+","+str(btb_size)+","+str(ras_size)+" --pipeline-width "+str(pipeline_width)+" --issue-wb-pipeline-width "+str(issue_wb_pipeline_width)+" --lq-size "+str(lq_size)+" --sq-size "+str(sq_size)+" --l1-data-size "+str(l1_data_size)+" --l1-inst-size "+str(l1_inst_size)+" --l2-size "+str(l2_size)+" --num-int-phys-regs "+str(phys_regs)+" --num-float-phys-regs "+str(phys_regs)+" --num-vec-phys-regs "+str(phys_regs)+" --name "+name, shell=True)
    
    filepath = os.path.join(name, 'results')

    # Initialize dictionary to store values
    results = {}

    # Define patterns for each value you want to extract
    patterns = {
        'Simulated seconds': r'(?i)Simulated\s*seconds\s*=\s*([\d\.Ee+-]+)',
        'Subthreshold Leakage': r'(?i)Subthreshold\s*Leakage\s*=\s*([\d\.Ee+-]+)',
        'Gate Leakage': r'(?i)Gate\s*Leakage\s*=\s*([\d\.Ee+-]+)',
        'Runtime Dynamic': r'(?i)Runtime\s*Dynamic\s*=\s*([\d\.Ee+-]+)',
    }

    # Read and parse file
    with open(filepath, 'r') as f:
        contents = f.read()

    # Extract each value
    for key, pattern in patterns.items():
        match = re.search(pattern, contents)
        if match:
            results[key] = float(match.group(1))
        else:
            results[key] = None  # or raise an error if required

    # Return all four values (in a tuple, for convenience)
    energy = - (results['Runtime Dynamic'] + results['Subthreshold Leakage'] + results['Gate Leakage']) * results['Simulated seconds']
    memo[name] = energy
    return energy

def create_initial_population(size, lower_bound, upper_bound):
    population = []
    for i in range(size):
        individual = (f"initial_{i}", *[random.randint(lower_bound, upper_bound) for i in range(7)], *[2 ** random.randint(MIN_EXP, MAX_EXP) for i in range(7)])
        individual = list(individual)
        if individual[1] > MAX_PIPELINE_WIDTH:
            individual[1] = MAX_PIPELINE_WIDTH
        if individual[1] < MIN_PIPELINE_WIDTH:
            individual[1] = MIN_PIPELINE_WIDTH
        if individual[2] > MAX_PIPELINE_WIDTH:
            individual[2] = MAX_PIPELINE_WIDTH
        if individual[2] < MIN_PIPELINE_WIDTH:
            individual[2] = MIN_PIPELINE_WIDTH

        if individual[10] < MIN_BTB:
            individual[10] = MIN_BTB

        if individual[11] < MIN_RAS:
            individual[11] = MIN_RAS

        population.append(tuple(individual))
    return population

# Selection function using tournament selection
def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Crossover function
def crossover(parent1, parent2, child_index):
    genes1 = parent1[1:]
    genes2 = parent2[1:]

    assert len(genes1) == len(genes2), "Parents must have same number of genes"

    child1_genes = []
    child2_genes = []

    for g1, g2 in zip(genes1, genes2):
        # For each gene, 50% chance of taking from one parent
        if random.random() < 0.5:
            child1_genes.append(g1)
            child2_genes.append(g2)
        else:
            child1_genes.append(g2)
            child2_genes.append(g1)

    child1_name = "child_a_" + child_index
    child2_name = "child_b_" + child_index

    child1 = (child1_name, *child1_genes)
    child2 = (child2_name, *child2_genes)

    return child1, child2

def mutation(individual, mutation_rate, lower_bound, upper_bound):
    individual = list(individual)
    for i in range(1, 8):
        if random.random() < mutation_rate:
            mutation_amount = random.randint(-128, 128)
            individual[i] += mutation_amount
            # mutation_amount = random.randint(1, 2)
            # if mutation_amount == 1:
            #     individual[i] = int(individual[i] * 2)
            # else:
            #     individual[i] = int(individual[i] / 2)
            # Ensure the individual stays within bounds
            individual[i] = max(min(individual[i], upper_bound), lower_bound)

            if individual[1] > MAX_PIPELINE_WIDTH:
                individual[1] = MAX_PIPELINE_WIDTH
            if individual[1] < MIN_PIPELINE_WIDTH:
                individual[1] = MIN_PIPELINE_WIDTH
            if individual[2] > MAX_PIPELINE_WIDTH:
                individual[2] = MAX_PIPELINE_WIDTH
            if individual[2] < MIN_PIPELINE_WIDTH:
                individual[2] = MIN_PIPELINE_WIDTH
    
    for i in range(8, len(individual)):
        if random.random() < mutation_rate:
            mutation_amount = random.randint(1, 2)
            if mutation_amount == 1:
                individual[i] = int(individual[i] * 2)
            else:
                individual[i] = int(individual[i] / 2)
            # Ensure the individual stays within bounds
            individual[i] = max(min(individual[i], 2 ** MAX_EXP), 2 ** MIN_EXP)

    if individual[10] < MIN_BTB:
        individual[10] = MIN_BTB 
    
    if individual[11] < MIN_RAS:
        individual[11] = MIN_RAS
            
    return tuple(individual)

def genetic_algorithm(population_size, lower_bound, upper_bound, generations, mutation_rate):
    population = create_initial_population(population_size, lower_bound, upper_bound)
    
    # Prepare for plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows, 1 column for subplots
    best_performers = []
    all_populations = []

    # Prepare for table
    table = PrettyTable()
    table.field_names = ["Generation", "name", "pipeline_width", "issue_wb_pipeline_width", "rob_size", "iq_size", "lq_size", "sq_size", "lsq_size", "local_pred_size", "global_pred_size", "btb_size", "ras_size", "l1_data_size", "l1_inst_size", "l2_size", "Fitness"]

    for generation in range(generations):
        fitnesses = [fitness_function(*ind) for ind in population]

        # Store the best performer of the current generation
        best_individual = max(population, key=lambda ind: memo[ind[0]])
        best_fitness = memo[best_individual[0]]
        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        table.add_row([generation + 1, best_individual[0], best_individual[1], best_individual[2], best_individual[3], best_individual[4], best_individual[5], best_individual[6], best_individual[7], best_individual[8], best_individual[9], best_individual[10], best_individual[11], best_individual[12], best_individual[13], best_individual[14], best_fitness])

        population = selection(population, fitnesses)

        next_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            child1, child2 = crossover(parent1, parent2, "gen_"+str(generation+1)+"_i_"+str(i))

            next_population.append(mutation(child1, mutation_rate, lower_bound, upper_bound))
            next_population.append(mutation(child2, mutation_rate, lower_bound, upper_bound))

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population

        print(population)

    # Print the table
    print(table)

    # # Plot the population of one generation (last generation)
    # final_population = all_populations[-1]
    # final_fitnesses = [memo[ind[0]] for ind in final_population]

    # axs[0].scatter(range(len(final_population)), [ind[0] for ind in final_population], color='blue', label='a')
    # axs[0].scatter([final_population.index(best_individual)], [best_individual[0]], color='cyan', s=100, label='Best Individual a')
    # axs[0].set_ylabel('a', color='blue')
    # axs[0].legend(loc='upper left')
    
    # axs[1].scatter(range(len(final_population)), [ind[1] for ind in final_population], color='green', label='b')
    # axs[1].scatter([final_population.index(best_individual)], [best_individual[1]], color='magenta', s=100, label='Best Individual b')
    # axs[1].set_ylabel('b', color='green')
    # axs[1].legend(loc='upper left')
    
    # axs[2].scatter(range(len(final_population)), [ind[2] for ind in final_population], color='red', label='c')
    # axs[2].scatter([final_population.index(best_individual)], [best_individual[2]], color='yellow', s=100, label='Best Individual c')
    # axs[2].set_ylabel('c', color='red')
    # axs[2].set_xlabel('Individual Index')
    # axs[2].legend(loc='upper left')
    
    # axs[0].set_title(f'Final Generation ({generations}) Population Solutions')

    # # Plot the values of a, b, and c over generations
    # generations_list = range(1, len(best_performers) + 1)
    # a_values = [ind[0][0] for ind in best_performers]
    # b_values = [ind[0][1] for ind in best_performers]
    # c_values = [ind[0][2] for ind in best_performers]
    # fig, ax = plt.subplots()
    # ax.plot(generations_list, a_values, label='a', color='blue')
    # ax.plot(generations_list, b_values, label='b', color='green')
    # ax.plot(generations_list, c_values, label='c', color='red')
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Parameter Values')
    # ax.set_title('Parameter Values Over Generations')
    # ax.legend()

    # Plot the fitness values over generations
    # best_fitness_values = [fit[1] for fit in best_performers]
    # min_fitness_values = [min([memo[ind[0]] for ind in population]) for population in all_populations]
    # max_fitness_values = [max([memo[ind[0]] for ind in population]) for population in all_populations]
    # fig, ax = plt.subplots()
    # ax.plot(generations_list, best_fitness_values, label='Best Fitness', color='black')
    # ax.fill_between(generations_list, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Fitness')
    # ax.set_title('Fitness Over Generations')
    # ax.legend()

    # plt.show()
    print(memo)
    return best_individual

population_size = 8
lower_bound = 16
upper_bound = 2048
generations = 10
mutation_rate = 1

# Run the genetic algorithm
best_solution = genetic_algorithm(population_size, lower_bound, upper_bound, generations, mutation_rate)
print(f"Best solution found: name = {best_solution[0]}, other variables = {best_solution[1:]}, energy = {memo[best_solution[0]]}")