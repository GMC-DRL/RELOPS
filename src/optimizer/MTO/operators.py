import copy
import numpy as np

def DE_mutation(populations):
    # input: pupulations [population_cnt, dim]
    # output: mutants [population_cnt, dim]
    F = 0.5
    pop_cnt, dim = populations.shape
    mutants = copy.deepcopy(populations)
    for j in range(pop_cnt):
        r1 = np.random.randint(low=0, high=pop_cnt)
        r2 = np.random.randint(low=0, high=pop_cnt)
        r3 = np.random.randint(low=0, high=pop_cnt)
        while r1 == j:
            r1 = np.random.randint(low=0, high=pop_cnt)
        while r2 == r1 or r2 == j:
            r2 = np.random.randint(low=0, high=pop_cnt)
        while r3 == r2 or r3 == r1 or r3 == j:
            r3 = np.random.randint(low=0, high=pop_cnt)

        x1 = populations[r1]
        x2 = populations[r2]
        x3 = populations[r3]
        mutant = x1 + F * (x2 - x3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants[j] = mutant

    return mutants

def DE_crossover(mutants, populations):
    CR = 0.7
    U = copy.deepcopy(mutants)
    try:
        population_cnt, dim = mutants.shape
    except ValueError as e:
        print("ValueError occurred:", e)
        print('mutant_shape',mutants.shape)

    #population_cnt, dim = mutants.shape
    for j in range(population_cnt):
        rand_pos = np.random.randint(low=0, high=dim)
        for k in range(dim):
            mutant = mutants[j]
            rand = np.random.rand()
            if rand <= CR or k == rand_pos:
                U[j][k] = mutant[k]

            if rand > CR and k != rand_pos:
                U[j][k] = populations[j][k]
    return U

def DE_rand_1(populations):
    mutants = DE_mutation(populations)
    DE_offsprings = DE_crossover(mutants, populations)
    return DE_offsprings


def mixed_DE(populations, source_pupulations, KT_index, action_2, action_3):
    population_target = populations[KT_index]
    pop_cnt, dim = source_pupulations.shape
    mutants = []
    F = 0.5
    for i in range(population_target.shape[0]):
        r1, r2, r3, r4, r5, r6 = np.random.choice(np.arange(pop_cnt),size=6, replace=False)
        X_r1 = populations[r1]
        X_r2 = source_pupulations[r2]
        X_r3 = populations[r3]
        X_r4 = populations[r4]
        X_r5 = source_pupulations[r5]
        X_r6 = source_pupulations[r6]

        mutant = (1 - action_2) * X_r1 + action_2 * X_r2 + F * (1 - action_3) * (X_r3 - X_r4) + F * action_3 * (
                    X_r5 - X_r6)

        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target)

    return U