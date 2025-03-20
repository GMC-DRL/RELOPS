import numpy as np
import torch
import copy
from .learnable_optimizer import Learnable_Optimizer
from .operators import DE_rand_1, mixed_DE

class L2O_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.dim = config.dim
        self.generation = 0
        self.pop_cnt = 100
        self.total_generation = 250 # need to add in config
        self.flag_improved = 0
        self.stagnation = 0 
        self.old_action_1 = 0
        self.old_action_2 = 0
        self.old_action_3 = 0
        self.N_kt = 0
        self.Q_kt = 0
        self.gbest = 1e+32

        self.task = None
        self.offsprings = None
        self.noKT_offsprings = None
        self.KT_offsprings = None
        self.KT_index = None
        self.parent_population = None
        self.reward = 0

    def get_state(self):
        states = []
        state_1 = self.stagnation / self.total_generation
        states.append(state_1)
        state_2 = self.flag_improved
        states.append(state_2)
        if self.N_kt == 0:
            state_3 = 0
        else:
            state_3 = self.Q_kt

        states.append(state_3)
        state_4 = np.mean(np.std(self.parent_population, axis=-1))
        states.append(state_4)
        state_5 = self.old_action_1
        state_6 = self.old_action_2
        state_7 = self.old_action_3

        states.append(state_5)
        states.append(state_6)
        states.append(state_7)

        self.generation += 1

        return np.array(states, dtype=np.float32)

    def init_population(self, task):
        self.parent_population = np.array([np.random.rand(self.dim) for i in range(self.pop_cnt)])
        self.task = task

        fitnesses = task.eval(self.parent_population)
        self.gbest = np.min(fitnesses, axis=-1)
        
        state = self.get_state()
        return state

    def self_update(self):
         self.noKT_offsprings = DE_rand_1(self.parent_population)
    

    def transfer(self,actions, source_population):
        action_1 = actions[0]
        action_2 = actions[1]
        action_3 = actions[2]

        self.N_kt = 0.5 * action_1
        self.KT_count = int(np.ceil(self.N_kt * self.pop_cnt))
        if self.KT_count == 0:
            self.KT_count = 1
        self.KT_index = np.random.choice(np.arange(self.pop_cnt), size=self.KT_count, replace=False)

        self.KT_offsprings = mixed_DE(self.parent_population, source_population, self.KT_index, action_2, action_3)
        self.offsprings = copy.deepcopy(self.noKT_offsprings)
        for i in range(self.KT_count):
            self.offsprings[self.KT_index[i]] = self.KT_offsprings[i]

        self.old_action_1 = action_1
        self.old_action_2 = action_2
        self.old_action_3 = action_3

    def seletion(self):
        parent_population_fitness = self.task.eval(self.parent_population)
        offsprings_population_fitness = self.task.eval(self.offsprings)

        next_population = copy.deepcopy(self.parent_population)

        S_update = 0
        S_KT = 0
        for i in range(self.pop_cnt):
            if offsprings_population_fitness[i] <= parent_population_fitness[i]:
                if i not in self.KT_index:
                    S_update += 1
                else:
                    S_KT += 1

                next_population[i] = self.offsprings[i]
            else:
                next_population[i] = self.parent_population[i]

        self.reward = (float)(S_update-S_KT) / self.pop_cnt
        self.Q_kt = float(S_KT) / self.KT_count

        flag = 0
        fitnesses = self.task.eval(next_population)
        best_fitness = np.min(fitnesses,axis=-1)
        if(best_fitness < self.gbest):
            self.gbest = best_fitness
            flag = 1

        if(flag):
            self.flag_improved = 1
        else:
            self.flag_improved = 0
            self.stagnation += 1

        self.parent_population = next_population

        return self.get_state()

    def update(self, actions, source_population):
        self.self_update()
        self.transfer(actions, source_population)
        state = self.seletion()
        return state