from typing import Any
import numpy as np


class MTO_Env:
    """
    An MTO environment with a set of tasks and a set of optimizers.
    """
    def __init__(self,
                 tasks,
                 optimizers,
                 ):
        self.tasks = tasks
        self.optimizers = optimizers
        self.task_cnt = len(tasks)

    def reset(self):
        for i in range(self.task_cnt):
            self.tasks[i].reset()
        
        state_o = self.optimizers[0].generation / self.optimizers[0].total_generation
        states = np.array([state_o], dtype=np.float32)

        for i in range(self.task_cnt):
            state_t = self.optimizers[i].init_population(self.tasks[i])
            states = np.concatenate((states,state_t),axis=-1)
        
        return states

    def step(self, action: Any):
        total_reward = 0
        state_o = self.optimizers[0].generation / self.optimizers[0].total_generation
        next_states = np.array([state_o], dtype=np.float32)

        for i in range(self.task_cnt):
            rand_source_index = np.random.randint(low=0,high=self.task_cnt)
            while rand_source_index == i:
                rand_source_index = np.random.randint(low=0, high=self.task_cnt)
            rand_source_optimizer = self.optimizers[rand_source_index]

            state_t = self.optimizers[i].update(action[int(3*i):int(3*(i+1))], rand_source_optimizer.parent_population)

            next_states = np.concatenate((next_states, state_t), axis=-1)

            total_reward += self.optimizers[i].reward

        is_end = False
        if self.optimizers[0].generation > self.optimizers[0].total_generation:
            is_end = True

        return next_states, total_reward, is_end

