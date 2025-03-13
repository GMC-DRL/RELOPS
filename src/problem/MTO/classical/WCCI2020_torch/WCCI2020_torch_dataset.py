from .WCCI2020_pytorch import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
import numpy as np
from torch.utils.data import Dataset
import torch
import os

class WCCI2020_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dim = 50
        self.task_cnt = 50
        self.N = 10
    
    def __getitem__(self, task_ID):
        dim = self.dim
        task_size = self.task_cnt
        choice_functions = []
        if task_ID == 0:
            choice_functions = [1]
        if task_ID == 1:
            choice_functions = [2]
        if task_ID == 2:
            choice_functions = [4]
        if task_ID == 3:
            choice_functions = [1,2,3]
        if task_ID == 4:
            choice_functions = [4,5,6]
        if task_ID == 5:
            choice_functions = [2,5,7]
        if task_ID == 6:
            choice_functions = [3,4,6]
        if task_ID == 7:
            choice_functions = [2,3,4,5,6]
        if task_ID == 8:
            choice_functions = [2,3,4,5,6,7]
        if task_ID == 9:
            choice_functions = [3,4,5,6,7]

        Tasks = []
        for task_id in range(1, task_size+1):
            id = (task_id-1) % len(choice_functions)
            func_id = choice_functions[id]

            folder_dir = os.path.join(os.path.dirname(__file__),'..','..','..','datafiles','MTO','classical','WCCI2020',f'benchmark_{task_ID+1}')
            shift_file = os.path.join(folder_dir, f'bias_{task_id}')
            rotate_file = os.path.join(folder_dir, f'matrix_{task_id}')

            rotate_matrix = np.loadtxt(rotate_file)
            shift = np.loadtxt(shift_file)
            
            rotate_matrix = torch.tensor(rotate_matrix, dtype=torch.float64)
            shift = torch.tensor(shift, dtype=torch.float64)

            if func_id == 1:
                task = Sphere(dim,shift, rotate_matrix)
                Tasks.append(task)

            if func_id == 2:
                task = Rosenbrock(dim, shift, rotate_matrix)
                Tasks.append(task)

            if func_id == 3:
                task = Ackley(dim, shift, rotate_matrix)
                Tasks.append(task)

            if func_id == 4:
                task = Rastrigin(dim, shift, rotate_matrix)
                Tasks.append(task)

            if func_id == 5:
                task = Griewank(dim, shift, rotate_matrix)
                Tasks.append(task)

            if func_id == 6:
                task = Weierstrass(dim, shift, rotate_matrix)
                Tasks.append(task)

            if func_id == 7:
                task = Schwefel(dim, shift, rotate_matrix)
                Tasks.append(task)

        return Tasks

    def __len__(self):
        return self.N


