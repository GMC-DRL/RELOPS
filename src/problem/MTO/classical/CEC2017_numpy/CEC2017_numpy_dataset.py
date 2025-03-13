from .CEC2017_np import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
import numpy as np
from torch.utils.data import Dataset
import os
import scipy.io as sio

class CEC2017_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.task_cnt = 2
        self.N = 9
    
    def mat2np(self, path):
        data = sio.loadmat(path)
        return data
    
    def __getitem__(self, task_ID):
        folder_dir = os.path.join(os.path.dirname(__file__),'..','..','..','datafiles','MTO','classical','CEC2017')

        Tasks = []
        if task_ID == 0:
            file_dir = os.path.join(folder_dir,'CI_H.mat')
            keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
            data = self.mat2np(file_dir)
            task1 = Griewank(50, data[keys[0]], data[keys[2]])
            task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
            Tasks = [task1, task2]

        if task_ID == 1:
            file_dir = os.path.join(folder_dir,'CI_M.mat')
            keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
            data = self.mat2np(file_dir)
            task1 = Ackley(50, data[keys[0]], data[keys[2]])
            task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
            Tasks = [task1, task2]

        if task_ID == 2:
            file_dir = os.path.join(folder_dir,'CI_L.mat')
            keys = ['GO_Task1', None, 'Rotation_Task1',None]
            data = self.mat2np(file_dir)
            task1 = Ackley(50, data[keys[0]], data[keys[2]])
            task2 = Schwefel(50)
            Tasks = [task1, task2]

        if task_ID == 3:
            file_dir = os.path.join(folder_dir,'PI_H.mat')
            keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
            data = self.mat2np(file_dir)
            task1 = Rastrigin(50, data[keys[0]], data[keys[2]])
            task2 = Sphere(50, data[keys[1]])
            Tasks = [task1, task2]

        if task_ID == 4:
            file_dir = os.path.join(folder_dir,'PI_M.mat')
            keys = ['GO_Task1',None, 'Rotation_Task1', None]
            data = self.mat2np(file_dir)
            task1 = Ackley(50, data[keys[0]], data[keys[2]])
            task2 = Rosenbrock(50)
            Tasks = [task1, task2]

        if task_ID == 5:
            file_dir = os.path.join(folder_dir,'PI_L.mat')
            keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
            data = self.mat2np(file_dir)
            task1 = Ackley(50, data[keys[0]], data[keys[2]])
            task2 = Weierstrass(25, data[keys[1]], data[keys[3]])
            Tasks = [task1, task2]

        if task_ID == 6:
            file_dir = os.path.join(folder_dir,'NI_H.mat')
            keys = [None, 'GO_Task2', None, 'Rotation_Task2']
            data = self.mat2np(file_dir)
            task1 = Rosenbrock(50)
            task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
            Tasks = [task1, task2]
           
        if task_ID == 7:
            file_dir = os.path.join(folder_dir,'NI_M.mat')
            keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
            data = self.mat2np(file_dir)
            task1 = Griewank(50, data[keys[0]], data[keys[2]])
            task2 = Weierstrass(50, data[keys[1]], data[keys[3]])
            Tasks = [task1, task2]

        if task_ID == 8:
            file_dir = os.path.join(folder_dir,'NI_L.mat')
            keys = ['GO_Task1',None, 'Rotation_Task1',None]
            data = self.mat2np(file_dir)
            task1 = Rastrigin(50, data[keys[0]], data[keys[2]])
            task2 = Schwefel(50)
            Tasks = [task1, task2]

        return Tasks

    def __len__(self):
        return self.N


