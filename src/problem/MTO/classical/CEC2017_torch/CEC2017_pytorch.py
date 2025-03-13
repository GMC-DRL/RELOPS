from .basic_problem import Basic_Problem
import numpy as np
import time 
import torch

class CEC2017_Basic_Problem(Basic_Problem):
    def __init__(self, dim, shift, rotate, bias):
        self.T1 = 0
        self.dim = dim
        self.shift = shift if shift is None else torch.tensor(shift, dtype=torch.float64)
        self.rotate = rotate if rotate is None else torch.tensor(rotate, dtype=torch.float64)
        self.bias = bias
        self.lb = -50
        self.ub = 50
        self.FES = 0

        self.opt = self.shift if self.shift is not None else torch.zeros(size=(self.dim,))
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError
    
    def decode(self, x):
        return x * (self.ub - self.lb) + self.lb

    def sr_func(self, x, shift, rotate):
        if shift is not None: 
            y = x - shift
        else:
            y = x
        
        if rotate is not None:
            z = torch.matmul(rotate, y.transpose(0,1)).transpose(0,1)
        else:
            z = y 
        return z
    
    def eval(self, x):
        """
        A specific version of func() with adaptation to evaluate both individual and population in MTO.
        """
        start=time.perf_counter()
        x = self.decode(x)  # the solution in MTO is constrained in a unified space [0,1]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            x = x[:self.dim]
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            x = x[:, :self.dim]
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            x = x[:,:,:self.dim]
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

class Sphere(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2, -1)

class Ackley(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        sum1 = -0.2 * torch.sqrt(torch.sum(z ** 2, -1) / self.dim)
        sum2 = torch.sum(torch.cos(2 * torch.pi * z), -1) / self.dim
        return torch.round(torch.e + 20 - 20 * torch.exp(sum1) - torch.exp(sum2), decimals = 15) + self.bias
    

class Griewank(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        s = torch.sum(z ** 2, -1)
        p = torch.ones(x.shape[0])
        for i in range(self.dim):
            p *= torch.cos(z[:, i] / torch.sqrt(torch.tensor(1 + i)))
        return 1 + s / 4000 - p + self.bias

class Rastrigin(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        return torch.sum(z ** 2 - 10 * torch.cos(2 * torch.pi * z) + 10, -1) + self.bias
    
class Rosenbrock(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        z += 1
        z_ = z[:, 1:]
        z = z[:, :-1]
        tmp1 = z ** 2 - z_
        return torch.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1) + self.bias


class Weierstrass(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -0.5
        self.ub = 0.5

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a, b, k_max = torch.tensor(0.5), torch.tensor(3.0), torch.tensor(20)
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += torch.sum(torch.pow(a, k) * torch.cos(2 * torch.pi * torch.pow(b, k) * (z + 0.5)), -1)
            sum2 += torch.pow(a, k) * torch.cos(2 * torch.pi * torch.pow(b, k) * 0.5)
        return sum1 - self.dim * sum2 + self.bias
    
class Schwefel(CEC2017_Basic_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        super().__init__(dim, shift, rotate, bias)
        self.lb = -500
        self.ub = 500

    def func(self, x):
        z = self.sr_func(x, self.shift, self.rotate)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        g = z * torch.sin(torch.sqrt(torch.abs(z)))
        return b * self.dim - torch.sum(g,-1)